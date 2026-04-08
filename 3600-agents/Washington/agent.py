import os
import random
import sys

from collections.abc import Callable

from game import board, enums

NOISE = {
    enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
    enums.Cell.SPACE: (0.7, 0.15, 0.15),
    enums.Cell.PRIMED: (0.1, 0.8, 0.1),
    enums.Cell.CARPET: (0.1, 0.1, 0.8),
}
DIST = ((0.12, -1), (0.7, 0), (0.12, 1), (0.06, 2))
DIRS = (
    enums.Direction.UP,
    enums.Direction.DOWN,
    enums.Direction.LEFT,
    enums.Direction.RIGHT,
)
BS = enums.BOARD_SIZE
NC = BS * BS
EPS = 1e-12
INF = 10**18


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        try:
            self.T = transition_matrix.tolist()
        except Exception:
            self.T = None

        self.base_prior = self._compute_spawn_prior()
        self.belief = self.base_prior[:]
        self.turn = 0
        self._last_search_sig = None

        self._prev_opp = None
        self._prev_carpet_mask = None
        self._opp_carpet_dir = None
        self._opp_carpet_run = 0
        self._opp_pattern_age = 999
        self._block_target = None

        self._failed_searches = 0
        self._search_cooldown = 0

    def commentate(self):
        return ""

    def play(self, board: board.Board, sensor_data, time_left: Callable):
        noise_obs, dist_obs = sensor_data

        self._apply_search_evidence(board)
        self._predict_one_step()
        self._observe(board, noise_obs, dist_obs)
        self.turn += 1

        self._detect_opp_pattern(board)
        self._block_target = self._find_block_target(board)
        if self._search_cooldown > 0:
            self._search_cooldown -= 1

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        high_value = [
            m
            for m in moves
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            and enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0) >= 10
        ]
        if high_value:
            return max(
                high_value,
                key=lambda m: enums.CARPET_POINTS_TABLE.get(
                    getattr(m, "roll_length", 0), 0
                ),
            )

        search_move, search_prob = self._best_search(moves)
        non_search_moves = [
            m for m in moves if getattr(m, "move_type", None) != enums.MoveType.SEARCH
        ]
        scored_non_search = []
        for move in non_search_moves:
            scored_non_search.append((self._score_move_fast(board, move), move))
        scored_non_search.sort(key=lambda item: item[0], reverse=True)

        best_non_search_score = scored_non_search[0][0] if scored_non_search else -INF
        candidates = [move for _, move in scored_non_search[: self._beam_width(board)]]

        if search_move is not None and self._should_search(
            board, search_prob, best_non_search_score
        ):
            candidates.append(search_move)

        if not candidates:
            return random.choice(moves)

        best_move = candidates[0]
        best_value = -INF
        max_depth = self._max_depth(board, time_left)
        for depth in range(1, max_depth + 1):
            current_best_move = best_move
            current_best_value = -INF
            for move in candidates:
                value = self._root_value(board, move, depth)
                if value > current_best_value:
                    current_best_value = value
                    current_best_move = move
            best_move = current_best_move
            best_value = current_best_value

            if max_depth >= 3 and self._depth_margin_clear(
                candidates, board, depth, best_value
            ):
                break

        return best_move

    def _root_value(self, board, move, depth):
        if move.move_type == enums.MoveType.SEARCH:
            loc = self._search_target(move)
            if loc is None:
                return -INF
            prob = self._belief_at(loc)
            return self._search_value(board, prob)

        forecast = self._forecast(board, move)
        if forecast is None:
            return -INF
        if depth <= 1:
            return self._evaluate_board(forecast)

        forecast.reverse_perspective()
        return self._search_tree(forecast, depth - 1, False, -INF, INF)

    def _search_tree(self, board, depth, maximizing, alpha, beta):
        if depth <= 0:
            score = self._evaluate_board(board)
            return score if maximizing else -score

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            score = self._evaluate_board(board)
            return score if maximizing else -score

        ordered = []
        for move in moves:
            ordered.append((self._score_move_fast(board, move), move))
        ordered.sort(key=lambda item: item[0], reverse=True)
        beam = self._beam_width_for_depth(depth)
        candidates = [move for _, move in ordered[:beam]]

        if maximizing:
            value = -INF
            for move in candidates:
                child = self._forecast(board, move)
                if child is None:
                    continue
                child.reverse_perspective()
                value = max(
                    value, self._search_tree(child, depth - 1, False, alpha, beta)
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = INF
        for move in candidates:
            child = self._forecast(board, move)
            if child is None:
                continue
            child.reverse_perspective()
            value = min(value, self._search_tree(child, depth - 1, True, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _depth_margin_clear(self, candidates, board, depth, best_value):
        if depth < 2 or len(candidates) < 2:
            return False
        second = -INF
        for move in candidates[:2]:
            value = self._root_value(board, move, depth)
            if value > second and value < best_value + EPS:
                second = value
        return best_value - second > 5.0

    def _score_move_fast(self, board, move):
        move_type = move.move_type
        me = self._my_loc(board)
        end = self._end_loc(me, move)

        future_carpet = self._future_carpet(board, end)
        two_ply = self._two_ply_carpet_value(board, me, move)
        adjacent_space = self._adjacent_space(board, end)
        mobility = self._mobility(board, end)
        opponent_pressure = self._opponent_pressure(board, end)
        local1 = self._local_mass(end, 1)
        local2 = self._local_mass(end, 2)
        expected_dist = self._expected_distance(end)
        top1 = max(self.belief) if self.belief else 0.0
        top3 = self._top_mass(3)
        hot = top1 >= 0.28 or top3 >= 0.60
        if hot:
            rat_bonus = 0.45 * local1 + 0.20 * local2 - 0.08 * expected_dist
        else:
            rat_bonus = 0.08 * local1 - 0.01 * expected_dist

        center_penalty = 0.035 * self._manhattan(end, (BS // 2, BS // 2))
        block_bonus = self._block_bonus(board, end)
        control_bonus = self._control_bonus(board, end)
        chain_bonus = self._chain_shape_bonus(board, end)
        turns_left = self._turns_remaining(board)

        if move_type == enums.MoveType.CARPET:
            pts = enums.CARPET_POINTS_TABLE.get(getattr(move, "roll_length", 0), 0)
            score = (
                3.05 * pts
                + 0.16 * future_carpet
                + 0.12 * two_ply
                + 0.06 * adjacent_space
                + 0.25 * rat_bonus
                + 0.12 * mobility
                + 0.10 * control_bonus
                - center_penalty
            )
            if pts <= 1 and future_carpet >= pts + 3:
                score -= 4.5
            elif pts <= 0:
                score -= 2.5
            elif pts == 2:
                score -= 0.35
            return score + block_bonus

        if move_type == enums.MoveType.PRIME:
            score = (
                1.85
                + 0.86 * future_carpet
                + 0.62 * two_ply
                + 0.22 * adjacent_space
                + 0.22 * rat_bonus
                + 0.14 * mobility
                + 0.22 * chain_bonus
                + 0.16 * control_bonus
                - center_penalty
            )
            if turns_left <= 10:
                score += 0.30
            return score + block_bonus

        if move_type == enums.MoveType.PLAIN:
            score = (
                -0.90
                + 0.76 * future_carpet
                + 0.35 * two_ply
                + 0.16 * adjacent_space
                + 0.50 * rat_bonus
                + 0.18 * mobility
                + 0.14 * control_bonus
                - center_penalty
            )
            if turns_left <= 8:
                score += 0.25
            return score + 0.55 * block_bonus

        return -INF

    def _evaluate_board(self, board):
        my_points = board.player_worker.get_points()
        opp_points = board.opponent_worker.get_points()
        score = 2.3 * (my_points - opp_points)

        my_loc = self._my_loc(board)
        opp_loc = board.opponent_worker.get_location()
        score += 0.20 * self._mobility(board, my_loc)
        score -= 0.16 * self._mobility_enemy(board)
        score += 0.12 * self._future_carpet(board, my_loc)
        score -= 0.10 * self._enemy_future_carpet(board)
        score += 0.14 * self._chain_shape_bonus(board, my_loc)
        score -= 0.10 * self._distance_to_edge_frontier(my_loc, opp_loc)
        score += 0.06 * self._local_mass(my_loc, 2)
        score -= 0.03 * self._manhattan(my_loc, opp_loc)
        if self._block_target is not None:
            score -= 0.10 * self._manhattan(my_loc, self._block_target)
        return score

    def _best_search(self, moves):
        best_move = None
        best_prob = -1.0
        for move in moves:
            if getattr(move, "move_type", None) != enums.MoveType.SEARCH:
                continue
            loc = self._search_target(move)
            if loc is None:
                continue
            prob = self._belief_at(loc)
            if prob > best_prob:
                best_prob = prob
                best_move = move
        return best_move, best_prob

    def _should_search(self, board, prob, best_non_search_score):
        if self._search_cooldown > 0:
            return False

        turns_left = self._turns_remaining(board)
        top3 = self._top_mass(3)
        top5 = self._top_mass(5)
        ev = self._search_value(board, prob)
        if ev <= 0.6:
            return False

        penalty = min(self._failed_searches * 0.06, 0.24)
        if turns_left >= 28:
            threshold = 0.65
        elif turns_left >= 13:
            threshold = 0.58
        else:
            threshold = 0.50
        threshold -= penalty

        diff = board.player_worker.get_points() - board.opponent_worker.get_points()
        if diff >= 6:
            threshold += 0.06
        elif diff <= -6:
            threshold -= 0.04

        if ev < 0.55 * best_non_search_score:
            return False
        if prob >= threshold:
            return True
        if turns_left <= 8 and prob >= threshold - 0.06:
            return True
        if turns_left <= 4 and prob >= threshold - 0.10:
            return True
        if top3 >= 0.82 and prob >= threshold - 0.04:
            return True
        if top5 >= 0.90 and prob >= threshold - 0.06:
            return True
        return False

    def _search_value(self, board, prob):
        ev = 4.0 * prob - 2.0 * (1.0 - prob)
        diff = board.player_worker.get_points() - board.opponent_worker.get_points()
        if diff <= -6:
            ev += 0.20
        elif diff >= 6:
            ev -= 0.15
        return ev

    def _compute_spawn_prior(self):
        if self.T is None:
            return [1.0 / NC] * NC
        belief = [0.0] * NC
        belief[0] = 1.0
        for _ in range(1000):
            belief = self._matvec(belief)
        self._normalize(belief)
        return belief

    def _predict_one_step(self):
        if self.T is None:
            return
        self.belief = self._matvec(self.belief)
        self._normalize(self.belief)

    def _observe(self, board, noise_obs, reported_dist):
        me = self._my_loc(board)
        for idx in range(NC):
            x, y = idx % BS, idx // BS
            noise_likelihood = NOISE.get(
                board.get_cell((x, y)), NOISE[enums.Cell.SPACE]
            )[noise_obs.value]
            actual_dist = abs(x - me[0]) + abs(y - me[1])
            dist_likelihood = 0.0
            for prob, offset in DIST:
                if max(0, actual_dist + offset) == reported_dist:
                    dist_likelihood += prob
            self.belief[idx] *= noise_likelihood * dist_likelihood

        if sum(self.belief) <= EPS:
            self.belief = self.base_prior[:]
        else:
            self._normalize(self.belief)

    def _apply_search_evidence(self, board):
        my_info = self._extract_search(
            board,
            [
                "player_search",
                "your_search_location_and_result",
                "player_search_location_and_result",
                "my_search_location_and_result",
                "your_search_info",
                "player_search_info",
            ],
        )
        opp_info = self._extract_search(
            board,
            [
                "opponent_search",
                "opponent_search_location_and_result",
                "enemy_search_location_and_result",
                "opp_search_location_and_result",
                "opponent_search_info",
            ],
        )

        signature = (my_info, opp_info)
        if signature == self._last_search_sig:
            return
        self._last_search_sig = signature

        for loc, success in (my_info, opp_info):
            if loc is not None and success:
                self.belief = self.base_prior[:]
                self._failed_searches = 0
                self._search_cooldown = 0
                return

        changed = False
        for loc, success in (my_info, opp_info):
            if loc is None or success:
                continue
            idx = self._idx(loc)
            if 0 <= idx < NC and self.belief[idx] > 0.0:
                self.belief[idx] = 0.0
                changed = True
                self._failed_searches += 1
                self._search_cooldown = 3

        if changed:
            if sum(self.belief) <= EPS:
                self.belief = self.base_prior[:]
            else:
                self._normalize(self.belief)

    def _extract_search(self, board, names):
        for name in names:
            if not hasattr(board, name):
                continue
            value = getattr(board, name)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, tuple) and len(value) == 2:
                return value
        return (None, False)

    def _detect_opp_pattern(self, board):
        opp = board.opponent_worker.get_location()
        cur_carpet = getattr(board, "_carpet_mask", None)

        if self._prev_opp is None or cur_carpet is None:
            self._prev_opp = opp
            self._prev_carpet_mask = cur_carpet
            return

        new_carpet = 0
        if self._prev_carpet_mask is not None:
            new_carpet = cur_carpet & ~self._prev_carpet_mask

        if new_carpet:
            direction, length = self._extract_carpet_info(self._prev_opp, new_carpet)
            if direction is not None:
                self._opp_carpet_dir = direction
                self._opp_carpet_run = length
                self._opp_pattern_age = 0
            else:
                self._opp_pattern_age += 1
        else:
            self._opp_pattern_age += 1

        if self._opp_pattern_age > 4:
            self._opp_carpet_dir = None
            self._opp_carpet_run = 0

        self._prev_opp = opp
        self._prev_carpet_mask = cur_carpet

    def _extract_carpet_info(self, start_pos, new_carpet_mask):
        best_direction = None
        best_length = 0
        for direction in DIRS:
            mask = 1 << (start_pos[1] * BS + start_pos[0])
            length = 0
            for _ in range(BS - 1):
                mask = self._shift_mask(direction, mask)
                if not mask:
                    break
                if mask & new_carpet_mask:
                    length += 1
                else:
                    break
            if length > best_length:
                best_length = length
                best_direction = direction
        if best_length > 0:
            return best_direction, best_length
        return None, 0

    def _find_block_target(self, board):
        if self._opp_carpet_dir is None or self._opp_pattern_age > 2:
            return None
        cur = board.opponent_worker.get_location()
        for _ in range(BS - 1):
            cur = enums.loc_after_direction(cur, self._opp_carpet_dir)
            if not self._in_bounds(cur):
                break
            cell = board.get_cell(cur)
            if cell == enums.Cell.BLOCKED:
                break
            if cell == enums.Cell.SPACE:
                return cur
        return None

    def _future_carpet(self, board, loc):
        best = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not self._in_bounds(nxt) or board.get_cell(nxt) != enums.Cell.PRIMED:
                continue
            run = 0
            cur = nxt
            while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
                run += 1
                cur = enums.loc_after_direction(cur, direction)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _enemy_future_carpet(self, board):
        opp = board.opponent_worker.get_location()
        best = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(opp, direction)
            if not self._in_bounds(nxt) or board.get_cell(nxt) != enums.Cell.PRIMED:
                continue
            run = 0
            cur = nxt
            while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
                if cur == self._my_loc(board):
                    break
                run += 1
                cur = enums.loc_after_direction(cur, direction)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _two_ply_carpet_value(self, board, start, move):
        if getattr(move, "move_type", None) == enums.MoveType.SEARCH:
            return 0.0
        forecast = self._forecast(board, move)
        if forecast is None:
            return 0.0
        end = self._end_loc(start, move)
        return self._future_carpet(forecast, end)

    def _adjacent_space(self, board, loc):
        total = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if self._in_bounds(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                if nxt != board.opponent_worker.get_location():
                    total += 1
        return total

    def _mobility(self, board, loc):
        total = 0
        opp = board.opponent_worker.get_location()
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not self._in_bounds(nxt):
                continue
            if nxt == opp:
                continue
            cell = board.get_cell(nxt)
            if cell == enums.Cell.SPACE or cell == enums.Cell.PRIMED:
                total += 1
        return total

    def _mobility_enemy(self, board):
        total = 0
        opp = board.opponent_worker.get_location()
        me = self._my_loc(board)
        for direction in DIRS:
            nxt = enums.loc_after_direction(opp, direction)
            if not self._in_bounds(nxt):
                continue
            if nxt == me:
                continue
            cell = board.get_cell(nxt)
            if cell == enums.Cell.SPACE or cell == enums.Cell.PRIMED:
                total += 1
        return total

    def _opponent_pressure(self, board, loc):
        return 1.0 / (1.0 + self._manhattan(loc, board.opponent_worker.get_location()))

    def _chain_shape_bonus(self, board, loc):
        best = 0
        for direction in DIRS:
            best = max(best, self._run_length(board, loc, direction))
        return best

    def _run_length(self, board, loc, direction):
        total = 0
        cur = enums.loc_after_direction(loc, direction)
        while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
            total += 1
            cur = enums.loc_after_direction(cur, direction)

        opposite = {
            enums.Direction.UP: enums.Direction.DOWN,
            enums.Direction.DOWN: enums.Direction.UP,
            enums.Direction.LEFT: enums.Direction.RIGHT,
            enums.Direction.RIGHT: enums.Direction.LEFT,
        }[direction]
        cur = enums.loc_after_direction(loc, opposite)
        while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
            total += 1
            cur = enums.loc_after_direction(cur, opposite)
        return total

    def _control_bonus(self, board, loc):
        opp = board.opponent_worker.get_location()
        bonus = 0.0
        dist = self._manhattan(loc, opp)
        if dist <= 2:
            bonus += 0.8 / (dist + 1)
        frontier = self._edge_frontier(opp)
        if frontier is not None:
            bonus += max(0.0, 0.6 - 0.1 * self._manhattan(loc, frontier))
        return bonus

    def _block_bonus(self, board, loc):
        if self._block_target is None or self._opp_carpet_dir is None:
            return 0.0
        if self._opp_pattern_age > 2 or loc != self._block_target:
            return 0.0
        dist = self._manhattan(loc, board.opponent_worker.get_location())
        bonus = 1.5 + 0.12 * min(self._opp_carpet_run, 4)
        if dist <= 2:
            bonus += 0.6
        elif dist <= 4:
            bonus += 0.25
        return bonus

    def _distance_to_edge_frontier(self, loc, opp):
        frontier = self._edge_frontier(opp)
        if frontier is None:
            return 0
        return self._manhattan(loc, frontier)

    def _edge_frontier(self, opp):
        x, y = opp
        candidates = []
        if x <= 2:
            candidates.append((x - 1, y))
        if x >= BS - 3:
            candidates.append((x + 1, y))
        if y <= 2:
            candidates.append((x, y - 1))
        if y >= BS - 3:
            candidates.append((x, y + 1))
        for loc in candidates:
            if self._in_bounds(loc):
                return loc
        return None

    def _local_mass(self, center, radius):
        total = 0.0
        for y in range(BS):
            for x in range(BS):
                if abs(x - center[0]) + abs(y - center[1]) <= radius:
                    total += self.belief[self._idx((x, y))]
        return total

    def _expected_distance(self, loc):
        total = 0.0
        for idx, prob in enumerate(self.belief):
            if prob <= 0.0:
                continue
            x, y = idx % BS, idx // BS
            total += prob * self._manhattan(loc, (x, y))
        return total

    def _top_mass(self, k):
        return sum(sorted(self.belief, reverse=True)[:k])

    def _belief_at(self, loc):
        if not self._in_bounds(loc):
            return 0.0
        return self.belief[self._idx(loc)]

    def _matvec(self, vec):
        if self.T is None:
            return [1.0 / NC] * NC
        out = [0.0] * NC
        for i, bi in enumerate(vec):
            if bi <= EPS:
                continue
            row = self.T[i]
            for j, pij in enumerate(row):
                if pij > 0.0:
                    out[j] += bi * pij
        return out

    def _normalize(self, vec):
        total = sum(vec)
        if total <= EPS:
            return
        inv = 1.0 / total
        for i in range(len(vec)):
            vec[i] *= inv

    def _forecast(self, board, move):
        if not hasattr(board, "forecast_move"):
            return None
        try:
            return board.forecast_move(move)
        except Exception:
            return None

    def _beam_width(self, board):
        turns_left = self._turns_remaining(board)
        if turns_left >= 25:
            return 8
        if turns_left >= 10:
            return 7
        return 6

    def _beam_width_for_depth(self, depth):
        if depth >= 3:
            return 4
        if depth == 2:
            return 5
        return 6

    def _max_depth(self, board, time_left):
        remaining = self._safe_time_left(time_left)
        turns_left = self._turns_remaining(board)
        if remaining >= 120 and turns_left <= 20:
            return 4
        if remaining >= 45:
            return 3
        return 2

    def _safe_time_left(self, time_left):
        if callable(time_left):
            try:
                return float(time_left())
            except Exception:
                return 60.0
        return 60.0

    def _my_loc(self, board):
        worker = board.player_worker
        if hasattr(worker, "get_location"):
            return worker.get_location()
        return worker.location

    def _turns_remaining(self, board):
        worker = board.player_worker
        for attr in ("turns_remaining", "turns_left"):
            if hasattr(worker, attr):
                return getattr(worker, attr)
        return 40

    def _end_loc(self, start, move):
        direction = getattr(move, "direction", None)
        if direction is None:
            return start
        steps = (
            getattr(move, "roll_length", 1)
            if move.move_type == enums.MoveType.CARPET
            else 1
        )
        loc = start
        for _ in range(steps):
            loc = enums.loc_after_direction(loc, direction)
        return loc

    def _search_target(self, move):
        for attr in (
            "search_loc",
            "location",
            "target",
            "search_location",
            "guess_location",
            "loc",
        ):
            if not hasattr(move, attr):
                continue
            value = getattr(move, attr)
            if isinstance(value, tuple) and len(value) == 2:
                return value
        return None

    def _idx(self, loc):
        return loc[1] * BS + loc[0]

    def _in_bounds(self, loc):
        return 0 <= loc[0] < BS and 0 <= loc[1] < BS

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _shift_mask(self, direction, mask):
        if direction == enums.Direction.UP:
            return (mask >> BS) & 0x00FFFFFFFFFFFFFF
        if direction == enums.Direction.DOWN:
            return (mask << BS) & 0xFFFFFFFFFFFFFF00
        if direction == enums.Direction.LEFT:
            return (mask >> 1) & 0x7F7F7F7F7F7F7F7F
        if direction == enums.Direction.RIGHT:
            return (mask << 1) & 0xFEFEFEFEFEFEFEFE
        return 0
