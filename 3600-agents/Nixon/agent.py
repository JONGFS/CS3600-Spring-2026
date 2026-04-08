import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

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
    """
    Blitz / denial hybrid:
    - HMM belief tracking with search evidence
    - Search only when the signal is clean
    - Cashes 5+ carpets immediately, 4s more aggressively
    - Builds prime lanes greedily
    - Contests opponent launch/frontier cells
    - Keeps lightweight lookahead from the stronger baseline
    """

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

        opp_cash = self._opponent_best_carpet_points(board)
        turns_left = self._turns_remaining(board)

        smash5 = [
            m
            for m in moves
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            and getattr(m, "roll_length", 0) >= 5
        ]
        if smash5:
            return max(
                smash5,
                key=lambda m: (
                    enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0),
                    self._score_move_fast(board, m),
                ),
            )

        smash4 = [
            m
            for m in moves
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            and getattr(m, "roll_length", 0) >= 4
        ]
        if smash4 and (opp_cash >= 6 or turns_left <= 18):
            return max(
                smash4,
                key=lambda m: (
                    enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0),
                    self._score_move_fast(board, m),
                ),
            )

        search_move, search_prob = self._best_search(moves)
        non_search_moves = [
            m for m in moves if getattr(m, "move_type", None) != enums.MoveType.SEARCH
        ]

        scored_non_search = [
            (self._score_move_fast(board, move), move)
            for move in non_search_moves
        ]
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
        opp = board.opponent_worker.get_location()
        end = self._end_loc(me, move)

        future_carpet = self._future_carpet(board, end)
        two_ply = self._two_ply_carpet_value(board, me, move)
        mobility = self._mobility(board, end)
        adjacent_space = self._adjacent_space(board, end)
        center_penalty = 0.03 * self._manhattan(end, (BS // 2, BS // 2))

        opp_cash = self._opponent_best_carpet_points(board)
        steal = self._steal_bonus(board, end)
        block_bonus = self._block_bonus(board, end)

        axis_bonus = self._axis_bias(board, end, blocker=opp)
        dir_bonus = self._direction_axis_bonus(board, me, move, blocker=opp)

        if move_type == enums.MoveType.CARPET:
            k = getattr(move, "roll_length", 0)
            pts = enums.CARPET_POINTS_TABLE.get(k, 0)

            score = (
                5.6 * pts
                + 0.45 * future_carpet
                + 0.18 * two_ply
                + 0.12 * mobility
                + 0.06 * adjacent_space
                + 0.90 * steal
                + 0.20 * axis_bonus
                - center_penalty
            )

            if k >= 5:
                score += 12.0
            elif k == 4:
                score += 6.5
            elif k == 3 and (opp_cash >= 6 or self._turns_remaining(board) <= 12):
                score += 2.5

            if pts <= 1:
                score -= 40.0
            elif pts == 2 and future_carpet >= 6:
                score -= 2.0

            if opp_cash >= 10 and pts >= 4:
                score += 3.0

            return score + block_bonus

        if move_type == enums.MoveType.PRIME:
            direction = getattr(move, "direction", None)
            lane = self._prime_lane_after(board, me, direction)

            nxt = enums.loc_after_direction(end, direction)
            forward = self._space_run(board, nxt, direction, blocker=opp)
            lane_pts = enums.CARPET_POINTS_TABLE.get(lane, 0)

            score = (
                2.1
                + 4.0 * lane
                + 1.7 * forward
                + 1.3 * max(future_carpet, lane_pts)
                + 0.32 * two_ply
                + 0.20 * mobility
                + 0.10 * adjacent_space
                + 0.85 * steal
                + 0.55 * axis_bonus
                + 1.10 * dir_bonus
                - center_penalty
            )

            if lane >= 5:
                score += 12.0
            elif lane == 4:
                score += 5.0
            elif lane == 3:
                score += 1.5

            if opp_cash >= 10:
                score += 1.5

            return score + block_bonus

        if move_type == enums.MoveType.PLAIN:
            launch = self._best_prime_lane_from(board, end, blocker=opp)

            score = (
                -1.2
                + 2.9 * launch
                + 0.8 * future_carpet
                + 0.18 * mobility
                + 0.10 * adjacent_space
                + 1.00 * steal
                + 0.75 * axis_bonus
                + 0.55 * dir_bonus
                - center_penalty
            )

            if opp_cash >= 10:
                score += 1.0

            return score + 0.6 * block_bonus

        return -INF

    def _evaluate_board(self, board):
        my_points = board.player_worker.get_points()
        opp_points = board.opponent_worker.get_points()

        my_loc = self._my_loc(board)
        opp_loc = board.opponent_worker.get_location()

        my_cash = self._future_carpet(board, my_loc)
        opp_cash = self._opponent_best_carpet_points(board)

        my_axis = self._axis_bias(board, my_loc, blocker=opp_loc)
        my_launch = self._best_prime_lane_from(board, my_loc, blocker=opp_loc)
        opp_launch = self._best_prime_lane_from(board, opp_loc, blocker=my_loc)

        score = 2.8 * (my_points - opp_points)
        score += 1.05 * my_launch
        score += 0.55 * my_cash
        score -= 0.95 * opp_cash
        score += 0.22 * my_axis
        score -= 0.65 * opp_launch
        score += 0.18 * self._mobility(board, my_loc)
        score -= 0.12 * self._mobility_enemy(board)
        score += 0.12 * self._steal_bonus(board, my_loc)
        score += 0.04 * self._local_mass(my_loc, 2)
        score -= 0.04 * self._manhattan(my_loc, opp_loc)

        if self._block_target is not None:
            score -= 0.08 * self._manhattan(my_loc, self._block_target)

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
        ev = self._search_value(board, prob)

        if turns_left > 20:
            threshold = 0.76
        elif turns_left > 10:
            threshold = 0.70
        elif turns_left > 5:
            threshold = 0.62
        else:
            threshold = 0.56

        if board.player_worker.get_points() < board.opponent_worker.get_points():
            threshold -= 0.03

        if self._opponent_best_carpet_points(board) >= 10 and turns_left > 6:
            threshold += 0.03

        if ev < max(1.7, 0.32 * best_non_search_score):
            return False

        return prob >= threshold

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
        opp = board.opponent_worker.get_location()
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if self._in_bounds(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                if nxt != opp:
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

    def _opposite_dir(self, direction):
        if direction == enums.Direction.UP:
            return enums.Direction.DOWN
        if direction == enums.Direction.DOWN:
            return enums.Direction.UP
        if direction == enums.Direction.LEFT:
            return enums.Direction.RIGHT
        if direction == enums.Direction.RIGHT:
            return enums.Direction.LEFT
        return None

    def _prime_lane_after(self, board, start, direction):
        if direction is None:
            return 0

        back = self._opposite_dir(direction)
        length = 1
        cur = start

        while True:
            cur = enums.loc_after_direction(cur, back)
            if not self._in_bounds(cur):
                break
            if board.get_cell(cur) != enums.Cell.PRIMED:
                break
            length += 1

        return length

    def _space_run(self, board, start, direction, blocker=None):
        if direction is None or not self._in_bounds(start):
            return 0
        if blocker is not None and start == blocker:
            return 0
        if board.get_cell(start) != enums.Cell.SPACE:
            return 0

        run = 0
        cur = start
        while self._in_bounds(cur):
            if blocker is not None and cur == blocker:
                break
            if board.get_cell(cur) != enums.Cell.SPACE:
                break
            run += 1
            cur = enums.loc_after_direction(cur, direction)
        return run

    def _best_prime_lane_from(self, board, loc, blocker=None):
        best = 0.0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not self._in_bounds(nxt):
                continue
            if blocker is not None and nxt == blocker:
                continue
            if board.get_cell(nxt) != enums.Cell.SPACE:
                continue

            lane = self._prime_lane_after(board, loc, direction)
            forward = self._space_run(board, nxt, direction, blocker=blocker)
            best = max(best, lane + 0.5 * forward)

        return best

    def _best_carpet_points_from(self, board, loc, blocker=None):
        best = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not self._in_bounds(nxt):
                continue
            if blocker is not None and nxt == blocker:
                continue
            if board.get_cell(nxt) != enums.Cell.PRIMED:
                continue

            run = 0
            cur = nxt
            while self._in_bounds(cur):
                if blocker is not None and cur == blocker:
                    break
                if board.get_cell(cur) != enums.Cell.PRIMED:
                    break
                run += 1
                cur = enums.loc_after_direction(cur, direction)

            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))

        return best

    def _opponent_best_carpet_points(self, board):
        return self._best_carpet_points_from(
            board,
            board.opponent_worker.get_location(),
            blocker=self._my_loc(board),
        )

    def _opp_prime_frontier(self, board):
        opp = board.opponent_worker.get_location()
        me = self._my_loc(board)

        best_cell = None
        best_score = -INF

        for direction in DIRS:
            nxt = enums.loc_after_direction(opp, direction)
            if not self._in_bounds(nxt):
                continue
            if nxt == me:
                continue
            if board.get_cell(nxt) != enums.Cell.SPACE:
                continue

            lane = self._prime_lane_after(board, opp, direction)
            forward = self._space_run(board, nxt, direction, blocker=me)
            score = 3.0 * lane + forward

            if score > best_score:
                best_score = score
                best_cell = nxt

        return best_cell

    def _steal_bonus(self, board, loc):
        bonus = 0.0

        frontier = self._opp_prime_frontier(board)
        if frontier is not None:
            d = self._manhattan(loc, frontier)
            if d == 0:
                bonus += 3.0
            elif d == 1:
                bonus += 1.0

        if self._block_target is not None and loc == self._block_target:
            bonus += 2.5

        return bonus
    
    def _axis_open_lengths(self, board, loc, blocker=None):
        """Open SPACE runway by axis from loc's neighbors."""
        x_len = 0
        y_len = 0

        # horizontal
        for direction in (enums.Direction.LEFT, enums.Direction.RIGHT):
            cur = enums.loc_after_direction(loc, direction)
            while self._in_bounds(cur):
                if blocker is not None and cur == blocker:
                    break
                if board.get_cell(cur) != enums.Cell.SPACE:
                    break
                x_len += 1
                cur = enums.loc_after_direction(cur, direction)

        # vertical
        for direction in (enums.Direction.UP, enums.Direction.DOWN):
            cur = enums.loc_after_direction(loc, direction)
            while self._in_bounds(cur):
                if blocker is not None and cur == blocker:
                    break
                if board.get_cell(cur) != enums.Cell.SPACE:
                    break
                y_len += 1
                cur = enums.loc_after_direction(cur, direction)

        return x_len, y_len

    def _axis_primed_lengths(self, board, loc, blocker=None):
        """Existing PRIMED lane by axis from loc's neighbors."""
        x_len = 0
        y_len = 0

        for direction in (enums.Direction.LEFT, enums.Direction.RIGHT):
            cur = enums.loc_after_direction(loc, direction)
            while self._in_bounds(cur):
                if blocker is not None and cur == blocker:
                    break
                if board.get_cell(cur) != enums.Cell.PRIMED:
                    break
                x_len += 1
                cur = enums.loc_after_direction(cur, direction)

        for direction in (enums.Direction.UP, enums.Direction.DOWN):
            cur = enums.loc_after_direction(loc, direction)
            while self._in_bounds(cur):
                if blocker is not None and cur == blocker:
                    break
                if board.get_cell(cur) != enums.Cell.PRIMED:
                    break
                y_len += 1
                cur = enums.loc_after_direction(cur, direction)

        return x_len, y_len

    def _axis_bias(self, board, loc, blocker=None):
        """
        Positive when the square sits on a long dominant axis.
        Rewards corridors, especially if one axis is clearly better.
        """
        open_x, open_y = self._axis_open_lengths(board, loc, blocker=blocker)
        primed_x, primed_y = self._axis_primed_lengths(board, loc, blocker=blocker)

        x_total = open_x + 1.4 * primed_x
        y_total = open_y + 1.4 * primed_y

        dominant = max(x_total, y_total)
        imbalance = abs(x_total - y_total)

        return dominant + 0.35 * imbalance

    def _direction_runway(self, board, loc, direction, blocker=None):
        """Runway only in the chosen direction."""
        cur = enums.loc_after_direction(loc, direction)
        run = 0
        while self._in_bounds(cur):
            if blocker is not None and cur == blocker:
                break
            if board.get_cell(cur) != enums.Cell.SPACE:
                break
            run += 1
            cur = enums.loc_after_direction(cur, direction)
        return run

    def _direction_axis_bonus(self, board, start, move, blocker=None):
        """
        Bonus for moving/priming along the longer board axis.
        """
        direction = getattr(move, "direction", None)
        if direction is None:
            return 0.0

        end = self._end_loc(start, move)
        runway = self._direction_runway(board, end, direction, blocker=blocker)

        if direction in (enums.Direction.LEFT, enums.Direction.RIGHT):
            axis_open, other_open = self._axis_open_lengths(board, end, blocker=blocker)
        else:
            other_open, axis_open = self._axis_open_lengths(board, end, blocker=blocker)

        return 0.8 * runway + 0.45 * max(0, axis_open - other_open)