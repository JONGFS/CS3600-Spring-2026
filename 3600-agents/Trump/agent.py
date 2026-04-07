from collections.abc import Callable
import random

from game import board, enums

NOISE_PROBS = {
    enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
    enums.Cell.SPACE: (0.7, 0.15, 0.15),
    enums.Cell.PRIMED: (0.1, 0.8, 0.1),
    enums.Cell.CARPET: (0.1, 0.1, 0.8),
}

DIST_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)
DIST_OFFSETS = (-1, 0, 1, 2)

BS = enums.BOARD_SIZE
NC = BS * BS
EPS = 1e-12


class PlayerAgent:
    """
    Kennedy:
    - board-first hybrid
    - HMM rat tracking with corrected spawn/reset handling
    - search only when belief is concentrated
    - lightweight opponent carpet-pattern detection
    - defensive blocking-prime bonus against continued carpet lanes
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        try:
            self.T = transition_matrix.tolist()
        except Exception:
            self.T = None

        self.base_prior = self._compute_spawn_prior()
        self.belief = self.base_prior[:]
        self.turn = 0
        self._last_search_signature = None

        # Opponent-pattern tracking
        self._prev_opponent_pos = None
        self._prev_carpet_mask = None
        self._opponent_carpet_direction = None
        self._opponent_carpet_run_length = 0
        self._opponent_pattern_age = 999
        self._blocking_prime_target = None

    def commentate(self):
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data,
        time_left: Callable,
    ):
        noise_obs, dist_obs = sensor_data

        # Belief update
        self._apply_search_evidence(board)
        self._predict_one_step()
        self._observe(board, noise_obs, dist_obs)
        self.turn += 1

        # Opponent-pattern update
        self._detect_opponent_pattern(board)
        self._blocking_prime_target = self._find_blocking_prime_target(board)

        all_moves = board.get_valid_moves(exclude_search=False)
        if not all_moves:
            return None

        search_moves = [
            m
            for m in all_moves
            if getattr(m, "move_type", None) == enums.MoveType.SEARCH
        ]
        non_search_moves = [
            m
            for m in all_moves
            if getattr(m, "move_type", None) != enums.MoveType.SEARCH
        ]

        high_value_carpets = [
            m
            for m in non_search_moves
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            and enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0) >= 10
        ]
        if high_value_carpets:
            return max(
                high_value_carpets,
                key=lambda m: enums.CARPET_POINTS_TABLE.get(
                    getattr(m, "roll_length", 0), 0
                ),
            )

        best_non_search = None
        best_non_search_score = float("-inf")
        for m in non_search_moves:
            s = self._score_non_search(board, m)
            if s > best_non_search_score:
                best_non_search_score = s
                best_non_search = m

        best_search = None
        best_search_prob = -1.0
        for m in search_moves:
            loc = self._search_target(m)
            if loc is None:
                continue
            p = self._belief_at(loc)
            if p > best_search_prob:
                best_search_prob = p
                best_search = m

        if best_search is not None and self._should_search(board, best_search_prob):
            return best_search

        if best_non_search is not None:
            return best_non_search

        return random.choice(all_moves)

    # ----------------------------
    # Search decision
    # ----------------------------

    def _should_search(self, board, best_search_prob):
        turns_left = self._turns_remaining(board)
        top3 = self._top_mass(3)
        top5 = self._top_mass(5)

        if best_search_prob >= 0.58:
            return True
        if turns_left <= 8 and best_search_prob >= 0.48:
            return True
        if top3 >= 0.72 and best_search_prob >= 0.50:
            return True
        if top5 >= 0.85 and best_search_prob >= 0.46:
            return True
        return False

    # ----------------------------
    # Non-search move scoring
    # ----------------------------

    def _score_non_search(self, board, m):
        move_type = getattr(m, "move_type", None)
        cur_loc = self._my_loc(board)
        end_loc = self._end_loc(cur_loc, m)

        next_board = None
        if hasattr(board, "forecast_move"):
            try:
                next_board = board.forecast_move(m)
            except Exception:
                next_board = None
        if next_board is None:
            next_board = board

        future_carpet = self._best_future_carpet_points(next_board, end_loc)
        adj_space = self._adjacent_space_count(next_board, end_loc)

        top1 = max(self.belief) if self.belief else 0.0
        top3 = self._top_mass(3)
        rat_mode = (top1 >= 0.28) or (top3 >= 0.60)

        local1 = self._local_mass(end_loc, 1)
        local2 = self._local_mass(end_loc, 2)
        exp_dist = self._expected_distance(end_loc)

        if rat_mode:
            rat_bonus = 0.45 * local1 + 0.20 * local2 - 0.08 * exp_dist
        else:
            rat_bonus = 0.10 * local1 - 0.02 * exp_dist

        center_penalty = 0.04 * self._manhattan(end_loc, (BS // 2, BS // 2))
        turns_left = self._turns_remaining(board)

        if move_type == enums.MoveType.CARPET:
            pts = enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0)
            score = (
                2.65 * pts
                + 0.10 * future_carpet
                + 0.05 * adj_space
                + 0.35 * rat_bonus
                - center_penalty
            )

            if pts <= 0:
                score -= 1.5
            elif pts == 2:
                score -= 0.2

            return score

        if move_type == enums.MoveType.PRIME:
            score = (
                1.55
                + 0.70 * future_carpet
                + 0.22 * adj_space
                + 0.25 * rat_bonus
                - center_penalty
            )

            # Defensive bonus: if we can prime the next likely square in the
            # opponent's carpet lane, reward that move.
            score += self._blocking_prime_bonus(board, end_loc)

            if turns_left <= 8:
                score += 0.20

            return score

        if move_type == enums.MoveType.PLAIN:
            score = (
                -0.75
                + 0.72 * future_carpet
                + 0.18 * adj_space
                + 0.60 * rat_bonus
                - center_penalty
            )

            # If pattern is active, slightly prefer repositioning toward the blocking square.
            score += self._blocking_plain_bonus(end_loc)

            if turns_left <= 6:
                score += 0.15

            return score

        return -9999.0

    def _blocking_prime_bonus(self, board, end_loc):
        if self._blocking_prime_target is None:
            return 0.0
        if self._opponent_carpet_direction is None:
            return 0.0
        if self._opponent_pattern_age > 2:
            return 0.0
        if end_loc != self._blocking_prime_target:
            return 0.0

        opp_pos = board.opponent_worker.get_location()
        d = self._manhattan(opp_pos, end_loc)

        bonus = 1.35
        if d <= 2:
            bonus += 0.55
        elif d <= 4:
            bonus += 0.25

        bonus += 0.10 * min(self._opponent_carpet_run_length, 4)
        return bonus

    def _blocking_plain_bonus(self, end_loc):
        if self._blocking_prime_target is None:
            return 0.0
        if self._opponent_pattern_age > 1:
            return 0.0

        # Mild positioning bias only; do not let this dominate board economy.
        d = self._manhattan(end_loc, self._blocking_prime_target)
        if d == 0:
            return 0.20
        if d == 1:
            return 0.10
        return 0.0

    # ----------------------------
    # Opponent pattern detection
    # ----------------------------

    def _detect_opponent_pattern(self, board):
        opp_pos = board.opponent_worker.get_location()
        cur_carpet = getattr(board, "_carpet_mask", None)

        if self._prev_opponent_pos is None or cur_carpet is None:
            self._prev_opponent_pos = opp_pos
            self._prev_carpet_mask = cur_carpet
            return

        new_carpet = 0
        if self._prev_carpet_mask is not None:
            new_carpet = cur_carpet & ~self._prev_carpet_mask

        if new_carpet:
            direction, length = self._extract_carpet_info(
                self._prev_opponent_pos, new_carpet
            )
            if direction is not None:
                self._opponent_carpet_direction = direction
                self._opponent_carpet_run_length = length
                self._opponent_pattern_age = 0
            else:
                self._opponent_pattern_age += 1
        else:
            self._opponent_pattern_age += 1

        if self._opponent_pattern_age > 4:
            self._opponent_carpet_direction = None
            self._opponent_carpet_run_length = 0

        self._prev_opponent_pos = opp_pos
        self._prev_carpet_mask = cur_carpet

    def _extract_carpet_info(self, start_pos, new_carpet_mask):
        best_direction = None
        best_length = 0

        for direction in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
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

    def _find_blocking_prime_target(self, board):
        if self._opponent_carpet_direction is None:
            return None
        if self._opponent_pattern_age > 2:
            return None

        opp_pos = board.opponent_worker.get_location()
        cur = opp_pos

        for _ in range(BS - 1):
            cur = enums.loc_after_direction(cur, self._opponent_carpet_direction)
            if not self._in_bounds(cur):
                break

            cell = board.get_cell(cur)

            if cell == enums.Cell.BLOCKED:
                break
            if cell == enums.Cell.SPACE:
                return cur
            # If primed or carpeted, continue scanning.

        return None

    # ----------------------------
    # Belief logic
    # ----------------------------

    def _compute_spawn_prior(self):
        if self.T is None:
            return [1.0 / NC] * NC

        belief = [0.0] * NC
        belief[0] = 1.0
        for _ in range(1000):
            belief = self._matvec(belief)
        self._normalize_in_place(belief)
        return belief

    def _predict_one_step(self):
        if self.T is None:
            return
        self.belief = self._matvec(self.belief)
        self._normalize_in_place(self.belief)

    def _observe(self, board, noise_obs, reported_dist):
        my_loc = self._my_loc(board)

        for idx in range(NC):
            x, y = idx % BS, idx // BS

            cell_type = board.get_cell((x, y))
            noise_likelihoods = NOISE_PROBS.get(
                cell_type, NOISE_PROBS[enums.Cell.SPACE]
            )
            noise_likelihood = noise_likelihoods[noise_obs.value]

            actual_dist = abs(x - my_loc[0]) + abs(y - my_loc[1])
            dist_likelihood = self._distance_likelihood(actual_dist, reported_dist)

            self.belief[idx] *= noise_likelihood * dist_likelihood

        if sum(self.belief) <= EPS:
            self.belief = self.base_prior[:]
        else:
            self._normalize_in_place(self.belief)

    def _distance_likelihood(self, actual_dist, reported_dist):
        p = 0.0
        for prob, off in zip(DIST_ERROR_PROBS, DIST_OFFSETS):
            shown = max(0, actual_dist + off)
            if shown == reported_dist:
                p += prob
        return p

    def _apply_search_evidence(self, board):
        my_info = self._extract_search_info(
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
        opp_info = self._extract_search_info(
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
        if signature == self._last_search_signature:
            return
        self._last_search_signature = signature

        for loc, success in (my_info, opp_info):
            if loc is not None and success:
                self.belief = self.base_prior[:]
                return

        changed = False
        for loc, success in (my_info, opp_info):
            if loc is not None and not success:
                idx = self._idx(loc)
                if 0 <= idx < NC and self.belief[idx] > 0.0:
                    self.belief[idx] = 0.0
                    changed = True

        if changed:
            if sum(self.belief) <= EPS:
                self.belief = self.base_prior[:]
            else:
                self._normalize_in_place(self.belief)

    def _extract_search_info(self, board, candidate_names):
        value = None
        for name in candidate_names:
            if hasattr(board, name):
                value = getattr(board, name)
                break

        if value is None:
            return (None, False)

        if callable(value):
            try:
                value = value()
            except Exception:
                return (None, False)

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and (value[0] is None or isinstance(value[0], tuple))
            and isinstance(value[1], bool)
        ):
            return value

        return (None, False)

    # ----------------------------
    # Board features
    # ----------------------------

    def _best_future_carpet_points(self, board, loc):
        best = 0
        for direction in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            run = self._primed_run_from_adjacent(board, loc, direction)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _primed_run_from_adjacent(self, board, loc, direction):
        nxt = enums.loc_after_direction(loc, direction)
        if not self._in_bounds(nxt):
            return 0
        if board.get_cell(nxt) != enums.Cell.PRIMED:
            return 0

        run = 0
        cur = nxt
        while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
            run += 1
            cur = enums.loc_after_direction(cur, direction)
        return run

    def _adjacent_space_count(self, board, loc):
        count = 0
        for direction in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            nxt = enums.loc_after_direction(loc, direction)
            if self._in_bounds(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                count += 1
        return count

    def _end_loc(self, start_loc, m):
        move_type = getattr(m, "move_type", None)
        direction = getattr(m, "direction", None)

        if direction is None:
            return start_loc

        steps = (
            getattr(m, "roll_length", 1) if move_type == enums.MoveType.CARPET else 1
        )

        loc = start_loc
        for _ in range(steps):
            loc = enums.loc_after_direction(loc, direction)
        return loc

    def _search_target(self, m):
        for attr in (
            "search_loc",
            "location",
            "target",
            "search_location",
            "guess_location",
            "loc",
        ):
            if hasattr(m, attr):
                value = getattr(m, attr)
                if isinstance(value, tuple) and len(value) == 2:
                    return value
        return None

    # ----------------------------
    # Belief features
    # ----------------------------

    def _belief_at(self, loc):
        if not self._in_bounds(loc):
            return 0.0
        return self.belief[self._idx(loc)]

    def _local_mass(self, center, radius):
        total = 0.0
        for y in range(BS):
            for x in range(BS):
                if abs(x - center[0]) + abs(y - center[1]) <= radius:
                    total += self.belief[self._idx((x, y))]
        return total

    def _expected_distance(self, loc):
        total = 0.0
        for idx, p in enumerate(self.belief):
            if p <= 0.0:
                continue
            x, y = idx % BS, idx // BS
            total += p * self._manhattan(loc, (x, y))
        return total

    def _top_mass(self, k):
        return sum(sorted(self.belief, reverse=True)[:k])

    # ----------------------------
    # Generic helpers
    # ----------------------------

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

    def _normalize_in_place(self, vec):
        s = sum(vec)
        if s <= EPS:
            return
        inv = 1.0 / s
        for i in range(len(vec)):
            vec[i] *= inv

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
