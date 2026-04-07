import sys
import os
import random

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
BS = enums.BOARD_SIZE
NC = BS * BS
EPS = 1e-12


class PlayerAgent:
    """
    Washington v2 - Aggressively conservative:
    - HMM belief tracking + search evidence
    - High-value carpet snap (>= 10 pts instant take)
    - STRICT search gating: only when EV clearly beats economy
    - Hard search cooldown after misses (3 turns)
    - 2-ply chain value for prime planning
    - Small carpet hold rule
    - Game-phase policy
    - Opponent blocking primes
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        try:
            self.T = transition_matrix.tolist()
        except Exception:
            self.T = None

        self.base_prior = self._compute_spawn_prior()
        self.belief = self.base_prior[:]
        self.turn = 0
        self._prev_opp = None
        self._last_search_sig = None

        self._opp_carpet_dir = None
        self._opp_carpet_run = 0
        self._opp_pattern_age = 999
        self._block_target = None

        self._failed_searches = 0
        self._search_cooldown = 0

    def commentate(self):
        return ""

    def play(self, board: board.Board, sensor_data, time_left: Callable = None):
        noise, dist = sensor_data

        self._apply_search_evidence(board)
        self._predict_one_step()
        self._observe(board, noise, dist)
        self.turn += 1

        self._detect_opp_pattern(board)
        self._block_target = self._find_block_target(board)

        if self._search_cooldown > 0:
            self._search_cooldown -= 1

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        searches = [
            m for m in moves if getattr(m, "move_type", None) == enums.MoveType.SEARCH
        ]
        non_search = [
            m for m in moves if getattr(m, "move_type", None) != enums.MoveType.SEARCH
        ]

        hv_carpets = [
            m
            for m in non_search
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            and enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0) >= 10
        ]
        if hv_carpets:
            return max(
                hv_carpets,
                key=lambda m: enums.CARPET_POINTS_TABLE.get(
                    getattr(m, "roll_length", 0), 0
                ),
            )

        best_non_search = None
        best_non_search_score = float("-inf")
        for m in non_search:
            s = self._score(board, m)
            if s > best_non_search_score:
                best_non_search_score = s
                best_non_search = m

        best_search = self._best_search(searches)
        if best_search is not None and self._should_search(
            board, best_search[1], best_non_search_score
        ):
            return best_search[0]

        if best_non_search is not None:
            return best_non_search

        return random.choice(moves)

    def _score(self, board, m):
        me = board.player_worker.get_location()
        end = self._end_loc(me, m)
        mt = m.move_type

        fc = self._future_carpet(board, end)
        adj = self._adj_space(board, end)
        mobility = self._mobility_score(board, end)

        top1 = max(self.belief) if self.belief else 0.0
        top3 = sum(sorted(self.belief, reverse=True)[:3])
        hot = top1 >= 0.25 or top3 >= 0.55

        l1 = self._local_mass(end, 1)
        l2 = self._local_mass(end, 2)
        ed = self._expected_dist(end)

        if hot:
            rb = 0.50 * l1 + 0.22 * l2 - 0.10 * ed
        else:
            rb = 0.08 * l1 - 0.01 * ed

        cp = 0.035 * self._manhattan(end, (BS // 2, BS // 2))
        tl = self._turns_remaining(board)

        block_bonus = self._block_bonus(board, end, mt)

        if mt == enums.MoveType.CARPET:
            pts = enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0)
            s = 2.90 * pts + 0.12 * fc + 0.05 * adj + 0.30 * rb - cp

            if pts <= 1 and fc >= pts + 3:
                s -= 4.0
            elif pts <= 0:
                s -= 2.0
            elif pts == 2:
                s -= 0.20

            return s + block_bonus + 0.15 * mobility

        if mt == enums.MoveType.PRIME:
            s = 1.70 + 0.75 * fc + 0.24 * adj + 0.24 * rb - cp

            two_ply = self._two_ply_carpet_value(board, me, m)
            s += 0.50 * two_ply

            if tl <= 10:
                s += 0.30
            return s + block_bonus + 0.15 * mobility

        if mt == enums.MoveType.PLAIN:
            s = -0.85 + 0.75 * fc + 0.18 * adj + 0.55 * rb - cp
            if tl <= 8:
                s += 0.25
            return s + block_bonus + 0.20 * mobility

        return -9999.0

    def _two_ply_carpet_value(self, board, me, prime_move):
        fc = board.forecast_move(prime_move)
        if fc is None:
            return 0.0

        end = self._end_loc(me, prime_move)
        best_after = 0
        for d in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            nxt = enums.loc_after_direction(end, d)
            if not self._in_bounds(nxt) or fc.get_cell(nxt) != enums.Cell.PRIMED:
                continue
            run = 0
            cur = nxt
            while self._in_bounds(cur) and fc.get_cell(cur) == enums.Cell.PRIMED:
                run += 1
                cur = enums.loc_after_direction(cur, d)
            best_after = max(best_after, enums.CARPET_POINTS_TABLE.get(run, 0))

        return best_after

    def _mobility_score(self, board, loc):
        count = 0
        for d in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            nxt = enums.loc_after_direction(loc, d)
            if self._in_bounds(nxt):
                cell = board.get_cell(nxt)
                if cell in (enums.Cell.SPACE, enums.Cell.PRIMED):
                    count += 1
        return count

    def _block_bonus(self, board, end, mt):
        if self._block_target is None or self._opp_carpet_dir is None:
            return 0.0
        if self._opp_pattern_age > 2:
            return 0.0
        if end != self._block_target:
            return 0.0

        opp = board.opponent_worker.get_location()
        d = self._manhattan(opp, end)

        bonus = 1.35
        if d <= 2:
            bonus += 0.55
        elif d <= 4:
            bonus += 0.25

        bonus += 0.10 * min(self._opp_carpet_run, 4)
        return bonus

    def _best_search(self, searches):
        best_m = None
        best_p = 0.0
        for m in searches:
            loc = self._search_target(m)
            if loc is None:
                continue
            p = self._belief_at(loc)
            if p > best_p:
                best_p = p
                best_m = m
        if best_m is None:
            return None
        return (best_m, best_p)

    def _should_search(self, board, p, best_non_search_score):
        if self._search_cooldown > 0:
            return False

        tl = self._turns_remaining(board)
        top3 = sum(sorted(self.belief, reverse=True)[:3])

        ev = 4.0 * p - 2.0 * (1.0 - p)
        if ev <= 0.5:
            return False

        penalty = min(self._failed_searches * 0.06, 0.24)

        if tl >= 26:
            base_thresh = 0.62
        elif tl >= 11:
            base_thresh = 0.55
        else:
            base_thresh = 0.48

        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        if diff > 5:
            base_thresh += 0.08
        elif diff < -5:
            base_thresh -= 0.06

        threshold = base_thresh - penalty

        if ev < best_non_search_score * 0.5:
            return False

        if p >= threshold:
            return True
        if tl <= 8 and p >= threshold - 0.06:
            return True
        if tl <= 4 and p >= threshold - 0.10:
            return True
        if top3 >= 0.80 and p >= threshold - 0.04:
            return True

        return False

    def _detect_opp_pattern(self, board):
        opp = board.opponent_worker.get_location()
        cur_carpet = getattr(board, "_carpet_mask", None)

        if self._prev_opp is None or cur_carpet is None:
            self._prev_opp = opp
            return

        prev_carpet = getattr(self, "_prev_carpet_mask", None)
        new_carpet = 0
        if prev_carpet is not None:
            new_carpet = cur_carpet & ~prev_carpet

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

    def _find_block_target(self, board):
        if self._opp_carpet_dir is None or self._opp_pattern_age > 2:
            return None

        opp = board.opponent_worker.get_location()
        cur = opp

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

    def _compute_spawn_prior(self):
        if self.T is None:
            return [1.0 / NC] * NC
        b = [0.0] * NC
        b[0] = 1.0
        for _ in range(1000):
            b = self._matvec(b)
        self._normalize(b)
        return b

    def _predict_one_step(self):
        if self.T is None:
            return
        self.belief = self._matvec(self.belief)
        self._normalize(self.belief)

    def _observe(self, board, noise_obs, reported_dist):
        me = board.player_worker.get_location()
        for idx in range(NC):
            x, y = idx % BS, idx // BS
            nl = NOISE.get(board.get_cell((x, y)), NOISE[enums.Cell.SPACE])[
                noise_obs.value
            ]
            ad = abs(x - me[0]) + abs(y - me[1])
            dl = sum(p for p, o in DIST if max(0, ad + o) == reported_dist)
            self.belief[idx] *= nl * dl

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

        sig = (my_info, opp_info)
        if sig == self._last_search_sig:
            return
        self._last_search_sig = sig

        for loc, success in (my_info, opp_info):
            if loc is not None and success:
                self.belief = self.base_prior[:]
                self._failed_searches = 0
                self._search_cooldown = 0
                return

        changed = False
        for loc, success in (my_info, opp_info):
            if loc is not None and not success:
                idx = loc[1] * BS + loc[0]
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
        for n in names:
            if hasattr(board, n):
                v = getattr(board, n)
                if isinstance(v, tuple) and len(v) == 2:
                    return v
        return (None, False)

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
        s = sum(vec)
        if s <= EPS:
            return
        inv = 1.0 / s
        for i in range(len(vec)):
            vec[i] *= inv

    def _belief_at(self, loc):
        if not self._in_bounds(loc):
            return 0.0
        return self.belief[loc[1] * BS + loc[0]]

    def _local_mass(self, center, radius):
        total = 0.0
        for y in range(BS):
            for x in range(BS):
                if abs(x - center[0]) + abs(y - center[1]) <= radius:
                    total += self.belief[y * BS + x]
        return total

    def _expected_dist(self, loc):
        total = 0.0
        for idx, p in enumerate(self.belief):
            if p <= 0.0:
                continue
            x, y = idx % BS, idx // BS
            total += p * (abs(x - loc[0]) + abs(y - loc[1]))
        return total

    def _future_carpet(self, board, loc):
        best = 0
        for d in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            nxt = enums.loc_after_direction(loc, d)
            if not self._in_bounds(nxt) or board.get_cell(nxt) != enums.Cell.PRIMED:
                continue
            run = 0
            cur = nxt
            while self._in_bounds(cur) and board.get_cell(cur) == enums.Cell.PRIMED:
                run += 1
                cur = enums.loc_after_direction(cur, d)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _adj_space(self, board, loc):
        count = 0
        for d in (
            enums.Direction.UP,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
            enums.Direction.RIGHT,
        ):
            nxt = enums.loc_after_direction(loc, d)
            if self._in_bounds(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                count += 1
        return count

    def _end_loc(self, start, m):
        d = getattr(m, "direction", None)
        if d is None:
            return start
        steps = (
            getattr(m, "roll_length", 1) if m.move_type == enums.MoveType.CARPET else 1
        )
        loc = start
        for _ in range(steps):
            loc = enums.loc_after_direction(loc, d)
        return loc

    def _search_target(self, m):
        for a in ("search_loc", "location", "target", "loc"):
            if hasattr(m, a):
                v = getattr(m, a)
                if isinstance(v, tuple) and len(v) == 2:
                    return v
        return None

    def _turns_remaining(self, board):
        worker = board.player_worker
        for attr in ("turns_remaining", "turns_left"):
            if hasattr(worker, attr):
                return getattr(worker, attr)
        return 40

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
