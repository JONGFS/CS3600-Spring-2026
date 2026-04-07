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
DIRS = (
    enums.Direction.UP,
    enums.Direction.DOWN,
    enums.Direction.LEFT,
    enums.Direction.RIGHT,
)


class PlayerAgent:
    """
    Biden v3 - Carpet Blitz:
    - EXTREMELY conservative search (only when near-certain)
    - Aggressive carpet expansion: primes everywhere, carpets whenever possible
    - High-value carpet snap (>= 10 pts instant take)
    - Hard search ban after any miss (5-turn cooldown)
    - Game-phase policy
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        try:
            self.T = transition_matrix.tolist()
        except Exception:
            self.T = None

        self.base_prior = [1.0 / NC] * NC
        if self.T:
            b = [0.0] * NC
            b[0] = 1.0
            for _ in range(1000):
                b = self._mv(b)
            self._norm(b)
            self.base_prior = b

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
        return "Carpet Blitz"

    def play(self, board: board.Board, sensor_data, time_left: Callable = None):
        noise, dist = sensor_data

        self._apply_search_evidence(board)
        if self.T:
            self.belief = self._mv(self.belief)
            self._norm(self.belief)

        me = self._loc(board.player_worker)
        for i in range(NC):
            x, y = i % BS, i // BS
            nl = NOISE.get(board.get_cell((x, y)), NOISE[enums.Cell.SPACE])[noise.value]
            ad = abs(x - me[0]) + abs(y - me[1])
            dl = sum(p for p, o in DIST if max(0, ad + o) == dist)
            self.belief[i] *= nl * dl

        if sum(self.belief) <= EPS:
            self.belief = self.base_prior[:]
        else:
            self._norm(self.belief)

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
        me = self._loc(board.player_worker)
        end = self._end(me, m)
        mt = m.move_type

        fc = self._future_carpet(board, end)
        adj = self._adj_space(board, end)

        top1 = max(self.belief) if self.belief else 0.0
        top3 = sum(sorted(self.belief, reverse=True)[:3])
        hot = top1 >= 0.25 or top3 >= 0.55

        l1 = self._mass(end, 1)
        l2 = self._mass(end, 2)
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
            s = 3.00 * pts + 0.15 * fc + 0.06 * adj + 0.30 * rb - cp

            if pts <= 1 and fc >= pts + 3:
                s -= 5.0
            elif pts <= 0:
                s -= 2.5
            elif pts == 2:
                s -= 0.15

            return s + block_bonus

        if mt == enums.MoveType.PRIME:
            s = 1.80 + 0.80 * fc + 0.26 * adj + 0.24 * rb - cp

            two_ply = self._two_ply_carpet_value(board, me, m)
            s += 0.60 * two_ply

            if tl <= 10:
                s += 0.35
            return s + block_bonus

        if mt == enums.MoveType.PLAIN:
            s = -0.90 + 0.80 * fc + 0.20 * adj + 0.55 * rb - cp
            if tl <= 8:
                s += 0.30
            return s + block_bonus

        return -9999.0

    def _two_ply_carpet_value(self, board, me, prime_move):
        fc = board.forecast_move(prime_move)
        if fc is None:
            return 0.0

        end = self._end(me, prime_move)
        best_after = 0
        for d in DIRS:
            nxt = enums.loc_after_direction(end, d)
            if not self._in(nxt) or fc.get_cell(nxt) != enums.Cell.PRIMED:
                continue
            run = 0
            cur = nxt
            while self._in(cur) and fc.get_cell(cur) == enums.Cell.PRIMED:
                run += 1
                cur = enums.loc_after_direction(cur, d)
            best_after = max(best_after, enums.CARPET_POINTS_TABLE.get(run, 0))

        return best_after

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
            loc = self._target(m)
            if loc is None:
                continue
            p = self._at(loc)
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
        if ev <= 1.0:
            return False

        penalty = min(self._failed_searches * 0.08, 0.32)

        if tl >= 26:
            base_thresh = 0.68
        elif tl >= 11:
            base_thresh = 0.60
        else:
            base_thresh = 0.52

        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        if diff > 3:
            base_thresh += 0.10
        elif diff < -8:
            base_thresh -= 0.08

        threshold = base_thresh - penalty

        if ev < best_non_search_score * 0.4:
            return False

        if p >= threshold:
            return True
        if tl <= 6 and p >= threshold - 0.08:
            return True
        if tl <= 3 and p >= threshold - 0.12:
            return True
        if top3 >= 0.85 and p >= threshold - 0.05:
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

        opp = board.opponent_worker.get_location()
        cur = opp

        for _ in range(BS - 1):
            cur = enums.loc_after_direction(cur, self._opp_carpet_dir)
            if not self._in(cur):
                break

            cell = board.get_cell(cur)
            if cell == enums.Cell.BLOCKED:
                break
            if cell == enums.Cell.SPACE:
                return cur

        return None

    def _apply_search_evidence(self, board):
        def get(names):
            for n in names:
                if hasattr(board, n):
                    v = getattr(board, n)
                    v = v() if callable(v) else v
                    if isinstance(v, tuple) and len(v) == 2:
                        return v
            return (None, False)

        my_info = get(
            [
                "player_search",
                "your_search_location_and_result",
                "player_search_location_and_result",
                "my_search_location_and_result",
                "your_search_info",
                "player_search_info",
            ]
        )
        opp_info = get(
            [
                "opponent_search",
                "opponent_search_location_and_result",
                "enemy_search_location_and_result",
                "opp_search_location_and_result",
                "opponent_search_info",
            ]
        )

        sig = (my_info, opp_info)
        if sig == self._last_search_sig:
            return
        self._last_search_sig = sig

        for loc, ok in (my_info, opp_info):
            if loc is not None and ok:
                self.belief = self.base_prior[:]
                self._failed_searches = 0
                self._search_cooldown = 0
                return

        changed = False
        for loc, ok in (my_info, opp_info):
            if loc is not None and not ok:
                idx = self._i(loc)
                if 0 <= idx < NC and self.belief[idx] > 0.0:
                    self.belief[idx] = 0.0
                    changed = True
                    self._failed_searches += 1
                    self._search_cooldown = 5

        if changed:
            if sum(self.belief) <= EPS:
                self.belief = self.base_prior[:]
            else:
                self._norm(self.belief)

    def _future_carpet(self, board, loc):
        best = 0
        for d in DIRS:
            p = enums.loc_after_direction(loc, d)
            if not self._in(p) or board.get_cell(p) != enums.Cell.PRIMED:
                continue
            run = 0
            while self._in(p) and board.get_cell(p) == enums.Cell.PRIMED:
                run += 1
                p = enums.loc_after_direction(p, d)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _adj_space(self, board, loc):
        count = 0
        for d in DIRS:
            nxt = enums.loc_after_direction(loc, d)
            if self._in(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                count += 1
        return count

    def _mv(self, v):
        if self.T is None:
            return [1.0 / NC] * NC
        out = [0.0] * NC
        for i, bi in enumerate(v):
            if bi <= EPS:
                continue
            for j, pij in enumerate(self.T[i]):
                if pij > 0:
                    out[j] += bi * pij
        return out

    def _norm(self, v):
        s = sum(v)
        if s <= EPS:
            return
        for i in range(NC):
            v[i] /= s

    def _loc(self, w):
        return w.get_location() if hasattr(w, "get_location") else w.location

    def _end(self, start, m):
        d = getattr(m, "direction", None)
        if d is None:
            return start
        steps = (
            getattr(m, "roll_length", 1)
            if getattr(m, "move_type", None) == enums.MoveType.CARPET
            else 1
        )
        p = start
        for _ in range(steps):
            p = enums.loc_after_direction(p, d)
        return p

    def _target(self, m):
        for a in ("search_loc", "location", "target", "search_location", "loc"):
            if hasattr(m, a):
                v = getattr(m, a)
                if isinstance(v, tuple) and len(v) == 2:
                    return v
        return (0, 0)

    def _i(self, loc):
        return loc[1] * BS + loc[0]

    def _at(self, loc):
        return self.belief[self._i(loc)] if self._in(loc) else 0.0

    def _mass(self, c, r):
        return sum(
            self.belief[self._i((x, y))]
            for y in range(BS)
            for x in range(BS)
            if abs(x - c[0]) + abs(y - c[1]) <= r
        )

    def _expected_dist(self, loc):
        return sum(
            p * (abs(i % BS - loc[0]) + abs(i // BS - loc[1]))
            for i, p in enumerate(self.belief)
            if p > 0
        )

    def _turns_remaining(self, board):
        worker = board.player_worker
        for attr in ("turns_remaining", "turns_left"):
            if hasattr(worker, attr):
                return getattr(worker, attr)
        return 40

    def _in(self, loc):
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
