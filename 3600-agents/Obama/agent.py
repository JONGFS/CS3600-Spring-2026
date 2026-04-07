from game import enums

NOISE = {
    enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
    enums.Cell.SPACE:   (0.7, 0.15, 0.15),
    enums.Cell.PRIMED:  (0.1, 0.8, 0.1),
    enums.Cell.CARPET:  (0.1, 0.1, 0.8),
}
ERR = ((0.12, -1), (0.7, 0), (0.12, 1), (0.06, 2))
DIRS = (enums.Direction.UP, enums.Direction.DOWN, enums.Direction.LEFT, enums.Direction.RIGHT)
BS = enums.BOARD_SIZE
NC = BS * BS
EPS = 1e-12

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left=None):
        try: self.T = transition_matrix.tolist()
        except Exception: self.T = None
        self.base = [1 / NC] * NC
        if self.T:
            self.base = [0.0] * NC
            self.base[0] = 1.0
            for _ in range(1000): self.base = self._mv(self.base)
            self._norm(self.base)
        self.b = self.base[:]
        self.last = None
        self.turn = 0

    def commentate(self):
        return "Blitzkrieg"

    def play(self, board, sensor_data, time_left):
        self.turn += 1
        self._apply_search_info(board)
        if self.T:
            self.b = self._mv(self.b)
            self._norm(self.b)

        noise, dist = sensor_data
        me = self._loc(board.player_worker)
        for i in range(NC):
            x, y = i % BS, i // BS
            nl = NOISE.get(board.get_cell((x, y)), NOISE[enums.Cell.SPACE])[noise.value]
            ad = abs(x - me[0]) + abs(y - me[1])
            dl = sum(p for p, o in ERR if max(0, ad + o) == dist)
            self.b[i] *= nl * dl
        self._norm(self.b) if sum(self.b) > EPS else self._reset()

        moves = board.get_valid_moves(exclude_search=False)
        if not moves: return None

        searches = [m for m in moves if getattr(m, "move_type", None) == enums.MoveType.SEARCH]
        if searches:
            s = max(searches, key=lambda m: self._at(self._target(m)))
            p = self._at(self._target(s))
            if p >= 0.58 or (p >= 0.50 and self._top3() >= 0.72) or (p >= 0.44 and self._turns(board) <= 6):
                return s

        non = [m for m in moves if getattr(m, "move_type", None) != enums.MoveType.SEARCH]
        if not non: return moves[0]

        def score(m):
            tp = m.move_type
            end = self._end(me, m)
            fut = self._future(board, end)
            ext = self._extend(board, end)
            adj = self._adj(board, end)
            rat = self._mass(end, 1)
            opp = self._opp_pressure(board, end)

            if tp == enums.MoveType.CARPET:
                pts = enums.CARPET_POINTS_TABLE.get(getattr(m, "roll_length", 0), 0)
                return 3.2 * pts + 0.15 * fut + 0.15 * opp + (1.0 if pts >= 4 else 0.0) - (1.3 if pts <= 0 else 0.0)

            if tp == enums.MoveType.PRIME:
                opening = self.turn <= 12
                return (
                    1.7
                    + 1.25 * fut
                    + 0.9 * ext
                    + 0.18 * adj
                    + 0.20 * opp
                    + (0.15 * rat if not opening else 0.0)
                )

            return (
                -0.25
                + 1.10 * fut
                + 0.55 * ext
                + 0.10 * adj
                + 0.30 * opp
                + 0.20 * rat
                - 0.03 * self._ed(end)
            )

        return max(non, key=score)

    def _apply_search_info(self, board):
        def get(names):
            for n in names:
                if hasattr(board, n):
                    v = getattr(board, n)
                    v = v() if callable(v) else v
                    if isinstance(v, tuple) and len(v) == 2: return v
            return (None, False)

        info = (
            get(["your_search_location_and_result","player_search_location_and_result","my_search_location_and_result","your_search_info","player_search_info"]),
            get(["opponent_search_location_and_result","enemy_search_location_and_result","opp_search_location_and_result","opponent_search_info"]),
        )
        if info == self.last: return
        self.last = info
        for loc, ok in info:
            if loc is not None and ok:
                self._reset()
                return
        changed = False
        for loc, ok in info:
            if loc is not None and not ok:
                self.b[self._i(loc)] = 0.0
                changed = True
        if changed:
            self._norm(self.b) if sum(self.b) > EPS else self._reset()

    def _future(self, board, loc):
        best = 0
        for d in DIRS:
            p = enums.loc_after_direction(loc, d)
            run = 0
            while self._in(p) and board.get_cell(p) == enums.Cell.PRIMED:
                run += 1
                p = enums.loc_after_direction(p, d)
            best = max(best, enums.CARPET_POINTS_TABLE.get(run, 0))
        return best

    def _extend(self, board, loc):
        return max(self._run(board, loc, d) for d in DIRS)

    def _run(self, board, loc, d):
        a = b = 0
        p = enums.loc_after_direction(loc, d)
        while self._in(p) and board.get_cell(p) == enums.Cell.PRIMED:
            a += 1; p = enums.loc_after_direction(p, d)
        od = {DIRS[0]: DIRS[1], DIRS[1]: DIRS[0], DIRS[2]: DIRS[3], DIRS[3]: DIRS[2]}[d]
        p = enums.loc_after_direction(loc, od)
        while self._in(p) and board.get_cell(p) == enums.Cell.PRIMED:
            b += 1; p = enums.loc_after_direction(p, od)
        return a + b

    def _opp_pressure(self, board, loc):
        opp = self._loc(board.opponent_worker)
        return 1.0 / (1 + abs(loc[0] - opp[0]) + abs(loc[1] - opp[1]))

    def _adj(self, board, loc):
        return sum(self._in(enums.loc_after_direction(loc, d)) and board.get_cell(enums.loc_after_direction(loc, d)) == enums.Cell.SPACE for d in DIRS)

    def _mass(self, c, r):
        return sum(self.b[self._i((x, y))] for y in range(BS) for x in range(BS) if abs(x - c[0]) + abs(y - c[1]) <= r)

    def _ed(self, loc):
        return sum(p * (abs(i % BS - loc[0]) + abs(i // BS - loc[1])) for i, p in enumerate(self.b) if p > 0)

    def _top3(self):
        return sum(sorted(self.b, reverse=True)[:3])

    def _mv(self, v):
        out = [0.0] * NC
        for i, bi in enumerate(v):
            if bi <= EPS: continue
            for j, pij in enumerate(self.T[i]):
                if pij > 0: out[j] += bi * pij
        return out

    def _norm(self, v):
        s = sum(v)
        if s <= EPS: return
        for i in range(NC): v[i] /= s

    def _loc(self, w):
        return w.get_location() if hasattr(w, "get_location") else w.location

    def _turns(self, board):
        w = board.player_worker
        return getattr(w, "turns_remaining", getattr(w, "turns_left", 40))

    def _end(self, start, m):
        d = getattr(m, "direction", None)
        if d is None: return start
        s = getattr(m, "roll_length", 1) if getattr(m, "move_type", None) == enums.MoveType.CARPET else 1
        p = start
        for _ in range(s): p = enums.loc_after_direction(p, d)
        return p

    def _target(self, m):
        for a in ("location", "target", "search_location", "guess_location", "loc"):
            if hasattr(m, a):
                v = getattr(m, a)
                if isinstance(v, tuple) and len(v) == 2: return v
        return (0, 0)

    def _i(self, loc):
        return loc[1] * BS + loc[0]

    def _at(self, loc):
        return self.b[self._i(loc)] if self._in(loc) else 0.0

    def _in(self, loc):
        return 0 <= loc[0] < BS and 0 <= loc[1] < BS

    def _reset(self):
        self.b = self.base[:]