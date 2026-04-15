from __future__ import annotations

from collections.abc import Callable
from math import inf
from typing import Dict, List, Sequence, Tuple

from game import board as game_board
from game import enums, move


BOARD_SIZE = enums.BOARD_SIZE
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_DISTANCE_OBS = (BOARD_SIZE - 1) * 2 + 2
DIRS = (
    enums.Direction.UP,
    enums.Direction.RIGHT,
    enums.Direction.DOWN,
    enums.Direction.LEFT,
)
NOISE_PROBS = {
    enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
    enums.Cell.SPACE: (0.7, 0.15, 0.15),
    enums.Cell.PRIMED: (0.1, 0.8, 0.1),
    enums.Cell.CARPET: (0.1, 0.1, 0.8),
}
DISTANCE_ERROR_OFFSETS = (-1, 0, 1, 2)
DISTANCE_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)
CARPET_VALUES = enums.CARPET_POINTS_TABLE

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class PlayerAgent:
    def __init__(
        self,
        board: game_board.Board,
        transition_matrix=None,
        time_left: Callable | None = None,
    ):
        self.transitions = self._build_sparse_transitions(transition_matrix)
        self.search_prior = self._compute_headstart_prior()
        self.belief = list(self.search_prior)
        self.distance_likelihood = self._precompute_distance_likelihoods()

        self.last_player_search = (None, False)
        self.last_opponent_search = (None, False)
        self.last_turn_seen = -1

        self.ttable: Dict[Tuple, Tuple[int, float, int]] = {}
        self.tt_move: Dict[Tuple, Tuple] = {}

        self.turn_depths: List[int] = []
        self.turn_times: List[float] = []
        self.max_depth_reached = 0
        self.search_candidates_seen = 0
        self.search_moves_played = 0
        self.search_ev_sum = 0.0
        self._last_search_turn = -99
        self._consecutive_misses = 0
        self._failed_search_turns: Dict[int, int] = {}
        self._denial_cache: Dict[Tuple, Tuple[float, float, float, float]] = {}

    def commentate(self):
        if not self.turn_depths:
            return ""
        avg_depth = sum(self.turn_depths) / len(self.turn_depths)
        avg_time = sum(self.turn_times) / max(1, len(self.turn_times))
        return (
            f"kevin nguyen d={avg_depth:.2f}/{self.max_depth_reached} "
            f"t={avg_time:.2f}s searches={self.search_moves_played} "
            f"sev={self.search_ev_sum:.2f}"
        )

    def play(
        self,
        board: game_board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self._synchronize_belief(board, sensor_data)
        self._denial_cache.clear()

        legal_moves = board.get_valid_moves(exclude_search=True)
        if not legal_moves:
            return move.Move.search(self._best_belief_cell(self.belief))

        legal_moves = self._drop_len1_carpets(board, legal_moves)

        forced_move = self._forced_tactical_move(board, legal_moves, self.belief)
        if forced_move is not None:
            return forced_move

        search_candidates = self._candidate_search_moves(board, self.belief)
        self.search_candidates_seen += len(search_candidates)
        root_moves = self._build_root_candidates(
            board, self.belief, legal_moves, search_candidates
        )
        if not root_moves:
            return legal_moves[0]

        start_remaining = time_left()
        budget = self._turn_budget(board, start_remaining)
        deadline = max(0.02, start_remaining - budget)

        best_move = root_moves[0]
        best_score = 0.0
        completed_depth = 0
        max_depth = self._max_search_depth(board, root_moves, self.belief)
        self.ttable.clear()
        self.tt_move.clear()

        for depth in range(1, max_depth + 1):
            if time_left() <= deadline:
                break
            try:
                window = 130.0 if depth >= 2 else 100000.0
                alpha = -inf
                beta = inf
                if depth >= 2:
                    alpha = best_score - window
                    beta = best_score + window
                value, chosen = self._search_root(
                    board,
                    self.belief,
                    root_moves,
                    depth,
                    alpha,
                    beta,
                    deadline,
                    time_left,
                )
                if depth >= 2 and (value <= alpha or value >= beta):
                    value, chosen = self._search_root(
                        board,
                        self.belief,
                        root_moves,
                        depth,
                        -inf,
                        inf,
                        deadline,
                        time_left,
                    )
            except TimeoutError:
                break
            if chosen is not None:
                best_score = value
                best_move = chosen
                completed_depth = depth
                root_moves = self._promote_best_move(root_moves, chosen)

        if completed_depth == 0:
            selected = root_moves[0]
        else:
            selected = best_move

        self.turn_depths.append(completed_depth)
        self.max_depth_reached = max(self.max_depth_reached, completed_depth)
        self.turn_times.append(max(0.0, start_remaining - time_left()))

        if selected.move_type == enums.MoveType.SEARCH:
            self.search_moves_played += 1
            self._last_search_turn = board.turn_count
            p = self.belief[self._pos_to_idx(selected.search_loc)]
            self.search_ev_sum += self._search_ev(p)

        return selected

    # ------------------------------------------------------------------
    # Infrastructure (belief tracking, transitions, likelihoods)
    # ------------------------------------------------------------------

    def _build_sparse_transitions(
        self, transition_matrix
    ) -> List[List[Tuple[int, float]]]:
        transitions: List[List[Tuple[int, float]]] = []
        for i in range(CELL_COUNT):
            row: List[Tuple[int, float]] = []
            for j in range(CELL_COUNT):
                prob = float(transition_matrix[i][j])
                if prob > 0.0:
                    row.append((j, prob))
            if not row:
                row.append((i, 1.0))
            transitions.append(row)
        return transitions

    def _compute_headstart_prior(self) -> List[float]:
        belief = [0.0] * CELL_COUNT
        belief[0] = 1.0
        for _ in range(1000):
            belief = self._advance_belief_once(belief)
        return belief

    def _precompute_distance_likelihoods(self) -> List[List[List[float]]]:
        likelihoods: List[List[List[float]]] = []
        for worker_idx in range(CELL_COUNT):
            wx, wy = self._idx_to_pos(worker_idx)
            per_observation = []
            for observed in range(MAX_DISTANCE_OBS + 1):
                row = [0.0] * CELL_COUNT
                for rat_idx in range(CELL_COUNT):
                    rx, ry = self._idx_to_pos(rat_idx)
                    actual = abs(wx - rx) + abs(wy - ry)
                    p = 0.0
                    for offset, prob in zip(
                        DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS
                    ):
                        estimate = max(0, actual + offset)
                        if estimate == observed:
                            p += prob
                    row[rat_idx] = p
                per_observation.append(row)
            likelihoods.append(per_observation)
        return likelihoods

    def _synchronize_belief(self, board: game_board.Board, sensor_data: Tuple):
        if board.turn_count == 0 and self.last_turn_seen == -1:
            self.belief = list(self.search_prior)

        self._apply_search_feedback(
            board.player_search, board.turn_count, is_player=True
        )
        self._apply_search_feedback(board.opponent_search, board.turn_count)

        predicted = self._advance_belief_once(self.belief)
        noise_obs, distance_obs = sensor_data
        worker_idx = self._pos_to_idx(board.player_worker.get_location())
        distance_obs = int(max(0, min(MAX_DISTANCE_OBS, distance_obs)))
        distance_row = self.distance_likelihood[worker_idx][distance_obs]

        updated = [0.0] * CELL_COUNT
        total = 0.0
        noise_idx = int(noise_obs)
        for idx, prior in enumerate(predicted):
            if prior <= 0.0:
                continue
            loc = self._idx_to_pos(idx)
            cell_prob = NOISE_PROBS[board.get_cell(loc)][noise_idx]
            v = prior * cell_prob * distance_row[idx]
            updated[idx] = v
            total += v

        if total <= 0.0:
            self.belief = list(predicted)
            self._normalize(self.belief)
        else:
            inv = 1.0 / total
            self.belief = [v * inv for v in updated]

        self.last_player_search = board.player_search
        self.last_opponent_search = board.opponent_search
        self.last_turn_seen = board.turn_count

    def _apply_search_feedback(
        self,
        search_info: Tuple,
        current_turn: int,
        is_player: bool = False,
    ):
        loc, result = search_info
        if loc is None:
            return
        if (
            search_info == self.last_player_search
            or search_info == self.last_opponent_search
        ):
            return

        idx = self._pos_to_idx(loc)
        if result:
            self.belief = list(self.search_prior)
            self._consecutive_misses = 0
            return

        if is_player:
            self._consecutive_misses += 1
            self._failed_search_turns[idx] = current_turn

        self.belief[idx] = 0.0
        if sum(self.belief) <= 0.0:
            self.belief = list(self.search_prior)
            return
        self._normalize(self.belief)

    def _advance_belief_once(self, belief: Sequence[float]) -> List[float]:
        nxt = [0.0] * CELL_COUNT
        for src_idx, src_prob in enumerate(belief):
            if src_prob <= 0.0:
                continue
            for dst_idx, prob in self.transitions[src_idx]:
                nxt[dst_idx] += src_prob * prob
        return nxt

    # ------------------------------------------------------------------
    # Critical block & high-roll detection
    # ------------------------------------------------------------------

    def _forced_tactical_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
        belief: Sequence[float],
    ) -> move.Move | None:
        threat = self._best_enemy_immediate_carpet_points(board)

        best_cash = self._best_cash_now_move(board, legal_moves)
        if best_cash is not None:
            pts = float(CARPET_VALUES[best_cash.roll_length])
            if pts >= 10.0 or self._is_contested_cash_move(board, best_cash):
                return best_cash
            if threat >= 6.0 and pts >= 4.0:
                denied_now, denied_total, after_now, _ = self._move_denial_delta(
                    board, best_cash
                )
                if denied_now >= 2.0 or after_now <= max(0.0, threat - 2.0):
                    return best_cash

        best_carpet = self._best_counter_carpet_move(board, legal_moves)
        if best_carpet is not None:
            best_carpet_pts = float(CARPET_VALUES[best_carpet.roll_length])
            if best_carpet_pts >= 10.0:
                return best_carpet
            if threat >= 10.0 and best_carpet_pts >= 6.0:
                return best_carpet
            if threat >= 6.0 and best_carpet_pts >= max(4.0, threat - 2.0):
                return best_carpet

        denial_move = self._best_denial_move(board, legal_moves)
        if denial_move is not None:
            denied_now, denied_total, after_now, _ = self._move_denial_delta(
                board, denial_move
            )
            if threat >= 10.0 and (denied_now >= 6.0 or after_now <= 4.0):
                return denial_move
            if threat >= 6.0 and denied_total >= 4.5:
                return denial_move

        forced_block = self._critical_block_move(board, legal_moves)
        if forced_block is not None:
            return forced_block

        if best_cash is not None and float(CARPET_VALUES[best_cash.roll_length]) >= 6.0:
            return best_cash
        if best_carpet is not None and float(CARPET_VALUES[best_carpet.roll_length]) >= 6.0:
            return best_carpet

        top_idx = max(range(CELL_COUNT), key=lambda i: belief[i])
        top_p = belief[top_idx]
        top_ev = self._search_ev(top_p) - self._search_repeat_penalty(board, top_idx)
        turns_left = board.player_worker.turns_left
        my_now = self._best_immediate_carpet_points(board, enemy=False)
        if threat < 4.0 and my_now < 4.0 and top_ev >= 3.9 and top_p >= 0.93 and turns_left > 2:
            return move.Move.search(self._idx_to_pos(top_idx))

        return None

    def _best_counter_carpet_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
    ) -> move.Move | None:
        best = None
        best_score = -inf
        opp_now = self._best_enemy_immediate_carpet_points(board)
        my_now = self._best_immediate_carpet_points(board, enemy=False)
        for mv in legal_moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            pts = float(CARPET_VALUES[mv.roll_length])
            denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
            contested = self._is_contested_cash_move(board, mv)
            score = 156.0 * pts + 20.0 * mv.roll_length
            score += 170.0 * denied_now + 110.0 * denied_total
            if contested:
                score += 190.0
            if pts <= 0.0:
                score += 170.0 * denied_total - 110.0
            elif pts == 2.0:
                score -= 12.0
            elif pts == 4.0:
                score += 30.0
            if pts >= opp_now and pts >= 4.0:
                score += 180.0
            if pts >= my_now and pts >= 4.0:
                score += 70.0
            if pts >= 10.0:
                score += 230.0
            if opp_now >= 6.0 and after_now <= max(0.0, opp_now - 4.0):
                score += 160.0
            if score > best_score:
                best_score = score
                best = mv
        return best

    def _critical_block_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
    ) -> move.Move | None:
        threat = self._best_enemy_immediate_carpet_points(board)
        if threat < 6.0:
            return None

        lane_cells = set(self._enemy_lane_cells(board))
        if not lane_cells:
            return None

        best = None
        best_score = -inf
        my_loc = board.player_worker.get_location()
        for mv in legal_moves:
            if mv.move_type not in (
                enums.MoveType.PLAIN,
                enums.MoveType.PRIME,
                enums.MoveType.CARPET,
            ):
                continue

            denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
            score = 160.0 * denied_now + 120.0 * denied_total

            if mv.move_type == enums.MoveType.CARPET:
                pts = float(CARPET_VALUES[mv.roll_length])
                score += 80.0 * pts + 10.0 * mv.roll_length
                if pts <= 0.0:
                    score += 120.0 * denied_total
            else:
                dest = enums.loc_after_direction(my_loc, mv.direction)
                in_lane = dest in lane_cells
                if not in_lane:
                    best_dist = min(
                        abs(dest[0] - x) + abs(dest[1] - y) for x, y in lane_cells
                    )
                    if best_dist > 1 and denied_total < 3.0:
                        continue
                    score += 700.0 - 95.0 * best_dist
                else:
                    score += 980.0
                if mv.move_type == enums.MoveType.PRIME:
                    score += 18.0
                score += 5.0 * self._open_neighbors(board, dest)
                score += 24.0 * self._future_carpet_points_from(board, dest)

            if threat >= 10.0 and after_now <= 4.0:
                score += 140.0
            if score > best_score:
                best_score = score
                best = mv
        return best

    def _best_high_roll_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
    ) -> move.Move | None:
        best = None
        best_score = -inf
        for mv in legal_moves:
            if mv.move_type != enums.MoveType.CARPET or mv.roll_length < 4:
                continue
            pts = float(CARPET_VALUES[mv.roll_length])
            score = 100.0 * pts + 12.0 * mv.roll_length
            if score > best_score:
                best_score = score
                best = mv
        return best

    # ------------------------------------------------------------------
    # Search candidate construction
    # ------------------------------------------------------------------

    def _candidate_search_moves(
        self,
        board: game_board.Board,
        belief: Sequence[float],
    ) -> List[move.Move]:
        top = sorted(range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True)[:6]
        searches: List[move.Move] = []
        best_board = self._best_board_move_heuristic(board, belief)

        miss_gate = self._consecutive_misses >= 2
        point_margin_now = (
            board.player_worker.get_points() - board.opponent_worker.get_points()
        )
        my_now = self._best_immediate_carpet_points(board, enemy=False)
        opp_now = self._best_enemy_immediate_carpet_points(board)

        for idx in top:
            p = belief[idx]
            repeat_pen = self._search_repeat_penalty(board, idx)
            ev = self._search_ev(p) - repeat_pen
            turns_left = board.player_worker.turns_left

            if ev < 0.35:
                continue
            if ev < 0.8 and best_board > 560.0:
                continue
            if miss_gate and ev < 4.2:
                continue
            if point_margin_now <= -4 and ev < 1.8:
                continue
            if repeat_pen >= 2.4 and turns_left <= 8:
                continue
            if my_now >= 4.0 and ev < 3.8:
                continue
            if opp_now >= 4.0 and ev < 3.0:
                continue
            if turns_left <= 6 and ev < 1.1 and best_board > 560.0:
                continue
            if turns_left <= 4 and ev < 2.1 and best_board > 470.0:
                continue
            if turns_left <= 2 and ev < 3.6:
                continue

            if opp_now >= 10.0 and ev < 3.8:
                continue
            if opp_now >= 6.0 and ev < 3.2:
                continue

            searches.append(move.Move.search(self._idx_to_pos(idx)))
            if p < 0.45 or len(searches) >= 1:
                break

        entropy_bucket = self._belief_entropy_bucket(belief)
        turns_left = board.player_worker.turns_left
        max_keep = len(searches)
        if entropy_bucket >= 1:
            max_keep = min(max_keep, 1)
        if turns_left <= 10:
            max_keep = min(max_keep, 1)
        if max_keep < len(searches):
            searches = searches[:max_keep]

        return searches

    def _build_root_candidates(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        legal_moves: Sequence[move.Move],
        search_moves: Sequence[move.Move],
    ) -> List[move.Move]:
        scored = []
        for mv in legal_moves:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        for mv in search_moves:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        scored.sort(key=lambda item: item[0], reverse=True)

        out: List[move.Move] = []
        seen = set()
        categories = (
            enums.MoveType.CARPET,
            enums.MoveType.PRIME,
            enums.MoveType.PLAIN,
        )
        if self._should_seed_search_root(board, belief, legal_moves, search_moves):
            categories = categories + (enums.MoveType.SEARCH,)
        for cat in categories:
            for _, mv in scored:
                if mv.move_type != cat:
                    continue
                key = self._move_key(mv)
                if key in seen:
                    continue
                out.append(mv)
                seen.add(key)
                break

        for _, mv in scored:
            key = self._move_key(mv)
            if key in seen:
                continue
            out.append(mv)
            seen.add(key)
            if len(out) >= self._root_width(board):
                break
        return out

    def _should_seed_search_root(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        legal_moves: Sequence[move.Move],
        search_moves: Sequence[move.Move],
    ) -> bool:
        if not search_moves:
            return False

        turns_left = board.player_worker.turns_left
        opp_now = self._best_enemy_immediate_carpet_points(board)

        best_search = max(
            self._move_heuristic(board, belief, mv) for mv in search_moves
        )
        best_board = max(self._move_heuristic(board, belief, mv) for mv in legal_moves)

        if opp_now >= 6.0 and best_search < best_board + 320.0:
            return False
        if turns_left <= 2:
            return best_search >= best_board + 220.0
        if turns_left <= 4:
            return best_search >= best_board + 185.0
        if turns_left <= 6:
            return best_search >= best_board + 145.0
        return best_search >= best_board + 170.0

    # ------------------------------------------------------------------
    # Expectimax search with exact TT
    # ------------------------------------------------------------------

    def _search_root(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        moves: Sequence[move.Move],
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
        time_left: Callable,
    ) -> Tuple[float, move.Move | None]:
        del alpha, beta
        best_move = None
        best_val = -inf

        for mv in moves:
            if time_left() <= deadline:
                raise TimeoutError
            child_board, child_belief = self._simulate_action(board, belief, mv)
            if child_board is None:
                continue
            child_board.reverse_perspective()
            value = self._expectiminimax_opp(
                child_board,
                child_belief,
                depth - 1,
                deadline,
                time_left,
            )
            if value > best_val:
                best_val = value
                best_move = mv
        return best_val, best_move

    def _expectimax_max(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        depth: int,
        deadline: float,
        time_left: Callable,
    ) -> float:
        if time_left() <= deadline:
            raise TimeoutError

        if depth <= 0 or board.is_game_over():
            return self._evaluate(board, belief)

        key = ("MAX", self._tt_key(board, belief))
        hit = self._probe_tt(key, depth)
        if hit is not None:
            return hit

        legal = board.get_valid_moves(exclude_search=True)
        legal = self._drop_len1_carpets(board, legal)
        searches = self._candidate_searches_for_belief(belief)
        moves = self._node_moves(board, belief, legal, searches, key)
        if not moves:
            return self._evaluate(board, belief)

        best_val = -inf
        best_move_key = None
        for mv in moves:
            child_board, child_belief = self._simulate_action(board, belief, mv)
            if child_board is None:
                continue
            child_board.reverse_perspective()
            ext = 1 if self._is_tactical_move(board, belief, mv) and depth <= 3 else 0
            child_depth = max(0, depth - 1 + ext)
            value = self._expectiminimax_opp(
                child_board,
                child_belief,
                child_depth,
                deadline,
                time_left,
            )
            if value > best_val:
                best_val = value
                best_move_key = self._move_key(mv)

        self.ttable[key] = (depth, best_val, TT_EXACT)
        if best_move_key is not None:
            self.tt_move[key] = best_move_key
        return best_val

    def _expectiminimax_opp(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        depth: int,
        deadline: float,
        time_left: Callable,
    ) -> float:
        if time_left() <= deadline:
            raise TimeoutError

        if depth <= 0 or board.is_game_over():
            return -self._evaluate(board, belief)

        key = ("OPP", self._tt_key(board, belief))
        hit = self._probe_tt(key, depth)
        if hit is not None:
            return hit

        legal = board.get_valid_moves(exclude_search=True)
        legal = self._drop_len1_carpets(board, legal)
        searches = self._candidate_searches_for_belief(belief)
        scored = []
        for mv in legal:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        for mv in searches:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return -self._evaluate(board, belief)

        scored = scored[: self._opp_chance_width(board)]
        probs = self._opponent_policy_probs(scored)
        adv_weight = self._opp_adversarial_weight(board, scored)

        exp_val = 0.0
        min_val = inf
        saw_child = False
        for prob, (_, mv) in zip(probs, scored):
            child_board, child_belief = self._simulate_action(board, belief, mv)
            if child_board is None:
                continue
            saw_child = True
            child_board.reverse_perspective()
            ext = 1 if self._is_tactical_move(board, belief, mv) and depth <= 3 else 0
            child_depth = max(0, depth - 1 + ext)
            val = self._expectimax_max(
                child_board,
                child_belief,
                child_depth,
                deadline,
                time_left,
            )
            exp_val += prob * val
            if val < min_val:
                min_val = val

        if not saw_child:
            node_val = -self._evaluate(board, belief)
        else:
            node_val = adv_weight * min_val + (1.0 - adv_weight) * exp_val

        self.ttable[key] = (depth, node_val, TT_EXACT)
        return node_val

    def _simulate_action(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        mv: move.Move,
    ) -> Tuple[game_board.Board | None, List[float]]:
        child = board.forecast_move(mv)
        if child is None:
            return None, list(belief)

        next_belief = list(belief)
        if mv.move_type == enums.MoveType.SEARCH:
            idx = self._pos_to_idx(mv.search_loc)
            p_hit = belief[idx]

            child.player_worker.points += self._search_ev(p_hit)

            failed = list(belief)
            failed[idx] = 0.0
            if sum(failed) <= 0.0:
                failed = list(self.search_prior)
            else:
                self._normalize(failed)

            next_belief = [
                p_hit * self.search_prior[i] + (1.0 - p_hit) * failed[i]
                for i in range(CELL_COUNT)
            ]

        next_belief = self._advance_belief_once(next_belief)
        return child, next_belief

    # ------------------------------------------------------------------
    # Evaluation — enhanced over LeBlanc baseline
    # ------------------------------------------------------------------

    def _evaluate(self, board: game_board.Board, belief: Sequence[float]) -> float:
        if board.get_winner() == enums.Result.PLAYER:
            return 100000.0
        if board.get_winner() == enums.Result.ENEMY:
            return -100000.0
        if board.get_winner() == enums.Result.TIE:
            return 0.0

        turns_left = board.player_worker.turns_left
        my_points = float(board.player_worker.get_points())
        opp_points = float(board.opponent_worker.get_points())
        point_margin = my_points - opp_points

        my_now = self._best_immediate_carpet_points(board, enemy=False)
        opp_now = self._best_immediate_carpet_points(board, enemy=True)
        my_next = self._future_carpet_potential(board, enemy=False)
        opp_next = self._future_carpet_potential(board, enemy=True)

        my_access = self._carpet_accessibility(board, enemy=False)
        opp_access = self._carpet_accessibility(board, enemy=True)

        my_mob = len(board.get_valid_moves(exclude_search=True))
        opp_mob = len(board.get_valid_moves(enemy=True, exclude_search=True))

        block_term = self._blocking_term(board)
        search_term = max(0.0, self._best_search_ev(belief))
        threat_gap = max(0.0, opp_now - my_now)

        score = 36.0 * point_margin
        score += 16.0 * my_now - 24.0 * opp_now
        score += 5.0 * my_next - 9.5 * opp_next
        score += 5.5 * my_access - 8.0 * opp_access
        score += 1.7 * (my_mob - opp_mob)
        score += block_term
        score += 0.55 * search_term

        if my_now >= 4.0:
            score += 28.0
        if my_now >= 6.0:
            score += 42.0
        if my_now >= 10.0:
            score += 56.0
        if opp_now >= 4.0:
            score -= 44.0
        if opp_now >= 6.0:
            score -= 82.0
        if opp_now >= 10.0:
            score -= 132.0

        score += 12.0 * (my_now - opp_now)
        score -= 8.0 * max(0.0, opp_access - my_access)

        if turns_left <= 16:
            score += 10.0 * my_now
            score -= 18.0 * opp_now
            score += 5.0 * my_access
            score -= 8.0 * opp_access
        if turns_left <= 10:
            score += 14.0 * my_now
            score -= 25.0 * opp_now
            score += 7.0 * my_access
            score -= 12.0 * opp_access
            score -= 8.0 * max(0.0, opp_next - my_next)
        if turns_left <= 6:
            score += 18.0 * my_now
            score -= 32.0 * opp_now
            score += 10.0 * my_access
            score -= 16.0 * opp_access
            score -= 12.0 * threat_gap
            score -= 2.5 * search_term

        if point_margin < -4:
            score += 8.0 * (my_now - opp_now)
            score += 5.0 * (my_access - opp_access)
            score -= 8.0 * threat_gap

        return score

    def _carpet_accessibility(self, board: game_board.Board, enemy: bool) -> float:
        """Best carpet-run potential reachable in one step from current position."""
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        best = self._future_carpet_points_from(board, loc)
        for d in DIRS:
            neighbor = enums.loc_after_direction(loc, d)
            if not board.is_valid_cell(neighbor):
                continue
            if board.get_cell(neighbor) == enums.Cell.BLOCKED:
                continue
            pts = self._future_carpet_points_from(board, neighbor)
            best = max(best, pts)
        return best

    # ------------------------------------------------------------------
    # Move heuristic — enhanced PLAIN move scoring
    # ------------------------------------------------------------------

    def _move_heuristic(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        mv: move.Move,
    ) -> float:
        if mv.move_type == enums.MoveType.SEARCH:
            idx = self._pos_to_idx(mv.search_loc)
            return self._search_quick_score(board, belief, idx)

        denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
        my_now = self._best_immediate_carpet_points(board, enemy=False)
        opp_now = self._best_enemy_immediate_carpet_points(board)

        if mv.move_type == enums.MoveType.CARPET:
            pts = float(CARPET_VALUES[mv.roll_length])
            contested = self._is_contested_cash_move(board, mv)
            score = 450.0 + 162.0 * pts + 20.0 * mv.roll_length
            score += 150.0 * denied_now + 98.0 * denied_total
            if contested:
                score += 230.0
            if pts <= 0.0:
                score += 185.0 * denied_total - 130.0
            elif pts == 2.0:
                score -= 8.0
            elif pts == 4.0:
                score += 42.0
            if pts >= opp_now and pts >= 4.0:
                score += 180.0
            if pts >= my_now and pts >= 4.0:
                score += 95.0
            if pts >= 10.0:
                score += 230.0
            if opp_now >= 6.0 and after_now <= max(0.0, opp_now - 4.0):
                score += 160.0
            return score

        current = board.player_worker.get_location()
        dest = enums.loc_after_direction(current, mv.direction)
        leave_cash_pen = 0.0
        if my_now >= 4.0 and after_now >= max(4.0, my_now - 2.0):
            leave_cash_pen += 115.0 * my_now + 70.0 * after_now
        if opp_now >= 4.0 and denied_now <= 0.5:
            leave_cash_pen += 45.0 * opp_now

        if mv.move_type == enums.MoveType.PRIME:
            future = self._future_carpet_points_from(board, dest)
            opp_loc = board.opponent_worker.get_location()
            opp_dist = abs(current[0] - opp_loc[0]) + abs(current[1] - opp_loc[1])
            if opp_dist >= 4:
                chain_val = self._chain_value_through(board, current, mv.direction)
                future = max(future, chain_val)
            score = (
                232.0
                + 30.0 * future
                + self._block_bonus_at(board, dest)
                + 9.0 * self._open_neighbors(board, dest)
                + 86.0 * denied_now
                + 56.0 * denied_total
            )
            if board.player_worker.turns_left <= 10 and future < 4.0:
                score -= 20.0
            score -= leave_cash_pen
            return score

        score = (
            92.0
            + self._block_bonus_at(board, dest)
            + 8.0 * self._open_neighbors(board, dest)
            + 4.0 * self._belief_mass_near(dest, belief)
            + 7.0 * self._future_carpet_points_from(board, dest)
            + 80.0 * denied_now
            + 54.0 * denied_total
        )
        score -= leave_cash_pen
        return score

    def _search_quick_score(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        idx: int,
    ) -> float:
        p = belief[idx]
        raw_ev = self._search_ev(p)
        repeat_pen = self._search_repeat_penalty(board, idx)
        ev = raw_ev - repeat_pen
        my_now = self._best_immediate_carpet_points(board, enemy=False)
        opp_now = self._best_immediate_carpet_points(board, enemy=True)
        turns_left = board.player_worker.turns_left

        if ev > 0.0:
            score = 410.0 + 735.0 * ev + 215.0 * p
        else:
            score = -130.0 + 120.0 * p + 90.0 * ev
        if repeat_pen > 0.0:
            score -= 250.0 * repeat_pen
        if my_now >= 4.0:
            score -= 140.0 * my_now
        if opp_now >= 4.0:
            score -= 110.0 * opp_now
        if opp_now >= 6.0:
            score -= 180.0
        if turns_left <= 12:
            score -= 95.0 * max(my_now, opp_now)
        if turns_left <= 6 and ev < 1.8:
            score -= 210.0
        if turns_left <= 4 and ev < 2.6:
            score -= 260.0
        if turns_left <= 2 and ev < 3.9:
            score -= 560.0
        return score

    def _best_board_move_heuristic(
        self,
        board: game_board.Board,
        belief: Sequence[float],
    ) -> float:
        legal = board.get_valid_moves(exclude_search=True)
        if not legal:
            return -inf
        return max(self._move_heuristic(board, belief, mv) for mv in legal)

    def _best_quick_non_search_score(
        self,
        board: game_board.Board,
        belief: Sequence[float],
    ) -> float:
        legal = board.get_valid_moves(exclude_search=True)
        if not legal:
            return -inf
        return max(self._move_heuristic(board, belief, mv) for mv in legal)

    def _node_moves(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        legal: Sequence[move.Move],
        searches: Sequence[move.Move],
        tt_key: Tuple,
    ) -> List[move.Move]:
        scored = []
        for mv in legal:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        for mv in searches:
            scored.append((self._move_heuristic(board, belief, mv), mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        moves = [mv for _, mv in scored[: self._node_width(board)]]
        tt_best = self.tt_move.get(tt_key)
        if tt_best is not None:
            for i, mv in enumerate(moves):
                if self._move_key(mv) == tt_best:
                    if i != 0:
                        moves[0], moves[i] = moves[i], moves[0]
                    break
        return moves

    def _opp_chance_width(self, board: game_board.Board) -> int:
        turns_left = board.player_worker.turns_left
        if turns_left <= 6:
            return 4
        if turns_left <= 14:
            return 3
        return 2

    def _opponent_policy_probs(
        self,
        scored_moves: Sequence[Tuple[float, move.Move]],
    ) -> List[float]:
        if not scored_moves:
            return []
        best = scored_moves[0][0]
        tau = 85.0
        raw = []
        total = 0.0
        from math import exp

        for score, _ in scored_moves:
            x = max(-8.0, min(0.0, (score - best) / tau))
            w = exp(x)
            raw.append(w)
            total += w
        if total <= 0.0:
            return [1.0 / len(scored_moves)] * len(scored_moves)
        return [w / total for w in raw]

    def _opp_adversarial_weight(
        self,
        board: game_board.Board,
        scored_moves: Sequence[Tuple[float, move.Move]],
    ) -> float:
        turns_left = board.player_worker.turns_left
        immediate = self._best_immediate_carpet_points(board, enemy=False)
        access = self._carpet_accessibility(board, enemy=False)
        gap = 0.0
        if len(scored_moves) >= 2:
            gap = scored_moves[0][0] - scored_moves[1][0]

        weight = 0.56
        if turns_left <= 18:
            weight += 0.05
        if turns_left <= 10:
            weight += 0.08
        if turns_left <= 6:
            weight += 0.10
        if immediate >= 4.0:
            weight += 0.06
        if immediate >= 6.0:
            weight += 0.10
        if immediate >= 10.0:
            weight += 0.06
        if access >= 6.0:
            weight += 0.05
        if gap >= 120.0:
            weight += 0.07
        elif gap <= 35.0:
            weight -= 0.05
        return max(0.35, min(0.90, weight))

    def _max_search_depth(
        self,
        board: game_board.Board,
        ordered_moves: Sequence[move.Move],
        belief: Sequence[float],
    ) -> int:
        if not ordered_moves:
            return 6
        if len(ordered_moves) == 1:
            return 8
        first = self._move_heuristic(board, belief, ordered_moves[0])
        second = self._move_heuristic(board, belief, ordered_moves[1])
        gap = first - second
        turns_left = board.player_worker.turns_left
        base = 6
        if turns_left <= 14:
            base += 1
        if turns_left <= 8:
            base += 1
        if gap >= 260.0:
            base += 1
        elif gap <= 60.0:
            base -= 1
        return max(5, min(8, base))

    def _root_width(self, board: game_board.Board) -> int:
        return self._root_move_limit(board)

    def _node_width(self, board: game_board.Board) -> int:
        return self._node_move_limit(board)

    def _root_move_limit(self, board: game_board.Board) -> int:
        my_turn_index = 40 - board.player_worker.turns_left
        if my_turn_index == 0:
            return 6
        if my_turn_index == 1:
            return 7
        if my_turn_index == 2:
            return 8
        if my_turn_index < 8:
            return 9
        if board.player_worker.turns_left <= 6:
            return 9
        if board.player_worker.turns_left <= 10:
            return 10
        if my_turn_index < 24:
            return 10
        return 9

    def _node_move_limit(self, board: game_board.Board) -> int:
        my_turn_index = 40 - board.player_worker.turns_left
        if my_turn_index == 0:
            return 4
        if my_turn_index == 1:
            return 5
        if my_turn_index == 2:
            return 6
        if my_turn_index < 8:
            return 7
        if board.player_worker.turns_left <= 6:
            return 5
        if board.player_worker.turns_left <= 10:
            return 6
        if my_turn_index < 24:
            return 7
        return 6

    def _turn_budget(self, board: game_board.Board, time_remaining: float) -> float:
        turns_left = max(1, board.player_worker.turns_left)
        if time_remaining > 150.0:
            reserve = 48.0
        elif time_remaining > 90.0:
            reserve = 32.0
        elif time_remaining > 40.0:
            reserve = 18.0
        else:
            reserve = 7.0

        spendable = max(0.4, time_remaining - reserve)
        base = spendable / turns_left
        turn_idx = 40 - turns_left

        if turn_idx == 0:
            boost = 0.42
        elif turn_idx == 1:
            boost = 0.62
        elif turn_idx == 2:
            boost = 0.82
        elif turn_idx < 8:
            boost = 1.10
        elif turn_idx < 24:
            boost = 1.36
        else:
            boost = 1.16

        if self._best_search_ev(self.belief) > 1.0:
            boost += 0.10
        if self._best_enemy_immediate_carpet_points(board) >= 6.0:
            boost += 0.12
        if self._best_immediate_carpet_points(board, enemy=False) >= 6.0:
            boost += 0.10
        if turns_left <= 10:
            boost += 0.14
        if turns_left <= 6:
            boost += 0.10

        return min(9.5, max(1.10, base * boost))

    def _probe_tt(
        self,
        key: Tuple,
        depth: int,
    ) -> float | None:
        hit = self.ttable.get(key)
        if hit is None:
            return None
        d, val, flag = hit
        if d < depth or flag != TT_EXACT:
            return None
        return val

    def _tt_key(self, board: game_board.Board, belief: Sequence[float]) -> Tuple:
        top = sorted(range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True)[:5]
        belief_sig = tuple((idx, round(belief[idx], 3)) for idx in top)
        return (
            board._primed_mask,
            board._carpet_mask,
            board._blocked_mask,
            board.player_worker.position,
            board.opponent_worker.position,
            board.player_worker.get_points(),
            board.opponent_worker.get_points(),
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
            belief_sig,
        )

    # ------------------------------------------------------------------
    # Board geometry helpers
    # ------------------------------------------------------------------

    def _enemy_lane_cells(self, board: game_board.Board) -> List[Tuple[int, int]]:
        opp_loc = board.opponent_worker.get_location()
        out: List[Tuple[int, int]] = []
        for d in DIRS:
            cur = opp_loc
            while True:
                cur = enums.loc_after_direction(cur, d)
                if not board.is_valid_cell(cur) or not board.is_cell_carpetable(cur):
                    break
                out.append(cur)
        return out

    def _best_immediate_carpet_points(
        self, board: game_board.Board, enemy: bool
    ) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        best = 0.0
        for d in DIRS:
            cur = loc
            run = 0
            while True:
                cur = enums.loc_after_direction(cur, d)
                if not board.is_valid_cell(cur) or not board.is_cell_carpetable(cur):
                    break
                run += 1
            best = max(best, float(CARPET_VALUES.get(run, 0)))
        return best

    def _best_enemy_immediate_carpet_points(self, board: game_board.Board) -> float:
        return self._best_immediate_carpet_points(board, enemy=True)

    def _future_carpet_points_from(
        self, board: game_board.Board, loc: Tuple[int, int]
    ) -> float:
        if not board.is_valid_cell(loc) or board.get_cell(loc) != enums.Cell.SPACE:
            return 0.0
        best_run = 0
        for d in DIRS:
            cur = enums.loc_after_direction(loc, d)
            run = 0
            while board.is_valid_cell(cur) and board.is_cell_carpetable(cur):
                run += 1
                cur = enums.loc_after_direction(cur, d)
            best_run = max(best_run, run + 1)
        return float(CARPET_VALUES.get(best_run, 0))

    def _chain_value_through(
        self,
        board: game_board.Board,
        current: Tuple[int, int],
        direction,
    ) -> float:
        opposite = {
            enums.Direction.UP: enums.Direction.DOWN,
            enums.Direction.DOWN: enums.Direction.UP,
            enums.Direction.LEFT: enums.Direction.RIGHT,
            enums.Direction.RIGHT: enums.Direction.LEFT,
        }.get(direction)
        if opposite is None:
            return 0.0
        run = 1
        cur = enums.loc_after_direction(current, opposite)
        while board.is_valid_cell(cur) and board.is_cell_carpetable(cur):
            run += 1
            cur = enums.loc_after_direction(cur, opposite)
        return float(CARPET_VALUES.get(run + 1, 0))

    def _future_carpet_potential(self, board: game_board.Board, enemy: bool) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        return self._future_carpet_points_from(board, worker.get_location())

    def _blocking_term(self, board: game_board.Board) -> float:
        lane_cells = self._enemy_lane_cells(board)
        if not lane_cells:
            return 0.0
        my_loc = board.player_worker.get_location()
        best = min(abs(my_loc[0] - x) + abs(my_loc[1] - y) for x, y in lane_cells)
        opp_threat = self._best_enemy_immediate_carpet_points(board)
        # Scale blocking reward with threat level
        threat_mult = 1.5 if opp_threat >= 6.0 else 1.0
        if best == 0:
            return 8.0 * threat_mult
        if best == 1:
            return 3.0 * threat_mult
        return -1.2 * best

    def _block_bonus_at(self, board: game_board.Board, dest: Tuple[int, int]) -> float:
        lane_cells = self._enemy_lane_cells(board)
        if not lane_cells:
            return 0.0
        if dest in lane_cells:
            return 110.0
        best = min(abs(dest[0] - x) + abs(dest[1] - y) for x, y in lane_cells)
        return max(0.0, 70.0 - 18.0 * best)

    def _open_neighbors(self, board: game_board.Board, loc: Tuple[int, int]) -> int:
        count = 0
        for d in DIRS:
            nxt = enums.loc_after_direction(loc, d)
            if not board.is_cell_blocked(nxt):
                count += 1
        return count

    def _belief_mass_near(self, loc: Tuple[int, int], belief: Sequence[float]) -> float:
        x, y = loc
        total = 0.0
        for idx, p in enumerate(belief):
            if p <= 0.0:
                continue
            rx, ry = self._idx_to_pos(idx)
            dist = abs(x - rx) + abs(y - ry)
            total += p * max(0, 4 - dist)
        return total

    def _search_repeat_penalty(self, board: game_board.Board, idx: int) -> float:
        last_turn = self._failed_search_turns.get(idx)
        if last_turn is None:
            return 0.0
        age = board.turn_count - last_turn
        if age <= 2:
            return 3.2
        if age <= 4:
            return 1.8
        if age <= 6:
            return 0.8
        return 0.0

    def _search_ev(self, p: float) -> float:
        return 6.0 * p - 2.0

    def _opp_threat_metrics(self, board: game_board.Board) -> Tuple[float, float, float]:
        opp_now = self._best_immediate_carpet_points(board, enemy=True)
        opp_next = self._future_carpet_potential(board, enemy=True)
        opp_access = self._carpet_accessibility(board, enemy=True)
        return opp_now, opp_next, opp_access

    def _opp_threat_score(
        self,
        board: game_board.Board | None = None,
        metrics: Tuple[float, float, float] | None = None,
    ) -> float:
        if metrics is None:
            assert board is not None
            metrics = self._opp_threat_metrics(board)
        opp_now, opp_next, opp_access = metrics
        return opp_now + 0.65 * opp_next + 0.45 * opp_access

    def _move_denial_delta(
        self,
        board: game_board.Board,
        mv: move.Move,
    ) -> Tuple[float, float, float, float]:
        cache_key = (
            board._primed_mask,
            board._carpet_mask,
            board._blocked_mask,
            board.player_worker.position,
            board.opponent_worker.position,
            board.player_worker.get_points(),
            board.opponent_worker.get_points(),
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
            self._move_key(mv),
        )
        hit = self._denial_cache.get(cache_key)
        if hit is not None:
            return hit

        before_metrics = self._opp_threat_metrics(board)
        before_now = before_metrics[0]
        before_total = self._opp_threat_score(metrics=before_metrics)
        child = board.forecast_move(mv)
        if child is None:
            out = (0.0, 0.0, before_now, before_total)
            self._denial_cache[cache_key] = out
            return out

        after_metrics = self._opp_threat_metrics(child)
        after_now = after_metrics[0]
        after_total = self._opp_threat_score(metrics=after_metrics)
        out = (
            max(0.0, before_now - after_now),
            max(0.0, before_total - after_total),
            after_now,
            after_total,
        )
        self._denial_cache[cache_key] = out
        return out

    def _best_denial_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
    ) -> move.Move | None:
        threat = self._best_enemy_immediate_carpet_points(board)
        best = None
        best_score = -inf
        for mv in legal_moves:
            denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
            if denied_now <= 0.0 and denied_total < 2.0:
                continue
            local = 0.0
            if mv.move_type == enums.MoveType.CARPET:
                local = float(CARPET_VALUES[mv.roll_length])
            elif mv.move_type == enums.MoveType.PRIME:
                local = 1.0
            score = 180.0 * denied_now + 120.0 * denied_total + 36.0 * local
            if mv.move_type == enums.MoveType.CARPET and local <= 0.0:
                score += 100.0 * denied_total
            if threat >= 6.0 and after_now <= max(0.0, threat - 4.0):
                score += 130.0
            if score > best_score:
                best_score = score
                best = mv
        return best

    def _best_cash_now_move(
        self,
        board: game_board.Board,
        legal_moves: Sequence[move.Move],
    ) -> move.Move | None:
        best = None
        best_score = -inf
        for mv in legal_moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            pts = float(CARPET_VALUES[mv.roll_length])
            if pts < 4.0:
                continue
            denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
            contested = self._is_contested_cash_move(board, mv)
            score = 210.0 * pts + 160.0 * denied_now + 105.0 * denied_total
            score += 28.0 * mv.roll_length
            if contested:
                score += 220.0
            if after_now <= max(0.0, self._best_enemy_immediate_carpet_points(board) - 2.0):
                score += 90.0
            if score > best_score:
                best_score = score
                best = mv
        return best

    def _is_contested_cash_move(
        self,
        board: game_board.Board,
        mv: move.Move,
    ) -> bool:
        if mv.move_type != enums.MoveType.CARPET:
            return False
        pts = float(CARPET_VALUES[mv.roll_length])
        if pts < 4.0:
            return False
        before_now = self._best_enemy_immediate_carpet_points(board)
        denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
        if before_now >= pts:
            return True
        if denied_now >= 2.0:
            return True
        if denied_total >= 3.5:
            return True
        if after_now <= max(0.0, before_now - 2.0):
            return True
        return False

    def _drop_len1_carpets(
        self,
        board: game_board.Board,
        moves: Sequence[move.Move],
    ) -> List[move.Move]:
        kept = []
        len1 = []
        for mv in moves:
            if mv.move_type == enums.MoveType.CARPET and mv.roll_length == 1:
                len1.append(mv)
            else:
                kept.append(mv)

        if not len1:
            return list(moves)
        if not kept:
            return list(moves)

        threat = self._best_enemy_immediate_carpet_points(board)
        for mv in len1:
            denied_now, denied_total, after_now, _ = self._move_denial_delta(board, mv)
            if denied_now >= 4.0 or denied_total >= 4.8:
                kept.append(mv)
            elif threat >= 6.0 and after_now <= max(0.0, threat - 4.0):
                kept.append(mv)
        return kept if kept else list(moves)

    def _is_quiet_move(self, mv: move.Move) -> bool:
        if mv.move_type == enums.MoveType.CARPET:
            return mv.roll_length <= 2
        return mv.move_type in (enums.MoveType.PLAIN, enums.MoveType.PRIME)

    def _is_tactical_move(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        mv: move.Move,
    ) -> bool:
        if mv.move_type == enums.MoveType.CARPET:
            if mv.roll_length >= 3:
                return True
            return self._is_contested_cash_move(board, mv)
        if mv.move_type == enums.MoveType.SEARCH:
            p = belief[self._pos_to_idx(mv.search_loc)]
            return p >= 0.7
        if mv.move_type in (enums.MoveType.PLAIN, enums.MoveType.PRIME):
            cur = board.player_worker.get_location()
            dest = enums.loc_after_direction(cur, mv.direction)
            denied_now, denied_total, _, _ = self._move_denial_delta(board, mv)
            return self._block_bonus_at(board, dest) >= 95.0 or denied_now >= 2.0 or denied_total >= 3.0
        return False

    def _best_search_ev(self, belief: Sequence[float]) -> float:
        return max((self._search_ev(p) for p in belief), default=-2.0)

    def _belief_entropy_bucket(self, belief: Sequence[float]) -> int:
        top_mass = sum(sorted(belief, reverse=True)[:5])
        if top_mass >= 0.9:
            return 0
        if top_mass >= 0.75:
            return 1
        if top_mass >= 0.6:
            return 2
        return 3

    def _candidate_searches_for_belief(
        self,
        belief: Sequence[float],
    ) -> List[move.Move]:
        top = sorted(range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True)[:5]
        out: List[move.Move] = []
        for idx in top:
            p = belief[idx]
            if p >= 0.40 or (idx == top[0] and p >= 0.34):
                out.append(move.Move.search(self._idx_to_pos(idx)))
        return out

    def _move_key(self, mv: move.Move) -> Tuple:
        if mv.move_type == enums.MoveType.SEARCH:
            return (mv.move_type, mv.search_loc)
        if mv.move_type == enums.MoveType.CARPET:
            return (mv.move_type, mv.direction, mv.roll_length)
        return (mv.move_type, mv.direction)

    def _promote_best_move(
        self, moves: Sequence[move.Move], best_move: move.Move
    ) -> List[move.Move]:
        out = [best_move]
        for mv in moves:
            if mv is not best_move:
                out.append(mv)
        return out

    def _best_belief_cell(self, belief: Sequence[float]) -> Tuple[int, int]:
        idx = max(range(CELL_COUNT), key=lambda i: belief[i])
        return self._idx_to_pos(idx)

    def _normalize(self, vec: List[float]):
        total = sum(vec)
        if total <= 0.0:
            uniform = 1.0 / CELL_COUNT
            for i in range(CELL_COUNT):
                vec[i] = uniform
            return
        inv = 1.0 / total
        for i in range(CELL_COUNT):
            vec[i] *= inv

    def _pos_to_idx(self, loc: Tuple[int, int]) -> int:
        return loc[1] * BOARD_SIZE + loc[0]

    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        return (idx % BOARD_SIZE, idx // BOARD_SIZE)
