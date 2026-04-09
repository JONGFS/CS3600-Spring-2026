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
        self.ttable: Dict[Tuple, Tuple[int, float]] = {}
        self.nodes = 0
        self.cutoffs = 0
        self.max_depth_reached = 0
        self.turn_depths: List[int] = []
        self.turn_times: List[float] = []
        self.last_player_search = (None, False)
        self.last_opponent_search = (None, False)
        self.last_turn_seen = -1

    def commentate(self):
        # if not self.turn_depths:
        #     return "Trump: no turns played"
        # avg_depth = sum(self.turn_depths) / len(self.turn_depths)
        # avg_time = sum(self.turn_times) / len(self.turn_times)
        # return (
        #     f"Trump: avg_depth={avg_depth:.2f}, max_depth={self.max_depth_reached}, "
        #     f"avg_time={avg_time:.2f}s, nodes={self.nodes}, cutoffs={self.cutoffs}"
        # )
        return ""

    def play(
        self,
        board: game_board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self._synchronize_belief(board, sensor_data)

        legal_moves = board.get_valid_moves()
        candidate_searches = self._candidate_search_moves(board)
        if not legal_moves and candidate_searches:
            return candidate_searches[0]
        if not legal_moves:
            return move.Move.search(self._best_belief_cell())

        start_remaining = time_left()
        budget = self._turn_budget(board, start_remaining)
        deadline = max(0.02, start_remaining - budget)

        ordered_moves = self._ordered_root_moves(board, legal_moves, candidate_searches)
        best_move = ordered_moves[0]
        best_value = -inf
        completed_depth = 0
        self.ttable.clear()

        for depth in range(1, 16):
            if time_left() <= max(0.02, deadline):
                break
            try:
                value, chosen = self._search_root(
                    board,
                    self.belief,
                    ordered_moves,
                    depth,
                    deadline,
                    time_left,
                )
            except TimeoutError:
                break
            if chosen is not None:
                best_move = chosen
                best_value = value
                completed_depth = depth
                ordered_moves = self._promote_best_move(ordered_moves, chosen)

        if completed_depth == 0:
            best_move = ordered_moves[0]

        self.turn_depths.append(completed_depth)
        self.max_depth_reached = max(self.max_depth_reached, completed_depth)
        self.turn_times.append(max(0.0, start_remaining - time_left()))
        self.last_turn_seen = board.turn_count
        return best_move

    def _build_sparse_transitions(
        self, transition_matrix
    ) -> List[List[Tuple[int, float]]]:
        transitions: List[List[Tuple[int, float]]] = []
        for i in range(CELL_COUNT):
            row = []
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
                        estimate = actual + offset
                        if estimate < 0:
                            estimate = 0
                        if estimate == observed:
                            p += prob
                    row[rat_idx] = p
                per_observation.append(row)
            likelihoods.append(per_observation)
        return likelihoods

    def _synchronize_belief(self, board: game_board.Board, sensor_data: Tuple):
        if board.turn_count == 0 and self.last_turn_seen == -1:
            self.belief = list(self.search_prior)

        self._apply_search_feedback(board.player_search)
        self._apply_search_feedback(board.opponent_search)

        noise_obs, distance_obs = sensor_data
        predicted = self._advance_belief_once(self.belief)
        worker_idx = self._pos_to_idx(board.player_worker.get_location())
        distance_obs = int(max(0, min(MAX_DISTANCE_OBS, distance_obs)))
        distance_row = self.distance_likelihood[worker_idx][distance_obs]
        updated = [0.0] * CELL_COUNT
        total = 0.0
        for idx, prior in enumerate(predicted):
            if prior <= 0.0:
                continue
            cell_prob = NOISE_PROBS[board.get_cell(self._idx_to_pos(idx))][
                int(noise_obs)
            ]
            value = prior * cell_prob * distance_row[idx]
            updated[idx] = value
            total += value

        if total <= 0.0:
            self.belief = list(predicted)
            self._normalize(self.belief)
        else:
            inv_total = 1.0 / total
            self.belief = [value * inv_total for value in updated]

        self.last_player_search = board.player_search
        self.last_opponent_search = board.opponent_search

    def _apply_search_feedback(self, search_info: Tuple):
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
            return
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

    def _candidate_search_moves(self, board: game_board.Board) -> List[move.Move]:
        top_cells = sorted(
            range(CELL_COUNT), key=lambda idx: self.belief[idx], reverse=True
        )[:5]
        searches = []
        for idx in top_cells:
            p = self.belief[idx]
            if p < 0.20 and searches:
                continue
            if p < (1.0 / 3.0) and len(searches) >= 2:
                continue
            searches.append(move.Move.search(self._idx_to_pos(idx)))
        return searches

    def _ordered_root_moves(
        self,
        board: game_board.Board,
        legal_moves: List[move.Move],
        candidate_searches: List[move.Move],
    ) -> List[move.Move]:
        moves = list(legal_moves)
        moves.extend(candidate_searches)
        scored = [(self._move_heuristic(board, mv, self.belief), mv) for mv in moves]
        scored.sort(key=lambda item: item[0], reverse=True)
        root_width = self._root_move_limit(board)
        return [mv for _, mv in scored[:root_width]]

    def _search_root(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        ordered_moves: Sequence[move.Move],
        depth: int,
        deadline: float,
        time_left: Callable,
    ) -> Tuple[float, move.Move | None]:
        alpha = -inf
        beta = inf
        best_move = None
        best_value = -inf
        for mv in ordered_moves:
            if time_left() <= deadline:
                raise TimeoutError
            child_board, child_belief = self._simulate_action(board, belief, mv)
            if child_board is None:
                continue
            child_board.reverse_perspective()
            value = -self._negamax(
                child_board,
                child_belief,
                depth - 1,
                -beta,
                -alpha,
                deadline,
                time_left,
            )
            if value > best_value:
                best_value = value
                best_move = mv
            if value > alpha:
                alpha = value
        return best_value, best_move

    def _negamax(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
        time_left: Callable,
    ) -> float:
        if time_left() <= deadline:
            raise TimeoutError

        self.nodes += 1
        if depth <= 0 or board.is_game_over():
            return self._evaluate(board, belief)

        key = self._transposition_key(board, belief)
        cached = self.ttable.get(key)
        if cached is not None and cached[0] >= depth:
            return cached[1]

        legal_moves = board.get_valid_moves()
        search_moves = self._candidate_searches_for_belief(belief)
        all_moves = self._order_moves_for_node(board, belief, legal_moves, search_moves)
        if not all_moves:
            return self._evaluate(board, belief)

        best_value = -inf
        local_alpha = alpha
        for mv in all_moves:
            child_board, child_belief = self._simulate_action(board, belief, mv)
            if child_board is None:
                continue
            child_board.reverse_perspective()
            value = -self._negamax(
                child_board,
                child_belief,
                depth - 1,
                -beta,
                -local_alpha,
                deadline,
                time_left,
            )
            if value > best_value:
                best_value = value
            if value > local_alpha:
                local_alpha = value
            if local_alpha >= beta:
                self.cutoffs += 1
                break

        self.ttable[key] = (depth, best_value)
        return best_value

    def _order_moves_for_node(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        legal_moves: List[move.Move],
        search_moves: List[move.Move],
    ) -> List[move.Move]:
        moves = list(legal_moves)
        moves.extend(search_moves)
        moves.sort(key=lambda mv: self._move_heuristic(board, mv, belief), reverse=True)
        return moves[: self._node_move_limit(board)]

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
            p = belief[idx]
            child.player_worker.points += 6.0 * p - 2.0
            failed = list(belief)
            failed[idx] = 0.0
            if sum(failed) <= 0.0:
                failed = list(self.search_prior)
            else:
                self._normalize(failed)
            next_belief = [
                p * self.search_prior[i] + (1.0 - p) * failed[i]
                for i in range(CELL_COUNT)
            ]

        next_belief = self._advance_belief_once(next_belief)
        return child, next_belief

    def _evaluate(self, board: game_board.Board, belief: Sequence[float]) -> float:
        if board.get_winner() == enums.Result.PLAYER:
            return 100000.0
        if board.get_winner() == enums.Result.ENEMY:
            return -100000.0
        if board.get_winner() == enums.Result.TIE:
            return 0.0

        my_points = float(board.player_worker.get_points())
        opp_points = float(board.opponent_worker.get_points())
        score = 28.0 * (my_points - opp_points)

        my_moves = board.get_valid_moves()
        opp_moves = board.get_valid_moves(enemy=True)
        score += 1.4 * (len(my_moves) - len(opp_moves))

        my_best_carpet, my_prime_count = self._line_features(board, enemy=False)
        opp_best_carpet, opp_prime_count = self._line_features(board, enemy=True)
        score += 7.0 * (
            CARPET_VALUES.get(my_best_carpet, 0) - CARPET_VALUES.get(opp_best_carpet, 0)
        )
        score += 1.1 * (my_prime_count - opp_prime_count)

        score += 2.0 * (
            self._future_carpet_potential(board, False)
            - self._future_carpet_potential(board, True)
        )
        score += 12.0 * self._best_search_ev(belief)
        score += 0.8 * (
            self._expected_distance_term(board, belief, False)
            - self._expected_distance_term(board, belief, True)
        )

        return score

    def _line_features(self, board: game_board.Board, enemy: bool) -> Tuple[int, int]:
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        best_carpet = 0
        prime_count = 0
        if board.get_cell(loc) == enums.Cell.SPACE:
            prime_count += 1
        for direction in DIRS:
            current = loc
            run = 0
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board.is_valid_cell(current):
                    break
                if board.is_cell_carpetable(current):
                    run += 1
                    prime_count += 1
                else:
                    break
            if run > best_carpet:
                best_carpet = run
        return best_carpet, prime_count

    def _future_carpet_potential(self, board: game_board.Board, enemy: bool) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        if board.get_cell(loc) != enums.Cell.SPACE:
            return 0.0
        best_after_prime = 0
        for direction in DIRS:
            current = enums.loc_after_direction(loc, direction)
            if board.is_cell_blocked(current):
                continue
            run = 0
            while board.is_valid_cell(current) and board.is_cell_carpetable(current):
                run += 1
                current = enums.loc_after_direction(current, direction)
            best_after_prime = max(best_after_prime, run + 1)
        return float(CARPET_VALUES.get(best_after_prime, 0))

    def _best_search_ev(self, belief: Sequence[float]) -> float:
        return max((6.0 * p - 2.0 for p in belief), default=-2.0)

    def _expected_distance_term(
        self, board: game_board.Board, belief: Sequence[float], enemy: bool
    ) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        wx, wy = worker.get_location()
        total = 0.0
        for idx, prob in enumerate(belief):
            if prob <= 0.0:
                continue
            rx, ry = self._idx_to_pos(idx)
            total += prob * (14 - abs(wx - rx) - abs(wy - ry))
        return total

    def _candidate_searches_for_belief(
        self, belief: Sequence[float]
    ) -> List[move.Move]:
        top_cells = sorted(
            range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True
        )[:3]
        return [
            move.Move.search(self._idx_to_pos(idx))
            for idx in top_cells
            if belief[idx] >= 0.20 or (idx == top_cells[0] and belief[idx] >= 0.15)
        ]

    def _move_heuristic(
        self, board: game_board.Board, mv: move.Move, belief: Sequence[float]
    ) -> float:
        if mv.move_type == enums.MoveType.SEARCH:
            p = belief[self._pos_to_idx(mv.search_loc)]
            ev = 6.0 * p - 2.0
            if ev > 0.0:
                return 1200.0 + 1400.0 * ev + 400.0 * p
            return 120.0 + 250.0 * p + 80.0 * ev
        if mv.move_type == enums.MoveType.CARPET:
            return 700.0 + 100.0 * CARPET_VALUES[mv.roll_length] + 15.0 * mv.roll_length
        if mv.move_type == enums.MoveType.PRIME:
            dest = enums.loc_after_direction(
                board.player_worker.get_location(), mv.direction
            )
            return 350.0 + 25.0 * self._open_neighbors(board, dest)
        dest = enums.loc_after_direction(
            board.player_worker.get_location(), mv.direction
        )
        return (
            100.0
            + 6.0 * self._open_neighbors(board, dest)
            + 12.0 * self._belief_mass_near(dest, belief)
        )

    def _open_neighbors(self, board: game_board.Board, loc: Tuple[int, int]) -> int:
        count = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not board.is_cell_blocked(nxt):
                count += 1
        return count

    def _belief_mass_near(self, loc: Tuple[int, int], belief: Sequence[float]) -> float:
        x, y = loc
        total = 0.0
        for idx, prob in enumerate(belief):
            if prob <= 0.0:
                continue
            rx, ry = self._idx_to_pos(idx)
            dist = abs(x - rx) + abs(y - ry)
            total += prob * max(0, 4 - dist)
        return total

    def _transposition_key(
        self, board: game_board.Board, belief: Sequence[float]
    ) -> Tuple:
        top = sorted(range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True)[:3]
        belief_sig = tuple((idx, round(belief[idx], 3)) for idx in top)
        return (
            board._primed_mask,
            board._carpet_mask,
            board._blocked_mask,
            board.player_worker.position,
            board.opponent_worker.position,
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
            board.is_player_a_turn,
            belief_sig,
        )

    def _turn_budget(self, board: game_board.Board, remaining_time: float) -> float:
        turns_left = max(1, board.player_worker.turns_left)
        my_turn_index = 40 - turns_left
        if remaining_time > 120.0:
            reserve = 45.0
        elif remaining_time > 60.0:
            reserve = 28.0
        elif remaining_time > 25.0:
            reserve = 14.0
        else:
            reserve = 5.0
        spendable = max(0.2, remaining_time - reserve)
        base = spendable / turns_left
        if my_turn_index < 8:
            boost = 0.95
        elif my_turn_index < 24:
            boost = 1.3
        else:
            boost = 1.05
        if self._best_search_ev(self.belief) > 0.8:
            boost += 0.15
        return min(10.0, max(0.75, base * boost))

    def _root_move_limit(self, board: game_board.Board) -> int:
        my_turn_index = 40 - board.player_worker.turns_left
        if my_turn_index < 8:
            return 8
        if my_turn_index < 24:
            return 12
        return 10

    def _node_move_limit(self, board: game_board.Board) -> int:
        my_turn_index = 40 - board.player_worker.turns_left
        if my_turn_index < 8:
            return 7
        if my_turn_index < 24:
            return 10
        return 9

    def _promote_best_move(
        self, moves: Sequence[move.Move], best_move: move.Move
    ) -> List[move.Move]:
        promoted = [best_move]
        for mv in moves:
            if mv is not best_move:
                promoted.append(mv)
        return promoted

    def _best_belief_cell(self) -> Tuple[int, int]:
        best_idx = max(range(CELL_COUNT), key=lambda idx: self.belief[idx])
        return self._idx_to_pos(best_idx)

    def _normalize(self, vec: List[float]):
        total = sum(vec)
        if total <= 0.0:
            uniform = 1.0 / CELL_COUNT
            for idx in range(CELL_COUNT):
                vec[idx] = uniform
            return
        inv_total = 1.0 / total
        for idx in range(CELL_COUNT):
            vec[idx] *= inv_total

    def _pos_to_idx(self, loc: Tuple[int, int]) -> int:
        return loc[1] * BOARD_SIZE + loc[0]

    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        return (idx % BOARD_SIZE, idx // BOARD_SIZE)
