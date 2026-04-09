from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import inf, log, sqrt, tanh
from random import random
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
UCB_C = 1.35


@dataclass
class MCTSNode:
    board: game_board.Board
    belief: List[float]
    parent: MCTSNode | None = None
    move_from_parent: move.Move | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Tuple, MCTSNode] = field(default_factory=dict)
    unexpanded_moves: List[move.Move] | None = None

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


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
        self.turn_depths: List[int] = []
        self.turn_times: List[float] = []
        self.sim_counts: List[int] = []

    def commentate(self):
        return ""

    def play(
        self,
        board: game_board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self._synchronize_belief(board, sensor_data)

        legal_moves = board.get_valid_moves()
        candidate_searches = self._candidate_search_moves(board, self.belief)
        root_moves = self._candidate_actions(
            board, self.belief, legal_moves, candidate_searches, True
        )
        if not root_moves:
            return move.Move.search(self._best_belief_cell(self.belief))

        start_remaining = time_left()
        budget = self._turn_budget(board, start_remaining)
        deadline = max(0.02, start_remaining - budget)

        root = MCTSNode(board=board.get_copy(), belief=list(self.belief))
        root.unexpanded_moves = list(root_moves)

        simulations = 0
        max_rollout_depth = 0
        while time_left() > deadline:
            try:
                depth = self._run_simulation(root, deadline, time_left)
            except TimeoutError:
                break
            simulations += 1
            if depth > max_rollout_depth:
                max_rollout_depth = depth

        best_child = self._best_root_child(root)
        if best_child is None:
            best_move = root_moves[0]
        else:
            best_move = best_child.move_from_parent

        self.turn_depths.append(max_rollout_depth)
        self.turn_times.append(max(0.0, start_remaining - time_left()))
        self.sim_counts.append(simulations)
        self.last_turn_seen = board.turn_count
        return best_move

    def _run_simulation(
        self,
        root: MCTSNode,
        deadline: float,
        time_left: Callable,
    ) -> int:
        node = root
        path = [node]
        depth = 0

        while True:
            if time_left() <= deadline:
                raise TimeoutError
            if node.board.is_game_over():
                break
            if node.unexpanded_moves is None:
                node.unexpanded_moves = self._candidate_actions(
                    node.board,
                    node.belief,
                    node.board.get_valid_moves(),
                    self._candidate_search_moves(node.board, node.belief),
                    False,
                )
            if node.unexpanded_moves:
                mv = node.unexpanded_moves.pop(0)
                child = self._expand_child(node, mv)
                node.children[self._move_key(mv)] = child
                node = child
                path.append(node)
                depth += 1
                break
            if not node.children:
                break
            node = self._select_child(node)
            path.append(node)
            depth += 1

        value, rollout_depth = self._rollout(
            node.board.get_copy(), list(node.belief), deadline, time_left
        )
        total_depth = depth + rollout_depth
        sign = 1.0
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += sign * value
            sign *= -1.0
        return total_depth

    def _expand_child(self, node: MCTSNode, mv: move.Move) -> MCTSNode:
        child_board, child_belief = self._simulate_action(node.board, node.belief, mv)
        if child_board is None:
            child_board = node.board.get_copy()
            child_belief = list(node.belief)
        child_board.reverse_perspective()
        return MCTSNode(
            board=child_board,
            belief=child_belief,
            parent=node,
            move_from_parent=mv,
        )

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -inf
        best_child = None
        parent_log = log(max(1, node.visits))
        for child in node.children.values():
            if child.visits == 0:
                return child
            exploit = -child.mean_value()
            explore = UCB_C * sqrt(parent_log / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            return next(iter(node.children.values()))
        return best_child

    def _best_root_child(self, root: MCTSNode) -> MCTSNode | None:
        best_child = None
        best_visits = -1
        best_value = -inf
        for child in root.children.values():
            child_value = -child.mean_value()
            if child.visits > best_visits or (
                child.visits == best_visits and child_value > best_value
            ):
                best_child = child
                best_visits = child.visits
                best_value = child_value
        return best_child

    def _rollout(
        self,
        board: game_board.Board,
        belief: List[float],
        deadline: float,
        time_left: Callable,
    ) -> Tuple[float, int]:
        depth_limit = self._rollout_depth_limit(board)
        for depth in range(depth_limit):
            if time_left() <= deadline:
                raise TimeoutError
            if board.is_game_over():
                return self._normalized_eval(board, belief), depth
            legal = board.get_valid_moves()
            searches = self._candidate_search_moves(board, belief)
            actions = self._candidate_actions(board, belief, legal, searches, False)
            if not actions:
                return self._normalized_eval(board, belief), depth
            mv = self._rollout_policy_move(board, belief, actions)
            board, belief = self._simulate_action(board, belief, mv)
            if board is None:
                return self._normalized_eval(board, belief), depth
            board.reverse_perspective()
        return self._normalized_eval(board, belief), depth_limit

    def _rollout_policy_move(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        actions: Sequence[move.Move],
    ) -> move.Move:
        scored = []
        for mv in actions:
            score = self._rollout_move_score(board, belief, mv)
            scored.append((score, mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: min(3, len(scored))]
        if len(top) > 1 and random() < 0.18:
            return top[1][1]
        if len(top) > 2 and random() < 0.08:
            return top[2][1]
        return top[0][1]

    def _rollout_move_score(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        mv: move.Move,
    ) -> float:
        if mv.move_type == enums.MoveType.CARPET:
            pts = CARPET_VALUES[mv.roll_length]
            score = 180.0 * pts
            if pts <= 0:
                score -= 400.0
            if pts >= 10:
                score += 150.0
            return score
        if mv.move_type == enums.MoveType.SEARCH:
            p = belief[self._pos_to_idx(mv.search_loc)]
            ev = self._search_ev(p)
            board_best = self._best_immediate_carpet_points(board, False)
            opp_best = self._best_immediate_carpet_points(board, True)
            return 60.0 * ev - 40.0 * max(board_best, opp_best)
        if mv.move_type == enums.MoveType.PRIME:
            dest = enums.loc_after_direction(
                board.player_worker.get_location(), mv.direction
            )
            future = self._future_carpet_points_from(board, dest)
            return 80.0 + 35.0 * future + self._prime_block_bonus(board, dest)
        dest = enums.loc_after_direction(
            board.player_worker.get_location(), mv.direction
        )
        return (
            25.0
            + self._plain_block_bonus(board, dest)
            + 8.0 * self._shape_score_at(board, dest)
        )

    def _candidate_actions(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        legal_moves: List[move.Move],
        search_moves: List[move.Move],
        is_root: bool,
    ) -> List[move.Move]:
        moves = list(legal_moves)
        moves.extend(search_moves)
        scored = [(self._move_heuristic(board, mv, belief), mv) for mv in moves]
        scored.sort(key=lambda item: item[0], reverse=True)
        limit = (
            self._root_move_limit(board) if is_root else self._node_move_limit(board)
        )
        return self._preserve_move_categories(board, belief, scored, limit)

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
            child.player_worker.points += self._search_ev(p)
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
        turns_left = board.player_worker.turns_left
        score = 30.0 * (my_points - opp_points)

        my_now = self._best_immediate_carpet_points(board, False)
        opp_now = self._best_immediate_carpet_points(board, True)
        my_future = self._future_carpet_potential(board, False)
        opp_future = self._future_carpet_potential(board, True)

        score += 8.0 * (my_now - opp_now)
        score += 3.8 * (my_future - opp_future)
        score += 6.0 * (
            self._strong_lane_count(board, False) - self._strong_lane_count(board, True)
        )
        score += 2.3 * (
            self._shape_score(board, False) - self._shape_score(board, True)
        )
        score += 2.1 * (
            len(board.get_valid_moves()) - len(board.get_valid_moves(enemy=True))
        )
        score += self._lane_contest_bonus(board)
        score -= 5.0 * self._low_value_carpet_penalty(board, False)
        score += 5.0 * self._low_value_carpet_penalty(board, True)
        score += 2.0 * max(0.0, self._best_search_ev(belief))
        score += 0.35 * (
            self._expected_distance_term(board, belief, False)
            - self._expected_distance_term(board, belief, True)
        )

        if turns_left <= 10:
            score += 10.0 * (my_points - opp_points)
            score += 10.0 * my_now
            score -= 14.0 * opp_now
            score -= 5.0 * opp_future

        if my_now >= 6.0 or opp_now >= 6.0:
            score -= 2.5 * max(0.0, self._best_search_ev(belief))

        return score

    def _normalized_eval(
        self, board: game_board.Board, belief: Sequence[float]
    ) -> float:
        return tanh(self._evaluate(board, belief) / 60.0)

    def _candidate_search_moves(
        self, board: game_board.Board, belief: Sequence[float]
    ) -> List[move.Move]:
        top_cells = sorted(
            range(CELL_COUNT), key=lambda idx: belief[idx], reverse=True
        )[:6]
        searches = []
        board_best = self._best_immediate_carpet_points(board, False)
        opp_best = self._best_immediate_carpet_points(board, True)
        for idx in top_cells:
            p = belief[idx]
            ev = self._search_ev(p)
            if ev <= 0.0:
                continue
            if max(board_best, opp_best) >= 6.0 and ev < 1.0:
                continue
            if p < 0.38 and len(searches) >= 1:
                continue
            searches.append(move.Move.search(self._idx_to_pos(idx)))
            if len(searches) >= 2:
                break
        return searches

    def _move_heuristic(
        self,
        board: game_board.Board,
        mv: move.Move,
        belief: Sequence[float],
    ) -> float:
        if mv.move_type == enums.MoveType.SEARCH:
            p = belief[self._pos_to_idx(mv.search_loc)]
            ev = self._search_ev(p)
            board_best = self._best_immediate_carpet_points(board, False)
            opp_best = self._best_immediate_carpet_points(board, True)
            score = 180.0 + 340.0 * ev - 80.0 * max(board_best, opp_best)
            if ev < 0.7:
                score -= 120.0
            return score
        if mv.move_type == enums.MoveType.CARPET:
            pts = CARPET_VALUES[mv.roll_length]
            score = 420.0 + 150.0 * pts + 20.0 * mv.roll_length
            if pts <= 0:
                score -= 260.0
            elif pts == 2:
                score -= 80.0
            if pts >= self._best_immediate_carpet_points(board, True) and pts >= 4:
                score += 120.0
            return score
        if mv.move_type == enums.MoveType.PRIME:
            dest = enums.loc_after_direction(
                board.player_worker.get_location(), mv.direction
            )
            return (
                300.0
                + 42.0 * self._future_carpet_points_from(board, dest)
                + self._prime_block_bonus(board, dest)
                + 15.0 * self._shape_score_at(board, dest)
            )
        dest = enums.loc_after_direction(
            board.player_worker.get_location(), mv.direction
        )
        return (
            120.0
            + self._plain_block_bonus(board, dest)
            + 12.0 * self._shape_score_at(board, dest)
        )

    def _rollout_depth_limit(self, board: game_board.Board) -> int:
        turns_left = board.player_worker.turns_left
        if turns_left <= 8:
            return 12
        if turns_left <= 18:
            return 10
        return 8

    def _turn_budget(self, board: game_board.Board, remaining_time: float) -> float:
        turns_left = max(1, board.player_worker.turns_left)
        turn_index = 40 - turns_left
        reserve = 18.0 if remaining_time > 60.0 else 8.0
        base = max(0.25, (remaining_time - reserve) / turns_left)
        if turn_index < 3:
            boost = 0.65
        elif turn_index < 10:
            boost = 0.95
        elif turn_index < 26:
            boost = 1.15
        else:
            boost = 1.0
        if self._best_search_ev(self.belief) > 1.0:
            boost += 0.1
        return min(8.0, max(0.8, base * boost))

    def _root_move_limit(self, board: game_board.Board) -> int:
        turn_index = 40 - board.player_worker.turns_left
        if turn_index < 3:
            return 5
        if turn_index < 10:
            return 7
        if board.player_worker.turns_left <= 8:
            return 8
        return 10

    def _node_move_limit(self, board: game_board.Board) -> int:
        turn_index = 40 - board.player_worker.turns_left
        if turn_index < 3:
            return 4
        if turn_index < 10:
            return 6
        if board.player_worker.turns_left <= 8:
            return 7
        return 8

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

    def _best_immediate_carpet_points(
        self, board: game_board.Board, enemy: bool
    ) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        best_points = 0.0
        for direction in DIRS:
            current = loc
            run = 0
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board.is_valid_cell(current) or not board.is_cell_carpetable(
                    current
                ):
                    break
                run += 1
            best_points = max(best_points, float(CARPET_VALUES.get(run, 0)))
        return best_points

    def _future_carpet_points_from(
        self, board: game_board.Board, loc: Tuple[int, int]
    ) -> float:
        if not board.is_valid_cell(loc) or board.get_cell(loc) != enums.Cell.SPACE:
            return 0.0
        best_after_prime = 0
        for direction in DIRS:
            current = enums.loc_after_direction(loc, direction)
            run = 0
            while board.is_valid_cell(current) and board.is_cell_carpetable(current):
                run += 1
                current = enums.loc_after_direction(current, direction)
            best_after_prime = max(best_after_prime, run + 1)
        return float(CARPET_VALUES.get(best_after_prime, 0))

    def _strong_lane_count(self, board: game_board.Board, enemy: bool) -> int:
        worker = board.opponent_worker if enemy else board.player_worker
        loc = worker.get_location()
        count = 0
        for direction in DIRS:
            current = loc
            run = 0
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board.is_valid_cell(current) or not board.is_cell_carpetable(
                    current
                ):
                    break
                run += 1
            if CARPET_VALUES.get(run, 0) >= 4:
                count += 1
        return count

    def _shape_score(self, board: game_board.Board, enemy: bool) -> float:
        worker = board.opponent_worker if enemy else board.player_worker
        return self._shape_score_at(board, worker.get_location())

    def _shape_score_at(self, board: game_board.Board, loc: Tuple[int, int]) -> float:
        score = float(self._open_neighbors(board, loc))
        if board.is_valid_cell(loc) and board.get_cell(loc) == enums.Cell.SPACE:
            score += 0.5
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if board.is_valid_cell(nxt) and board.get_cell(nxt) == enums.Cell.SPACE:
                score += 0.25
        return score

    def _lane_contest_bonus(self, board: game_board.Board) -> float:
        dist = self._block_distance_to_enemy_lane(board)
        if dist is None:
            return 0.0
        if dist == 0:
            return 9.0
        if dist == 1:
            return 4.0
        return -1.2 * dist

    def _low_value_carpet_penalty(self, board: game_board.Board, enemy: bool) -> float:
        points = self._best_immediate_carpet_points(board, enemy)
        if points <= 0:
            return 1.0
        if points <= 2:
            return 0.45
        return 0.0

    def _block_distance_to_enemy_lane(self, board: game_board.Board) -> int | None:
        opp_loc = board.opponent_worker.get_location()
        my_loc = board.player_worker.get_location()
        best_distance = None
        for direction in DIRS:
            current = opp_loc
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board.is_valid_cell(current) or not board.is_cell_carpetable(
                    current
                ):
                    break
                dist = abs(my_loc[0] - current[0]) + abs(my_loc[1] - current[1])
                if best_distance is None or dist < best_distance:
                    best_distance = dist
        return best_distance

    def _enemy_lane_cells(self, board: game_board.Board) -> List[Tuple[int, int]]:
        opp_loc = board.opponent_worker.get_location()
        cells: List[Tuple[int, int]] = []
        for direction in DIRS:
            current = opp_loc
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board.is_valid_cell(current) or not board.is_cell_carpetable(
                    current
                ):
                    break
                cells.append(current)
        return cells

    def _plain_block_bonus(
        self, board: game_board.Board, dest: Tuple[int, int]
    ) -> float:
        lane_cells = self._enemy_lane_cells(board)
        if not lane_cells:
            return 0.0
        if dest in lane_cells:
            return 150.0
        best_dist = min(abs(dest[0] - x) + abs(dest[1] - y) for x, y in lane_cells)
        return max(0.0, 90.0 - 20.0 * best_dist)

    def _prime_block_bonus(
        self, board: game_board.Board, dest: Tuple[int, int]
    ) -> float:
        lane_cells = self._enemy_lane_cells(board)
        if dest in lane_cells:
            return 115.0
        return 0.0

    def _best_search_ev(self, belief: Sequence[float]) -> float:
        return max((self._search_ev(p) for p in belief), default=-2.0)

    def _search_ev(self, p: float) -> float:
        return 6.0 * p - 2.0

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

    def _open_neighbors(self, board: game_board.Board, loc: Tuple[int, int]) -> int:
        count = 0
        for direction in DIRS:
            nxt = enums.loc_after_direction(loc, direction)
            if not board.is_cell_blocked(nxt):
                count += 1
        return count

    def _preserve_move_categories(
        self,
        board: game_board.Board,
        belief: Sequence[float],
        scored_moves: Sequence[Tuple[float, move.Move]],
        limit: int,
    ) -> List[move.Move]:
        if len(scored_moves) <= limit:
            return [mv for _, mv in scored_moves]
        chosen: List[move.Move] = []
        seen = set()
        wants = ["carpet", "block", "prime", "mobility"]
        best_search_score = max(
            (
                score
                for score, mv in scored_moves
                if mv.move_type == enums.MoveType.SEARCH
            ),
            default=-inf,
        )
        if best_search_score > 260.0:
            wants.append("search")
        for category in wants:
            for _, mv in scored_moves:
                key = self._move_key(mv)
                if key in seen:
                    continue
                if self._move_category(board, belief, mv) == category:
                    chosen.append(mv)
                    seen.add(key)
                    break
        for _, mv in scored_moves:
            key = self._move_key(mv)
            if key in seen:
                continue
            chosen.append(mv)
            seen.add(key)
            if len(chosen) >= limit:
                break
        return chosen[:limit]

    def _move_category(
        self, board: game_board.Board, belief: Sequence[float], mv: move.Move
    ) -> str:
        if mv.move_type == enums.MoveType.SEARCH:
            return "search"
        if mv.move_type == enums.MoveType.CARPET:
            return "carpet"
        if mv.move_type == enums.MoveType.PRIME:
            dest = enums.loc_after_direction(
                board.player_worker.get_location(), mv.direction
            )
            if self._prime_block_bonus(board, dest) > 0.0:
                return "block"
            return "prime"
        dest = enums.loc_after_direction(
            board.player_worker.get_location(), mv.direction
        )
        if self._plain_block_bonus(board, dest) >= 60.0:
            return "block"
        if self._shape_score_at(board, dest) >= 3.0:
            return "mobility"
        return "plain"

    def _move_key(self, mv: move.Move) -> Tuple:
        if mv.move_type == enums.MoveType.SEARCH:
            return (mv.move_type, mv.search_loc)
        if mv.move_type == enums.MoveType.CARPET:
            return (mv.move_type, mv.direction, mv.roll_length)
        return (mv.move_type, mv.direction)

    def _best_belief_cell(self, belief: Sequence[float]) -> Tuple[int, int]:
        best_idx = max(range(CELL_COUNT), key=lambda idx: belief[idx])
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
