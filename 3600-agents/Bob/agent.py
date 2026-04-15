from collections.abc import Callable
from typing import List, Optional, Tuple
import math
import heapq
from collections import deque

from game import board, enums, move
from game.enums import CARPET_POINTS_TABLE, RAT_BONUS, RAT_PENALTY, Result
from game.rat import DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS, NOISE_PROBS


class SearchTimeout(Exception):
    """Raised internally to stop search before the move timer expires."""


class PlayerAgent:
    def __init__(
        self, board, transition_matrix=None, time_left: Optional[Callable] = None
    ):
        self.transition_matrix = transition_matrix
        self.max_depth = 3
        self.root_width = 3
        self.node_width = 2
        self.max_search_candidates = 1
        self.belief: Optional[List[float]] = None
        self.root_is_player_a = True
        self.search_prior = self.compute_headstart_prior()

        self._time_left: Optional[Callable] = time_left
        self._time_safety_margin = 0.0

        # Anti-loop / anti-stall state across real turns.
        self.recent_positions = deque(maxlen=8)
        self.last_seen_points: Optional[int] = None
        self.zero_gain_streak = 0

    def commentate(self):
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.root_is_player_a = board.player_worker.is_player_a
        self._configure_timer(time_left)

        current_points = board.player_worker.get_points()
        if self.last_seen_points is not None:
            if current_points <= self.last_seen_points:
                self.zero_gain_streak = min(self.zero_gain_streak + 1, 6)
            else:
                self.zero_gain_streak = 0
        self.last_seen_points = current_points

        current_loc = board.player_worker.get_location()
        self.recent_positions.append(current_loc)

        self.update_belief(board, sensor_data)
        belief = (
            list(self.belief) if self.belief is not None else self.uniform_belief(board)
        )

        legal_moves = self.get_ordered_moves(
            board, belief, is_max_player=True, root=True
        )
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        if self._very_low_time():
            return best_move

        for depth in range(1, self.max_depth + 1):
            try:
                current_best_move = best_move
                current_best_score = -math.inf

                for candidate in legal_moves:
                    self._check_timeout()
                    score = self.evaluate_move(
                        board,
                        belief,
                        candidate,
                        depth,
                        is_max_player=True,
                    )
                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = candidate

                best_move = current_best_move
                legal_moves.sort(
                    key=lambda candidate: (
                        candidate != best_move,
                        -self.move_order_score(board, belief, candidate, True),
                    )
                )
            except SearchTimeout:
                break

        return best_move

    def _configure_timer(self, time_left: Optional[Callable]):
        self._time_left = time_left
        self._time_safety_margin = 0.0
        if time_left is None:
            return

        try:
            remaining = float(time_left())
        except Exception:
            self._time_left = None
            return

        if remaining > 100.0:
            self._time_safety_margin = max(15.0, remaining * 0.02)
        else:
            self._time_safety_margin = max(0.02, remaining * 0.02)

    def _very_low_time(self):
        if self._time_left is None:
            return False
        try:
            remaining = float(self._time_left())
        except Exception:
            return False
        return remaining <= self._time_safety_margin * 2.0

    def _check_timeout(self):
        if self._time_left is None:
            return
        try:
            remaining = float(self._time_left())
        except Exception:
            return
        if remaining <= self._time_safety_margin:
            raise SearchTimeout()

    def preview_move(self, board_state, candidate_move):
        next_board = board_state.get_copy(False)
        next_board.apply_move(candidate_move, check_ok=False)
        return next_board

    def evaluate_move(
        self,
        board_state,
        belief,
        candidate_move,
        depth,
        is_max_player,
    ):
        self._check_timeout()

        if candidate_move.move_type == enums.MoveType.SEARCH:
            return self.search_value(
                board_state,
                belief,
                candidate_move,
                depth,
                is_max_player,
            )

        preview_board = self.preview_move(board_state, candidate_move)
        score = self.action_adjustment(
            board_state,
            belief,
            candidate_move,
            is_max_player,
            preview_board=preview_board,
        )

        next_board, next_belief = self.simulate_turn(
            board_state, belief, candidate_move, preview_board=preview_board
        )
        return score + self.expectimax(
            next_board,
            next_belief,
            depth - 1,
            is_max_player=not is_max_player,
        )

    def expectimax(self, board_state, belief, depth, is_max_player):
        self._check_timeout()

        if depth <= 0 or board_state.is_game_over():
            return self.utility(board_state, belief)

        legal_moves = self.get_ordered_moves(
            board_state, belief, is_max_player=is_max_player, root=False
        )
        if not legal_moves:
            return self.utility(board_state, belief)

        if is_max_player:
            best_score = -math.inf
            for candidate in legal_moves:
                score = self.evaluate_move(
                    board_state,
                    belief,
                    candidate,
                    depth,
                    is_max_player=True,
                )
                best_score = max(best_score, score)
            return best_score

        best_score = math.inf
        for candidate in legal_moves:
            score = self.evaluate_move(
                board_state,
                belief,
                candidate,
                depth,
                is_max_player=False,
            )
            best_score = min(best_score, score)
        return best_score

    def utility(self, board_state, belief):
        if board_state.is_game_over():
            outcome = self.root_result(board_state)
            if outcome == 1:
                return 100000.0
            if outcome == -1:
                return -100000.0
            return 0.0

        root_worker, enemy_worker = self.root_workers(board_state)

        root_loc = root_worker.get_location()
        enemy_loc = enemy_worker.get_location()

        point_margin = 16.0 * (root_worker.get_points() - enemy_worker.get_points())

        my_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(board_state, root_loc)
        )
        enemy_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(board_state, enemy_loc)
        )

        my_lane = self.lane_profile_score_from_loc(board_state, root_loc)
        enemy_lane = self.lane_profile_score_from_loc(board_state, enemy_loc)

        my_area = self.area_profile_score_from_loc(board_state, root_loc)
        enemy_area = self.area_profile_score_from_loc(board_state, enemy_loc)

        my_frontier = self.frontier_pull_score(board_state, root_loc)
        enemy_frontier = self.frontier_pull_score(board_state, enemy_loc)

        conversion_margin = 5.2 * (my_now - enemy_now) + 2.6 * (my_lane - enemy_lane)
        area_margin = 0.10 * (my_area - enemy_area)
        frontier_margin = 0.45 * (my_frontier - enemy_frontier)

        side_term = 0.0
        prof_side, side_margin = self.profitable_side(board_state)
        if prof_side != 0:
            if self.side_of(root_loc) == prof_side:
                side_term += 0.8 * side_margin
            if self.side_of(enemy_loc) == prof_side:
                side_term -= 0.7 * side_margin

        threat_term = 0.0
        if enemy_now >= 10.0:
            threat_term -= 34.0
        elif enemy_now >= 6.0:
            threat_term -= 16.0
        elif enemy_now >= 4.0:
            threat_term -= 7.0

        if my_now >= 10.0:
            threat_term += 12.0
        elif my_now >= 6.0:
            threat_term += 5.0
        elif my_now >= 4.0:
            threat_term += 2.0

        rat_term = -0.35 * self.expected_distance(root_loc, belief)
        rat_term += 0.10 * self.expected_distance(enemy_loc, belief)

        turn_margin = 0.3 * (root_worker.turns_left - enemy_worker.turns_left)

        return (
            point_margin
            + conversion_margin
            + area_margin
            + frontier_margin
            + side_term
            + threat_term
            + rat_term
            + turn_margin
        )

    def simulate_turn(self, board_state, belief, candidate_move, preview_board=None):
        next_board = preview_board if preview_board is not None else self.preview_move(
            board_state, candidate_move
        )
        next_belief = self.advance_belief_distribution(belief)
        if not next_board.is_game_over():
            next_board.reverse_perspective()
        return next_board, next_belief

    def search_value(
        self,
        board_state,
        belief,
        candidate_move,
        depth,
        is_max_player,
    ):
        hit_prob = self.location_probability(candidate_move.search_loc, belief)

        hit_board, hit_belief = self.simulate_search_outcome(
            board_state, belief, candidate_move.search_loc, hit=True
        )
        miss_board, miss_belief = self.simulate_search_outcome(
            board_state, belief, candidate_move.search_loc, hit=False
        )

        next_is_max_player = not is_max_player

        hit_score = self.expectimax(
            hit_board,
            hit_belief,
            depth - 1,
            is_max_player=next_is_max_player,
        )
        miss_score = self.expectimax(
            miss_board,
            miss_belief,
            depth - 1,
            is_max_player=next_is_max_player,
        )
        return hit_prob * hit_score + (1.0 - hit_prob) * miss_score

    def simulate_search_outcome(self, board_state, belief, search_loc, hit):
        next_board = board_state.get_copy(False)
        next_board.apply_move(move.Move.search(search_loc), check_ok=False)

        if hit:
            next_board.player_worker.increment_points(RAT_BONUS)
            next_belief = list(self.search_prior)
        else:
            next_board.player_worker.decrement_points(RAT_PENALTY)
            next_belief = self.failed_search_belief(belief, search_loc, next_board)

        next_board.winner = None
        next_board.check_win()

        next_belief = self.advance_belief_distribution(next_belief)
        if not next_board.is_game_over():
            next_board.reverse_perspective()
        return next_board, next_belief

    def failed_search_belief(self, belief, search_loc, board_state):
        failed = list(belief)
        failed[self.pos_to_index(search_loc)] = 0.0
        return self.normalize(failed, board_state)

    def immediate_point_gain(self, board_state, candidate_move, preview_board=None):
        preview = preview_board if preview_board is not None else self.preview_move(
            board_state, candidate_move
        )
        return float(
            preview.player_worker.get_points() - board_state.player_worker.get_points()
        )

    def best_available_point_gain(self, board_state):
        best = 0.0
        for candidate in board_state.get_valid_moves(exclude_search=True):
            try:
                gain = self.immediate_point_gain(board_state, candidate)
            except Exception:
                continue
            if gain > best:
                best = gain
        return best

    def loop_penalty(self, next_loc, available_cash, point_gain):
        if point_gain > 0.0:
            return 0.0

        repeats = sum(1 for loc in self.recent_positions if loc == next_loc)
        if repeats <= 0:
            return 0.0

        recent = list(self.recent_positions)
        penalty = 2.5 * repeats

        if len(recent) >= 2 and next_loc == recent[-2]:
            penalty += 4.5
        if len(recent) >= 4 and next_loc in recent[-4:]:
            penalty += 2.0

        if available_cash >= 2.0:
            penalty += 1.5 * available_cash

        penalty += 1.2 * min(self.zero_gain_streak, 4)
        return penalty

    def action_adjustment(
        self,
        board_state,
        belief,
        candidate_move,
        actor_is_root,
        preview_board=None,
    ):
        sign = 1.0 if actor_is_root else -1.0

        if candidate_move.move_type == enums.MoveType.SEARCH:
            return sign * self.adjusted_search_expectation(
                board_state, candidate_move.search_loc, belief
            )

        preview = preview_board if preview_board is not None else self.preview_move(
            board_state, candidate_move
        )

        actor_loc = board_state.player_worker.get_location()
        opp_loc = board_state.opponent_worker.get_location()
        next_actor_loc = preview.player_worker.get_location()
        next_opp_loc = preview.opponent_worker.get_location()

        point_gain = self.immediate_point_gain(
            board_state, candidate_move, preview_board=preview
        )
        available_cash = self.best_available_point_gain(board_state)

        before_lane = self.lane_profile_score_from_loc(board_state, actor_loc)
        after_lane = self.lane_profile_score_from_loc(preview, next_actor_loc)

        opp_before_lane = self.lane_profile_score_from_loc(board_state, opp_loc)
        opp_after_lane = self.lane_profile_score_from_loc(preview, next_opp_loc)

        before_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(board_state, actor_loc)
        )
        after_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(preview, next_actor_loc)
        )
        opp_before_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(board_state, opp_loc)
        )
        opp_after_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(preview, next_opp_loc)
        )

        before_area = self.area_profile_score_from_loc(board_state, actor_loc)
        after_area = self.area_profile_score_from_loc(preview, next_actor_loc)
        opp_before_area = self.area_profile_score_from_loc(board_state, opp_loc)
        opp_after_area = self.area_profile_score_from_loc(preview, next_opp_loc)

        before_frontier = self.frontier_pull_score(board_state, actor_loc)
        after_frontier = self.frontier_pull_score(preview, next_actor_loc)
        opp_before_frontier = self.frontier_pull_score(board_state, opp_loc)
        opp_after_frontier = self.frontier_pull_score(preview, next_opp_loc)

        lane_delta = after_lane - before_lane
        now_delta = after_now - before_now
        area_delta = after_area - before_area
        frontier_delta = after_frontier - before_frontier

        deny_delta = (opp_before_lane - opp_after_lane) + 1.4 * (
            opp_before_now - opp_after_now
        )
        deny_area = opp_before_area - opp_after_area
        deny_frontier = opp_before_frontier - opp_after_frontier

        score = (
            5.0 * point_gain
            + 2.6 * lane_delta
            + 1.4 * now_delta
            + 0.08 * area_delta
            + 0.35 * frontier_delta
            + 1.35 * deny_delta
            + 0.08 * deny_area
            + 0.18 * deny_frontier
        )

        if point_gain >= 2.0:
            score += 6.0 + 2.0 * point_gain
        if point_gain >= 4.0:
            score += 6.0
        if point_gain >= available_cash >= 2.0:
            score += 4.0

        live_cash = before_now
        enemy_threat = opp_before_now

        if candidate_move.move_type == enums.MoveType.PRIME:
            if after_now > before_now:
                score += 2.5 * (after_now - before_now)
            if after_now >= 4.0 and after_now > before_now:
                score += 4.0
            if after_now >= 6.0 and after_now > before_now:
                score += 4.0
            if after_lane > before_lane and after_now >= 2.0:
                score += 2.0
            if after_frontier > before_frontier:
                score += 0.8 * (after_frontier - before_frontier)

            if live_cash >= 4.0 and point_gain <= 0.0 and enemy_threat < 6.0:
                score -= 18.0 + 3.0 * live_cash
            elif available_cash >= 4.0 and point_gain <= 0.0 and enemy_threat < 6.0:
                score -= 8.0 + 1.5 * available_cash

        elif candidate_move.move_type == enums.MoveType.CARPET:
            junk_penalty = 0.0

            if point_gain <= 0.0 and enemy_threat < 6.0:
                junk_penalty += 20.0

            if point_gain <= 2.0 and live_cash >= 4.0 and enemy_threat < 6.0:
                junk_penalty += 14.0

            if point_gain < live_cash and enemy_threat < 6.0:
                junk_penalty += 3.0 * (live_cash - point_gain)

            if after_lane < before_lane and point_gain < 4.0 and enemy_threat < 6.0:
                junk_penalty += 1.5 * (before_lane - after_lane)

            if (
                after_frontier + 1.5 < before_frontier
                and point_gain < 4.0
                and enemy_threat < 6.0
            ):
                junk_penalty += 0.8 * (before_frontier - after_frontier)

            if enemy_threat >= 10.0 and opp_after_now < enemy_threat:
                score += 10.0
            elif enemy_threat >= 6.0 and opp_after_now < enemy_threat:
                score += 5.0

            if point_gain >= 4.0:
                score += 12.0 + 2.0 * point_gain
            elif point_gain >= 2.0:
                score += 6.0 + 1.5 * point_gain

            if live_cash >= 4.0 and point_gain >= live_cash:
                score += 10.0

            score -= junk_penalty

        else:
            if after_now > before_now:
                score += 1.8 * (after_now - before_now)
            if after_lane > before_lane and after_now >= 2.0:
                score += 1.0
            if after_frontier > before_frontier:
                score += 1.0 * (after_frontier - before_frontier)

            if live_cash >= 4.0 and point_gain <= 0.0 and enemy_threat < 6.0:
                score -= 22.0 + 3.5 * live_cash
            elif available_cash >= 4.0 and point_gain <= 0.0:
                score -= (
                    12.0
                    + 2.5 * available_cash
                    + 2.0 * min(self.zero_gain_streak, 4)
                )
            elif available_cash >= 2.0 and point_gain <= 0.0:
                score -= (
                    6.0
                    + 1.5 * available_cash
                    + 1.5 * min(self.zero_gain_streak, 4)
                )

            if point_gain <= 0.0 and after_now < before_now and enemy_threat < 6.0:
                score -= 5.0 * (before_now - after_now)

        score -= self.loop_penalty(next_actor_loc, available_cash, point_gain)
        score -= self.side_abandonment_penalty(
            board_state,
            actor_loc,
            next_actor_loc,
            point_gain,
            before_frontier,
            after_frontier,
        )

        return sign * score

    def root_result(self, board_state):
        winner = board_state.get_winner()
        if winner not in (Result.PLAYER, Result.ENEMY, Result.TIE):
            return 0

        root_is_current = board_state.player_worker.is_player_a == self.root_is_player_a
        if winner == Result.TIE:
            return 0
        if root_is_current:
            return 1 if winner == Result.PLAYER else -1
        return 1 if winner == Result.ENEMY else -1

    def root_workers(self, board_state):
        if board_state.player_worker.is_player_a == self.root_is_player_a:
            return board_state.player_worker, board_state.opponent_worker
        return board_state.opponent_worker, board_state.player_worker

    def get_valid_moves(self, board_state, belief, is_max_player, root=False):
        board_moves = [
            candidate
            for candidate in board_state.get_valid_moves(exclude_search=True)
            if not (
                candidate.move_type == enums.MoveType.CARPET
                and candidate.roll_length == 1
            )
        ]
        if not board_moves:
            board_moves = list(board_state.get_valid_moves(exclude_search=True))

        moves = list(board_moves)

        if root:
            moves.extend(
                self.get_candidate_search_moves(
                    board_state, belief, board_moves, is_max_player
                )
            )

        return moves

    def get_ordered_moves(self, board_state, belief, is_max_player, root):
        moves = self.get_valid_moves(board_state, belief, is_max_player, root=root)
        if not moves:
            return []

        scored = [
            (
                self.move_order_score(board_state, belief, candidate, is_max_player),
                candidate,
            )
            for candidate in moves
        ]

        limit = self.root_width if root else self.node_width

        best = (
            heapq.nlargest(limit, scored, key=lambda item: item[0])
            if is_max_player
            else heapq.nsmallest(limit, scored, key=lambda item: item[0])
        )
        return [candidate for _, candidate in best]

    def move_order_score(self, board_state, belief, candidate_move, is_max_player):
        if candidate_move.move_type == enums.MoveType.SEARCH:
            return self.action_adjustment(
                board_state, belief, candidate_move, is_max_player
            )

        preview_board = self.preview_move(board_state, candidate_move)
        return self.action_adjustment(
            board_state,
            belief,
            candidate_move,
            is_max_player,
            preview_board=preview_board,
        )

    def get_candidate_search_moves(
        self, board_state, belief, board_moves, is_max_player
    ):
        if belief is None:
            return []

        turns_left = board_state.player_worker.turns_left
        point_diff = (
            board_state.player_worker.get_points()
            - board_state.opponent_worker.get_points()
        )
        my_now = max(
            0.0,
            self.best_immediate_carpet_points_from_loc(
                board_state, board_state.player_worker.get_location()
            ),
        )
        my_area = self.area_profile_score_from_loc(
            board_state, board_state.player_worker.get_location()
        )
        available_cash = self.best_available_point_gain(board_state)

        top_indices = sorted(
            range(len(belief)), key=lambda idx: belief[idx], reverse=True
        )[: self.max_search_candidates + 1]

        top_prob = belief[top_indices[0]] if top_indices else 0.0

        if turns_left >= 9:
            return []
        if my_now >= 4.0:
            return []
        if my_area >= 18.0:
            return []
        if available_cash >= 2.0:
            return []
        if top_prob < 0.50:
            return []
        if point_diff < -4 and top_prob < 0.62:
            return []

        search_moves = []
        for idx in top_indices:
            loc = self.index_to_pos(idx)
            expectation = self.adjusted_search_expectation(board_state, loc, belief)
            if expectation <= 0.0:
                continue
            search_moves.append(move.Move.search(loc))
            break

        return search_moves

    def lane_cash_value(self, run):
        if run < 2:
            return 0.0
        return float(CARPET_POINTS_TABLE.get(run, 0))

    def lane_runs_from_loc(self, board_state, loc):
        if not board_state.is_valid_cell(loc):
            return [0, 0, 0, 0]

        runs = []
        for direction in enums.Direction:
            current = loc
            run = 0
            while True:
                current = enums.loc_after_direction(current, direction)
                if not board_state.is_valid_cell(
                    current
                ) or not board_state.is_cell_carpetable(current):
                    break
                run += 1
            runs.append(run)

        runs.sort(reverse=True)
        return runs

    def lane_profile_score_from_loc(self, board_state, loc):
        runs = self.lane_runs_from_loc(board_state, loc)
        best_run = runs[0]
        second_run = runs[1]

        best_cash = self.lane_cash_value(best_run)
        second_cash = self.lane_cash_value(second_run)

        score = 1.9 * best_cash + 0.65 * second_cash

        if best_run >= 3:
            score += 4.0
        if best_run >= 4:
            score += 8.0
        if best_run >= 5:
            score += 14.0
        if best_run >= 6:
            score += 10.0

        if second_run >= 3:
            score += 2.5
        if second_run >= 4:
            score += 4.0

        return score

    def is_passable_for_area(self, board_state, loc):
        if not board_state.is_valid_cell(loc):
            return False
        return board_state.get_cell(loc) != enums.Cell.BLOCKED

    def passable_neighbors(self, board_state, loc):
        result = []
        for direction in enums.Direction:
            nxt = enums.loc_after_direction(loc, direction)
            if self.is_passable_for_area(board_state, nxt):
                result.append(nxt)
        return result

    def area_profile_score_from_loc(self, board_state, start_loc, max_depth=4):
        if not self.is_passable_for_area(board_state, start_loc):
            return -20.0

        queue = deque([(start_loc, 0)])
        seen = {start_loc}
        weighted_reachable = 0.0
        weighted_carpetable = 0.0
        weighted_branch = 0.0
        first_step_dirs = set()
        dead_ends = 0

        while queue:
            loc, depth = queue.popleft()
            neighbors = self.passable_neighbors(board_state, loc)
            weight = 1.0 / (1.0 + 0.55 * depth)

            if depth > 0:
                weighted_reachable += weight
                if board_state.is_cell_carpetable(loc):
                    weighted_carpetable += 1.35 * weight
                elif board_state.get_cell(loc) == enums.Cell.SPACE:
                    weighted_carpetable += 0.45 * weight

                weighted_branch += max(0, len(neighbors) - 1) * weight
                if len(neighbors) <= 1:
                    dead_ends += 1

            if depth == max_depth:
                continue

            for nxt in neighbors:
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, depth + 1))
                if depth == 0:
                    first_step_dirs.add(nxt)

        escape_dirs = len(first_step_dirs)
        local_degree = len(self.passable_neighbors(board_state, start_loc))

        score = (
            1.4 * weighted_reachable
            + 2.2 * weighted_carpetable
            + 1.15 * weighted_branch
            + 2.8 * escape_dirs
        )

        if local_degree <= 1:
            score -= 12.0
        elif local_degree == 2:
            score -= 3.0

        if dead_ends >= 5:
            score -= 0.9 * dead_ends

        if len(seen) <= 6:
            score -= 8.0
        elif len(seen) <= 10:
            score -= 3.0

        return score

    def side_of(self, loc):
        x, _ = loc
        mid = enums.BOARD_SIZE // 2
        if x < mid:
            return -1
        if x > mid:
            return 1
        return 0

    def local_open_degree(self, board_state, loc):
        return len(self.passable_neighbors(board_state, loc))

    def candidate_cells_for_frontier(self, board_state):
        cells = []
        my_loc = board_state.player_worker.get_location()
        opp_loc = board_state.opponent_worker.get_location()

        for y in range(enums.BOARD_SIZE):
            for x in range(enums.BOARD_SIZE):
                loc = (x, y)
                if not board_state.is_valid_cell(loc):
                    continue
                if board_state.get_cell(loc) == enums.Cell.BLOCKED:
                    continue
                if loc == my_loc or loc == opp_loc:
                    continue

                lane = self.lane_profile_score_from_loc(board_state, loc)
                now = max(
                    0.0, self.best_immediate_carpet_points_from_loc(board_state, loc)
                )
                open_deg = self.local_open_degree(board_state, loc)

                score = 1.9 * now + 1.25 * lane + 0.4 * open_deg
                if score > 0.0:
                    cells.append((score, loc))

        cells.sort(reverse=True, key=lambda item: item[0])
        return cells[:8]

    def frontier_pull_score(self, board_state, loc):
        frontier = self.candidate_cells_for_frontier(board_state)
        if not frontier:
            return 0.0

        total = 0.0
        for value, cell in frontier:
            d = self.manhattan(loc, cell)
            total += value / (1.0 + d)
        return total

    def frontier_side_balance(self, board_state):
        left = 0.0
        right = 0.0
        center = 0.0

        for value, loc in self.candidate_cells_for_frontier(board_state):
            side = self.side_of(loc)
            if side < 0:
                left += value
            elif side > 0:
                right += value
            else:
                center += value

        return left, center, right

    def profitable_side(self, board_state):
        left, center, right = self.frontier_side_balance(board_state)
        if left > right + 4.0:
            return -1, left - right
        if right > left + 4.0:
            return 1, right - left
        return 0, abs(right - left)

    def side_abandonment_penalty(
        self,
        board_state,
        current_loc,
        next_loc,
        point_gain,
        before_frontier_pull,
        after_frontier_pull,
    ):
        profitable_side, side_margin = self.profitable_side(board_state)
        if profitable_side == 0:
            return 0.0

        cur_side = self.side_of(current_loc)
        nxt_side = self.side_of(next_loc)

        penalty = 0.0

        if (
            cur_side == profitable_side
            and nxt_side != profitable_side
            and point_gain <= 0.0
        ):
            penalty += 7.0 + 0.9 * side_margin

        if after_frontier_pull + 2.0 < before_frontier_pull and point_gain <= 0.0:
            penalty += 1.2 * (before_frontier_pull - after_frontier_pull)

        return penalty

    def best_immediate_carpet_run_from_loc(self, board_state, loc):
        return self.lane_runs_from_loc(board_state, loc)[0]

    def best_immediate_carpet_points_from_loc(self, board_state, loc):
        run = self.best_immediate_carpet_run_from_loc(board_state, loc)
        return float(CARPET_POINTS_TABLE.get(run, 0))

    def search_expectation(self, loc, belief):
        probability = self.location_probability(loc, belief)
        return probability * RAT_BONUS - (1.0 - probability) * RAT_PENALTY

    def search_tempo_penalty(self, board_state, loc, belief):
        probability = self.location_probability(loc, belief)
        turns_left = board_state.player_worker.turns_left
        point_diff = (
            board_state.player_worker.get_points()
            - board_state.opponent_worker.get_points()
        )

        current_loc = board_state.player_worker.get_location()
        current_now = max(
            0.0, self.best_immediate_carpet_points_from_loc(board_state, current_loc)
        )
        current_lane = self.lane_profile_score_from_loc(board_state, current_loc)
        current_area = self.area_profile_score_from_loc(board_state, current_loc)
        available_cash = self.best_available_point_gain(board_state)

        penalty = 0.0

        if turns_left > 8:
            penalty += 2.5
        elif turns_left > 5:
            penalty += 1.0

        if probability < 0.55:
            penalty += 2.5
        if probability < 0.45:
            penalty += 3.5
        if point_diff < -4 and probability < 0.65:
            penalty += 3.0

        if current_now >= 4.0:
            penalty += 2.0
        if current_now >= 6.0:
            penalty += 3.0
        if current_lane >= 18.0:
            penalty += 2.0
        if current_area >= 18.0:
            penalty += 2.0
        if available_cash >= 2.0:
            penalty += 3.0
        if self.zero_gain_streak >= 2:
            penalty += 1.5

        return penalty

    def adjusted_search_expectation(self, board_state, loc, belief):
        return self.search_expectation(loc, belief) - self.search_tempo_penalty(
            board_state, loc, belief
        )

    def expected_distance(self, loc, belief):
        if belief is None:
            return 0.0

        total = 0.0
        for idx, probability in enumerate(belief):
            if probability <= 0:
                continue
            rat_loc = self.index_to_pos(idx)
            total += probability * self.manhattan(loc, rat_loc)
        return total

    def location_probability(self, loc, belief):
        if belief is None:
            return 0.0
        return belief[self.pos_to_index(loc)]

    def update_belief(self, board_state, sensor_data):
        prior = self.advance_belief(board_state)

        if board_state.opponent_search[0] is not None:
            search_loc, search_hit = board_state.opponent_search
            if search_hit:
                prior = [0.0] * (enums.BOARD_SIZE * enums.BOARD_SIZE)
                prior[self.pos_to_index(search_loc)] = 1.0
            else:
                prior[self.pos_to_index(search_loc)] = 0.0

        if sensor_data is None or len(sensor_data) != 2:
            self.belief = self.normalize(prior, board_state)
            return

        noise, sensed_distance = sensor_data
        worker_loc = board_state.player_worker.get_location()
        posterior = []

        for idx, prior_prob in enumerate(prior):
            loc = self.index_to_pos(idx)
            cell = board_state.get_cell(loc)
            noise_prob = NOISE_PROBS[cell][int(noise)]
            distance_prob = self.distance_likelihood(worker_loc, loc, sensed_distance)
            posterior.append(prior_prob * noise_prob * distance_prob)

        self.belief = self.normalize(posterior, board_state)

    def advance_belief(self, board_state):
        if self.belief is None:
            return self.uniform_belief(board_state)
        return self.advance_belief_distribution(self.belief)

    def advance_belief_distribution(self, belief):
        if belief is None:
            return list(self.search_prior)
        if self.transition_matrix is None:
            return list(belief)

        size = enums.BOARD_SIZE * enums.BOARD_SIZE
        advanced = [0.0] * size
        for src, src_prob in enumerate(belief):
            if src_prob <= 0:
                continue
            row = self.transition_matrix[src]
            for dst in range(size):
                advanced[dst] += src_prob * float(row[dst])

        return advanced

    def distance_likelihood(self, origin, target, sensed_distance):
        actual_distance = self.manhattan(origin, target)
        probability = 0.0
        for offset, weight in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS):
            measured = max(0, actual_distance + offset)
            if measured == sensed_distance:
                probability += weight
        return probability

    def uniform_belief(self, board_state):
        size = enums.BOARD_SIZE * enums.BOARD_SIZE
        belief = [0.0] * size
        open_cells = []

        occupied = {
            board_state.player_worker.get_location(),
            board_state.opponent_worker.get_location(),
        }

        for y in range(enums.BOARD_SIZE):
            for x in range(enums.BOARD_SIZE):
                loc = (x, y)
                if loc in occupied:
                    continue
                if board_state.get_cell(loc) == enums.Cell.BLOCKED:
                    continue
                open_cells.append(loc)

        if not open_cells:
            return belief

        weight = 1.0 / len(open_cells)
        for loc in open_cells:
            belief[self.pos_to_index(loc)] = weight
        return belief

    def compute_headstart_prior(self):
        size = enums.BOARD_SIZE * enums.BOARD_SIZE
        if self.transition_matrix is None:
            return [1.0 / size] * size

        belief = [0.0] * size
        belief[0] = 1.0
        for _ in range(64):
            belief = self.advance_belief_distribution(belief)

        total = sum(belief)
        if total <= 0:
            return [1.0 / size] * size
        return [prob / total for prob in belief]

    def normalize(self, probabilities, board_state):
        cleaned = list(probabilities)
        occupied = {
            board_state.player_worker.get_location(),
            board_state.opponent_worker.get_location(),
        }

        for loc in occupied:
            cleaned[self.pos_to_index(loc)] = 0.0

        for y in range(enums.BOARD_SIZE):
            for x in range(enums.BOARD_SIZE):
                loc = (x, y)
                if board_state.get_cell(loc) == enums.Cell.BLOCKED:
                    cleaned[self.pos_to_index(loc)] = 0.0

        total = sum(cleaned)
        if total <= 0:
            return self.uniform_belief(board_state)
        return [probability / total for probability in cleaned]

    def result_location(self, current_loc, candidate_move):
        if candidate_move.move_type == enums.MoveType.SEARCH:
            return current_loc

        next_loc = current_loc
        steps = 1
        if candidate_move.move_type == enums.MoveType.CARPET:
            steps = candidate_move.roll_length

        for _ in range(steps):
            next_loc = enums.loc_after_direction(next_loc, candidate_move.direction)

        if not (
            0 <= next_loc[0] < enums.BOARD_SIZE and 0 <= next_loc[1] < enums.BOARD_SIZE
        ):
            return None
        return next_loc

    def pos_to_index(self, loc):
        return loc[1] * enums.BOARD_SIZE + loc[0]

    def index_to_pos(self, idx):
        return (idx % enums.BOARD_SIZE, idx // enums.BOARD_SIZE)

    def manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])