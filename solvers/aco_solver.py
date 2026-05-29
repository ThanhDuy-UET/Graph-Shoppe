from __future__ import annotations

import random
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import (
    ALPHA,
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    is_valid_cell,
    move_cost,
    r_base,
    valid_next_pos,
)
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class ACOTarget:
    """Mục tiêu ngắn hạn được một kiến chọn cho một shipper."""

    __slots__ = ("kind", "pos", "score", "order_id", "key")

    def __init__(
        self,
        kind: str,
        pos: Position,
        score: float,
        order_id: Optional[int] = None,
    ) -> None:
        self.kind = kind
        self.pos = pos
        self.score = score
        self.order_id = order_id
        self.key = (kind, order_id if order_id is not None else pos)


class ACOSolver(Solver):
    """
    Ant Colony Optimization online cho MAPD.

    ACO ở đây không cố giải VRP toàn cục cho toàn bộ episode vì đơn hàng xuất hiện online.
    Mỗi timestep solver chạy một rolling-horizon nhỏ:
      1. Sinh candidate pickup/delivery/hotspot cho từng shipper.
      2. Nhiều kiến lấy mẫu một tổ hợp target, có ràng buộc một pickup chỉ một shipper.
      3. Chấm điểm tổ hợp theo reward/deadline/distance/conflict.
      4. Cập nhật pheromone trên các cạnh (shipper_id, target_key).
      5. Thực thi action đầu tiên hướng tới target tốt nhất.
    """

    method_name = "ACOSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.orders: List[Order] = []
        self._bfs_cache: Dict[Position, Tuple[Dict[Position, int], Dict[Position, Move]]] = {}
        self._pheromone: Dict[Tuple[int, Tuple[object, object]], float] = {}
        self._rng = random.Random(20240527 + env.N * 31 + env.C * 17 + env.G)
        self._avg_free_degree = self._compute_avg_free_degree()
        self._demand_pressure = env.G / max(env.T, 1)
        self._last_positions: Dict[int, Position] = {}
        self._stuck_ticks: Dict[int, int] = {}
        self._free_flow_steps = 0
        self._runtime_hardened = False
        self._runtime_explore = False
        self._runtime_high_pressure = False
        self._runtime_clustered = False
        self._runtime_stuck_agents = 0

        self.alpha = 1.1
        self.beta = 2.0
        self.evaporation = 0.12
        self.deposit_scale = 0.035
        self.max_candidates_per_shipper = 10
        self.n_ants = 18 if env.C <= 3 else 24
        if self._demand_pressure >= 0.15:
            self.n_ants += 8

    # ------------------------------------------------------------------
    # BFS helpers
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _compute_avg_free_degree(self) -> float:
        total_degree = 0
        free_count = 0
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell != 0:
                    continue
                free_count += 1
                total_degree += sum(1 for _ in self._neighbors((r, c)))
        return total_degree / max(free_count, 1)

    def _bfs_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        if start in self._bfs_cache:
            return self._bfs_cache[start]
        if not is_valid_cell(start, self.grid):
            self._bfs_cache[start] = ({}, {})
            return {}, {}

        dist: Dict[Position, int] = {start: 0}
        first_move: Dict[Position, Move] = {start: "S"}
        queue: deque[Position] = deque([start])
        while queue:
            current = queue.popleft()
            for move, nxt in self._neighbors(current):
                if nxt in dist:
                    continue
                dist[nxt] = dist[current] + 1
                first_move[nxt] = move if current == start else first_move[current]
                queue.append(nxt)

        self._bfs_cache[start] = (dist, first_move)
        return dist, first_move

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        dist, _ = self._bfs_from(start)
        return dist.get(goal, INF)

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        _, first_move = self._bfs_from(start)
        return first_move.get(goal, "S")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _move_cost_estimate(self, shipper: Shipper, orders: Dict[int, Order], distance: int) -> float:
        if distance >= INF:
            return -INF
        return distance * move_cost(self._carried_weight(shipper, orders), shipper.W_max)

    def _use_hardened_scoring(self) -> bool:
        return self._runtime_hardened

    def _use_elite_baseline(self) -> bool:
        if self._runtime_explore:
            return False
        return True

    def _update_runtime_mode(self, obs: dict) -> None:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        active_orders = [order for order in orders.values() if not order.picked and not order.delivered]
        active_ratio = len(active_orders) / max(len(shippers), 1)
        new_count = len(obs.get("new_order_ids", []))

        clustered_pairs = 0
        separated_pairs = 0
        sample = active_orders[:28]
        for i, first in enumerate(sample):
            for second in sample[i + 1 :]:
                manhattan = abs(first.sx - second.sx) + abs(first.sy - second.sy)
                if manhattan <= 3:
                    clustered_pairs += 1
                elif manhattan >= max(8, self.env.N // 2):
                    separated_pairs += 1

        self._runtime_clustered = clustered_pairs >= max(4, len(active_orders) // 2)
        runtime_surge = (
            active_ratio >= 4.0
            and self._runtime_clustered
            and new_count >= max(2, len(shippers) // 2)
        )
        self._runtime_high_pressure = self._demand_pressure >= 0.15 or runtime_surge

        stuck_agents = 0
        for shipper in shippers:
            previous = self._last_positions.get(shipper.id)
            if previous == shipper.position and shipper.bag:
                self._stuck_ticks[shipper.id] = self._stuck_ticks.get(shipper.id, 0) + 1
            else:
                self._stuck_ticks[shipper.id] = 0
            if self._stuck_ticks[shipper.id] >= 3:
                stuck_agents += 1
            self._last_positions[shipper.id] = shipper.position
        self._runtime_stuck_agents = stuck_agents

        maze_or_small = self._avg_free_degree < 2.35 or self.env.N <= 10
        self._runtime_hardened = self._runtime_high_pressure and maze_or_small

        multi_region = separated_pairs > clustered_pairs and len(active_orders) >= max(8, 2 * len(shippers))
        self._runtime_explore = (
            self._runtime_high_pressure
            and not self._runtime_hardened
            and self.env.N >= 18
            and self._avg_free_degree >= 2.8
            and (multi_region or self._runtime_clustered)
        )

    def _pickup_density_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        bonus = 0.0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            if abs(order.sx - other.sx) + abs(order.sy - other.sy) <= 3:
                bonus += 0.8 + 0.4 * other.p
        return min(10.0 if self._use_hardened_scoring() else 12.0, bonus)

    def _delivery_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int, T: int) -> float:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        if distance >= INF:
            return -INF

        eta = t + distance
        reward = delivery_reward(order, eta, T)
        slack = order.et - eta
        if self._use_hardened_scoring():
            urgency = (10.0 + 5.0 * order.p) / (slack + 1.0) if slack >= 0 else -2.0 * min(30, -slack) * order.p
            return (
                2.2 * reward
                + 8.0 * order.p
                + urgency
                + 1.5 * len(shipper.bag)
                + self._move_cost_estimate(shipper, orders, distance)
                - 0.06 * distance
            )

        urgency = (14.0 + 5.0 * order.p) / (slack + 1.0) if slack >= 0 else -2.2 * min(40, -slack) * order.p
        return (
            2.5 * reward
            + 10.0 * order.p
            + urgency
            + 4.0 * len(shipper.bag)
            + self._move_cost_estimate(shipper, orders, distance)
            - 0.08 * distance
        )

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int, T: int) -> float:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        d_pick = self._distance(shipper.position, pickup)
        d_drop = self._distance(pickup, dropoff)
        if d_pick >= INF or d_drop >= INF:
            return -INF

        eta = t + d_pick + d_drop
        reward = delivery_reward(order, eta, T)
        if reward <= 0.0:
            return -INF

        slack = order.et - eta
        if self._use_hardened_scoring():
            return self._hardened_pickup_score(
                shipper,
                order,
                orders,
                t,
                T,
                d_pick,
                d_drop,
                reward,
                slack,
            )

        urgency = (18.0 * order.p) / (slack + 1.0) if slack >= 0 else -1.8 * min(50, -slack) * order.p
        load_ratio = (self._carried_weight(shipper, orders) + order.w) / max(shipper.W_max, 1.0)

        detour_penalty = 0.0
        if shipper.bag:
            best_current = min(
                self._distance(shipper.position, (orders[oid].ex, orders[oid].ey))
                for oid in shipper.bag
                if oid in orders
            )
            best_after = min(
                [self._distance(pickup, (order.ex, order.ey))]
                + [
                    self._distance(pickup, (orders[oid].ex, orders[oid].ey))
                    for oid in shipper.bag
                    if oid in orders
                ]
            )
            detour_penalty = 1.4 * max(0, d_pick + best_after - best_current)

        density_bonus = self._pickup_density_bonus(order, orders)
        capacity_bonus = 1.8 * (shipper.K_max - len(shipper.bag))
        return (
            2.1 * reward
            + 10.0 * order.p
            + urgency
            + density_bonus
            + capacity_bonus
            + self._bundle_bonus(shipper, order, orders, t, T)
            + self._move_cost_estimate(shipper, orders, d_pick)
            - 0.06 * (d_pick + d_drop)
            - 2.0 * load_ratio
            - detour_penalty
        ) / (1.0 + 0.07 * d_pick)

    def _nearest_delivery_after_pickup(
        self,
        shipper: Shipper,
        candidate: Order,
        orders: Dict[int, Order],
        pickup_pos: Position,
    ) -> int:
        best = self._distance(pickup_pos, (candidate.ex, candidate.ey))
        for oid in shipper.bag:
            carried = orders.get(oid)
            if carried is None or carried.delivered:
                continue
            best = min(best, self._distance(pickup_pos, (carried.ex, carried.ey)))
        return best

    def _hardened_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        T: int,
        d_pick: int,
        d_drop: int,
        expected_reward: float,
        slack: int,
    ) -> float:
        priority_bonus = 9.0 * order.p
        if slack >= 0:
            urgency = (16.0 * order.p) / (slack + 1.0)
            on_time_bonus = 8.0 + 2.0 * order.p
        else:
            urgency = -1.6 * min(50.0, -slack) * order.p
            on_time_bonus = 0.0

        load_after = self._carried_weight(shipper, orders) + order.w
        load_ratio = load_after / max(shipper.W_max, 1.0)
        capacity_bonus = 1.5 * (shipper.K_max - len(shipper.bag))

        detour_penalty = 0.0
        pickup = (order.sx, order.sy)
        if shipper.bag:
            current_best_delivery = min(
                self._distance(shipper.position, (orders[oid].ex, orders[oid].ey))
                for oid in shipper.bag
                if oid in orders
            )
            after_pick_delivery = self._nearest_delivery_after_pickup(shipper, order, orders, pickup)
            detour = d_pick + after_pick_delivery - current_best_delivery
            tightest_slack = min(
                orders[oid].et - (t + d_pick + self._distance(pickup, (orders[oid].ex, orders[oid].ey)))
                for oid in shipper.bag
                if oid in orders
            )
            detour_penalty = max(0, detour) * (1.2 + 0.5 * max(0, order.p - 1))
            if tightest_slack < -5:
                detour_penalty += 25.0
            elif tightest_slack < 3:
                detour_penalty += 10.0

        travel = d_pick + d_drop
        movement_penalty = 0.05 * travel + 1.5 * load_ratio
        return (
            2.0 * expected_reward
            + priority_bonus
            + urgency
            + on_time_bonus
            + capacity_bonus
            + self._pickup_density_bonus(order, orders)
            + self._bundle_bonus(shipper, order, orders, t, T)
            + self._move_cost_estimate(shipper, orders, d_pick)
            - movement_penalty
            - detour_penalty
        ) / (1.0 + 0.08 * d_pick)

    def _bundle_bonus(self, shipper: Shipper, first: Order, orders: Dict[int, Order], t: int, T: int) -> float:
        if not self._runtime_high_pressure or self._avg_free_degree < 2.35 or self.env.N > 10 or len(shipper.bag) + 2 > shipper.K_max:
            return 0.0
        first_pick = (first.sx, first.sy)
        first_drop = (first.ex, first.ey)
        d_first = self._distance(shipper.position, first_pick)
        d_drop = self._distance(first_pick, first_drop)
        if d_first >= INF or d_drop >= INF:
            return 0.0

        baseline = delivery_reward(first, t + d_first + d_drop, T)
        carried_weight = self._carried_weight(shipper, orders)
        best = 0.0
        for second in orders.values():
            if second.id == first.id or second.picked or second.delivered:
                continue
            if carried_weight + first.w + second.w > shipper.W_max:
                continue
            second_pick = (second.sx, second.sy)
            gap = self._distance(first_pick, second_pick)
            if gap >= INF or gap > 6:
                continue
            second_drop = (second.ex, second.ey)
            d_s_d1 = self._distance(second_pick, first_drop)
            d_d1_d2 = self._distance(first_drop, second_drop)
            d_s_d2 = self._distance(second_pick, second_drop)
            d_d2_d1 = self._distance(second_drop, first_drop)

            route_value = -INF
            if d_s_d1 < INF and d_d1_d2 < INF:
                eta_first = t + d_first + gap + d_s_d1
                eta_second = eta_first + d_d1_d2
                route_value = max(route_value, delivery_reward(first, eta_first, T) + delivery_reward(second, eta_second, T))
            if d_s_d2 < INF and d_d2_d1 < INF:
                eta_second = t + d_first + gap + d_s_d2
                eta_first = eta_second + d_d2_d1
                route_value = max(route_value, delivery_reward(second, eta_second, T) + delivery_reward(first, eta_first, T))
            if route_value > baseline:
                coeff = 0.35 if self._use_hardened_scoring() else 0.30
                best = max(best, coeff * (route_value - baseline) + 2.0 * second.p)
        return min(24.0 if self._use_hardened_scoring() else 18.0, best)

    # ------------------------------------------------------------------
    # ACO construction
    # ------------------------------------------------------------------
    def _shipper_candidates(self, shipper: Shipper, orders: Dict[int, Order], t: int, T: int) -> List[ACOTarget]:
        candidates: List[ACOTarget] = []

        for oid in shipper.bag:
            order = orders.get(oid)
            if order is None or order.delivered:
                continue
            score = self._delivery_score(shipper, order, orders, t, T)
            if score > -INF:
                candidates.append(ACOTarget("deliver", (order.ex, order.ey), score, order.id))

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if not shipper.can_carry(order, orders):
                continue
            score = self._pickup_score(shipper, order, orders, t, T)
            if score > -INF:
                candidates.append(ACOTarget("pickup", (order.sx, order.sy), score, order.id))

        if not shipper.bag:
            hotspot = self._hotspot_candidate(shipper, orders)
            if hotspot is not None:
                candidates.append(hotspot)

        candidates.sort(key=lambda target: target.score, reverse=True)
        return candidates[: self.max_candidates_per_shipper]

    def _hotspot_candidate(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[ACOTarget]:
        best: Optional[ACOTarget] = None
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            pos = (order.sx, order.sy)
            distance = self._distance(shipper.position, pos)
            if distance >= INF:
                continue
            value = ALPHA[order.p] * r_base(order.w)
            if self._use_hardened_scoring():
                score = 0.6 * value + 2.0 * self._pickup_density_bonus(order, orders) - 0.4 * distance
            else:
                score = 0.55 * value + 2.2 * self._pickup_density_bonus(order, orders) - 0.35 * distance
            if best is None or score > best.score:
                best = ACOTarget("hotspot", pos, score)
        return best

    def _choose_target(
        self,
        shipper: Shipper,
        candidates: List[ACOTarget],
        reserved_pickups: Set[int],
    ) -> Optional[ACOTarget]:
        feasible = [
            target
            for target in candidates
            if target.kind != "pickup" or target.order_id not in reserved_pickups
        ]
        if not feasible:
            return None

        weights: List[float] = []
        for target in feasible:
            pheromone = self._pheromone.get((shipper.id, target.key), 1.0)
            heuristic = max(0.01, target.score + 40.0)
            weights.append((pheromone**self.alpha) * (heuristic**self.beta))

        total = sum(weights)
        if total <= 0:
            return max(feasible, key=lambda target: target.score)

        pick = self._rng.random() * total
        cumulative = 0.0
        for target, weight in zip(feasible, weights):
            cumulative += weight
            if cumulative >= pick:
                return target
        return feasible[-1]

    def _construct_ant_solution(
        self,
        shippers: List[Shipper],
        candidates_by_shipper: Dict[int, List[ACOTarget]],
    ) -> Dict[int, ACOTarget]:
        solution: Dict[int, ACOTarget] = {}
        reserved_pickups: Set[int] = set()
        order = list(shippers)
        self._rng.shuffle(order)

        # Shipper đang mang hàng được chọn trước để giải phóng capacity.
        order.sort(key=lambda shipper: (0 if shipper.bag else 1, shipper.id))
        for shipper in order:
            target = self._choose_target(shipper, candidates_by_shipper.get(shipper.id, []), reserved_pickups)
            if target is None:
                continue
            solution[shipper.id] = target
            if target.kind == "pickup" and target.order_id is not None:
                reserved_pickups.add(target.order_id)
        return solution

    def _construct_elite_solution(
        self,
        shippers: List[Shipper],
        candidates_by_shipper: Dict[int, List[ACOTarget]],
    ) -> Dict[int, ACOTarget]:
        """Kiến elite: chọn tham lam target score tốt nhất làm baseline ổn định."""
        all_candidates: List[Tuple[float, int, ACOTarget]] = []
        for shipper in shippers:
            for target in candidates_by_shipper.get(shipper.id, []):
                all_candidates.append((target.score, shipper.id, target))
        all_candidates.sort(key=lambda item: (-item[0], item[1]))

        solution: Dict[int, ACOTarget] = {}
        reserved_pickups: Set[int] = set()
        for _, sid, target in all_candidates:
            if sid in solution:
                continue
            if target.kind == "pickup" and target.order_id in reserved_pickups:
                continue
            solution[sid] = target
            if target.kind == "pickup" and target.order_id is not None:
                reserved_pickups.add(target.order_id)
        return solution

    def _solution_score(self, shippers: List[Shipper], solution: Dict[int, ACOTarget]) -> float:
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        score = sum(target.score for target in solution.values())

        desired: Dict[Position, int] = {}
        for sid, target in solution.items():
            shipper = shipper_by_id[sid]
            move = self._next_move(shipper.position, target.pos)
            nxt = valid_next_pos(shipper.position, move, self.grid)
            if nxt in desired:
                score -= 35.0
            desired[nxt] = sid

        return score

    def _aco_targets(self, obs: dict) -> Dict[int, ACOTarget]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t, T = int(obs["t"]), int(obs["T"])

        candidates_by_shipper = {
            shipper.id: self._shipper_candidates(shipper, orders, t, T)
            for shipper in shippers
        }

        if self._use_hardened_scoring():
            best_solution = self._construct_elite_solution(shippers, candidates_by_shipper)
            best_score = self._solution_score(shippers, best_solution)
            self._update_pheromone(best_solution, max(0.0, best_score))
            return best_solution

        if self._use_elite_baseline():
            best_solution = self._construct_elite_solution(shippers, candidates_by_shipper)
            best_score = self._solution_score(shippers, best_solution)
        else:
            best_solution = {}
            best_score = -INF
        for _ in range(self.n_ants):
            solution = self._construct_ant_solution(shippers, candidates_by_shipper)
            score = self._solution_score(shippers, solution)
            if score > best_score:
                best_score = score
                best_solution = solution

        self._update_pheromone(best_solution, max(0.0, best_score))
        return best_solution

    def _update_pheromone(self, solution: Dict[int, ACOTarget], score: float) -> None:
        for key in list(self._pheromone):
            self._pheromone[key] *= 1.0 - self.evaporation
            if self._pheromone[key] < 0.05:
                del self._pheromone[key]

        deposit = min(8.0, 1.0 + self.deposit_scale * score)
        for sid, target in solution.items():
            key = (sid, target.key)
            self._pheromone[key] = min(20.0, self._pheromone.get(key, 1.0) + deposit)

    # ------------------------------------------------------------------
    # Actions and movement
    # ------------------------------------------------------------------
    def _raw_action_for_target(self, shipper: Shipper, target: Optional[ACOTarget]) -> Action:
        if target is None:
            return "S", 0
        move = self._next_move(shipper.position, target.pos)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        if target.kind == "deliver" and nxt == target.pos:
            return move, 2
        if target.kind == "pickup" and nxt == target.pos:
            return move, 1
        return move, 0

    def _maybe_pick_or_deliver_here(
        self,
        shipper: Shipper,
        action: Action,
        orders: Dict[int, Order],
        t: int,
        T: int,
        target: Optional[ACOTarget],
    ) -> Action:
        move, op = action
        if op != 0:
            return action

        for oid in shipper.bag:
            order = orders.get(oid)
            if order is not None and (order.ex, order.ey) == shipper.position:
                return move, 2

        local_orders = [
            order
            for order in orders.values()
            if not order.picked
            and not order.delivered
            and (order.sx, order.sy) == shipper.position
            and shipper.can_carry(order, orders)
        ]
        if not local_orders:
            return action

        env_pick = min(local_orders, key=lambda order: (-order.p, order.et, order.id))
        if target is not None and target.kind == "pickup" and target.order_id == env_pick.id:
            return move, 1
        if not self._runtime_high_pressure or self._avg_free_degree < 2.35:
            return move, 1

        threshold = 18.0 + (8.0 if shipper.bag else 0.0)
        if self._pickup_score(shipper, env_pick, orders, t, T) >= threshold:
            return move, 1
        return action

    def _active_shipper_ids(self, shippers: List[Shipper], targets: Dict[int, ACOTarget]) -> Set[int]:
        active_limit = len(shippers)
        if self._runtime_hardened and self._avg_free_degree < 2.35:
            active_limit = min(active_limit, 2)
        elif self._runtime_hardened and self.env.N <= 10 and len(shippers) >= 3:
            active_limit = min(active_limit, 2)
        if active_limit >= len(shippers):
            return {shipper.id for shipper in shippers}

        ranked = []
        for shipper in shippers:
            target_score = targets[shipper.id].score if shipper.id in targets else -INF
            carried_bonus = 60.0 if shipper.bag else 0.0
            ranked.append((target_score + carried_bonus, shipper.id))
        ranked.sort(reverse=True)
        return {sid for _, sid in ranked[:active_limit]}

    def _use_free_flow(self) -> bool:
        if self._runtime_high_pressure:
            return True
        return self.env.N >= 20

    def _resolve_move_conflicts(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        targets: Dict[int, ACOTarget],
        actions: Dict[int, Action],
    ) -> Dict[int, Action]:
        old_pos = {shipper.id: shipper.position for shipper in shippers}
        desired = {
            shipper.id: valid_next_pos(shipper.position, actions.get(shipper.id, ("S", 0))[0], self.grid)
            for shipper in shippers
        }

        priority = {
            shipper.id: (targets[shipper.id].score if shipper.id in targets else -INF, -shipper.id)
            for shipper in shippers
        }

        blocked: Set[int] = set()
        by_target: Dict[Position, List[int]] = {}
        for sid, pos in desired.items():
            by_target.setdefault(pos, []).append(sid)
        for sids in by_target.values():
            if len(sids) <= 1:
                continue
            winner = max(sids, key=lambda sid: priority[sid])
            for sid in sids:
                if sid != winner:
                    blocked.add(sid)

        ids = [shipper.id for shipper in shippers]
        for i, sid in enumerate(ids):
            for other_id in ids[i + 1 :]:
                if desired[sid] == old_pos[other_id] and desired[other_id] == old_pos[sid]:
                    blocked.add(min((sid, other_id), key=lambda x: priority[x]))

        reserved = {desired[sid] for sid in ids if sid not in blocked}
        for shipper in shippers:
            sid = shipper.id
            move, op = actions.get(sid, ("S", 0))
            if sid in blocked:
                move = self._best_alternative_step(shipper, targets.get(sid), reserved, old_pos)[0]
                op = 0
            actions[sid] = self._maybe_pick_or_deliver_here(
                shipper,
                (move, op),
                orders,
                self.env.t,
                self.env.T,
                targets.get(sid),
            )
        return actions

    def _best_alternative_step(
        self,
        shipper: Shipper,
        target: Optional[ACOTarget],
        reserved: Set[Position],
        old_pos: Dict[int, Position],
    ) -> Tuple[Move, Position]:
        occupied = set(old_pos.values())
        current = shipper.position
        best_move = "S"
        best_pos = current
        best_dist = self._distance(current, target.pos) if target is not None else INF

        for move, nxt in self._neighbors(current):
            if nxt in reserved:
                continue
            if nxt in occupied and nxt != current:
                continue
            if target is None:
                return move, nxt
            dist = self._distance(nxt, target.pos)
            if dist < best_dist:
                best_dist = dist
                best_move = move
                best_pos = nxt
        return ("S", current) if current not in reserved else (best_move, best_pos)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        self._update_runtime_mode(obs)
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        targets = self._aco_targets(obs)
        active_ids = self._active_shipper_ids(shippers, targets)

        actions: Dict[int, Action] = {}
        for shipper in shippers:
            if shipper.id in active_ids:
                action = self._raw_action_for_target(shipper, targets.get(shipper.id))
            else:
                action = ("S", 0)
            actions[shipper.id] = self._maybe_pick_or_deliver_here(
                shipper,
                action,
                orders,
                int(obs["t"]),
                int(obs["T"]),
                targets.get(shipper.id),
            )

        if self._use_free_flow():
            return actions
        return self._resolve_move_conflicts(shippers, orders, targets, actions)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
