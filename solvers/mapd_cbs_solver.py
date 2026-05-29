from __future__ import annotations

import os
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
Task = Tuple[str, int, Position]

INF = 10**9
MOVES: Tuple[Move, ...] = ("S", "U", "D", "L", "R")


class MAPDCBSSolver(Solver):

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.strategy = os.getenv("MAPD_CBS_STRATEGY", "best")
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._seen_order_ids: Set[int] = set()
        self._source_scores: Dict[Position, float] = {}

    # ------------------------------------------------------------------
    # Static grid BFS
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        moves = MOVES if include_wait else MOVES[1:]
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if move == "S" or nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------
    def _remember_sources(self, orders: Dict[int, Order]) -> None:
        for order in orders.values():
            if order.id in self._seen_order_ids:
                continue
            self._seen_order_ids.add(order.id)
            source = (order.sx, order.sy)
            self._source_scores[source] = self._source_scores.get(source, 0.0) + 1.0 + 0.7 * order.p

    def _idle_target(self, shipper: Shipper, claimed_targets: Optional[Set[Position]] = None) -> Optional[Position]:
        claimed_targets = claimed_targets or set()
        if not self._source_scores:
            center = (self.env.N // 2, self.env.N // 2)
            return center if is_valid_cell(center, self.grid) and center not in claimed_targets else None

        best_target: Optional[Position] = None
        best_score = float("-inf")
        for pos, source_score in self._source_scores.items():
            if pos in claimed_targets:
                continue
            dist = self._distance(shipper.position, pos)
            if dist >= INF or dist == 0:
                continue
            distance_weight = 0.10 if self.env.N == 15 else 0.20
            score = source_score - distance_weight * dist
            if score > best_score:
                best_score = score
                best_target = pos
        return best_target

    def _bag_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _delivery_score(self, shipper: Shipper, order: Order, t: int) -> float:
        target = (order.ex, order.ey)
        dist = self._distance(shipper.position, target)
        if dist >= INF:
            return float(INF)

        eta = t + dist
        lateness = max(0, eta - order.et)
        slack = max(0, order.et - eta)
        reward = delivery_reward(order, eta, self.env.T)
        late_weight = 3.0
        reward_weight = 0.22
        slack_weight = 0.04 if self.env.N >= 18 else 0.015
        return dist + late_weight * lateness + slack_weight * slack - reward_weight * reward - 2.2 * order.p

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> float:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        to_pickup = self._distance(shipper.position, pickup)
        to_dropoff = self._distance(pickup, dropoff)
        if to_pickup >= INF or to_dropoff >= INF:
            return float(INF)

        eta_delivery = t + to_pickup + to_dropoff
        if eta_delivery >= self.env.T:
            return float(INF)
        lateness = max(0, eta_delivery - order.et)
        slack = max(0, order.et - eta_delivery)
        reward = delivery_reward(order, eta_delivery, self.env.T)
        load_ratio = (self._bag_weight(shipper, orders) + order.w) / max(shipper.W_max, 1.0)

        detour_penalty = 0.0
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if carried:
            next_delivery_cost = min(
                self._distance(pickup, (carried_order.ex, carried_order.ey))
                for carried_order in carried
            )
            next_delivery_late = min(
                max(
                    0,
                    t + to_pickup + self._distance(pickup, (carried_order.ex, carried_order.ey)) - carried_order.et,
                )
                for carried_order in carried
            )
            detour_penalty = 0.45 * next_delivery_cost + 4.0 * next_delivery_late

        late_weight = 2.6
        slack_weight = 0.05 if self.env.N >= 20 else (0.04 if self.env.N >= 18 else 0.012)
        reward_weight = 0.20

        return (
            to_pickup
            + 0.55 * to_dropoff
            + late_weight * lateness
            + slack_weight * slack
            + 1.5 * load_ratio
            + detour_penalty
            - reward_weight * reward
            - 2.0 * order.p
        )

    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried:
            return None

        return min(
            carried,
            key=lambda order: (self._delivery_score(shipper, order, t), order.et, -order.p, order.id),
        )

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_orders: Set[int],
        t: int,
    ) -> Optional[Order]:
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_orders:
                continue
            if not shipper.can_carry(order, orders):
                continue
            to_pickup = self._distance(shipper.position, (order.sx, order.sy))
            to_dropoff = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if to_pickup >= INF or to_dropoff >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: (
                self._pickup_score(shipper, order, orders, t),
                self._distance(shipper.position, (order.sx, order.sy)),
                -order.p,
                order.et,
                order.id,
            ),
        )

    def _free_pickup_at_current(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        assigned_orders: Set[int],
        t: int,
    ) -> Optional[Order]:
        candidates = [
            order
            for order in orders.values()
            if order.id not in assigned_orders
            and (order.sx, order.sy) == shipper.position
            and shipper.can_carry(order, orders)
            and self._pickup_score(shipper, order, orders, t) < INF
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda order: (-order.p, order.et, order.id))

    def _assign_tasks_sequential(self, obs: dict) -> Dict[int, Task]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        tasks: Dict[int, Task] = {}
        reserved_orders: Set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery = self._select_delivery(shipper, orders, t)
            pickup = self._select_pickup(shipper, orders, reserved_orders, t)

            if delivery is not None and pickup is not None:
                delivery_score = self._delivery_score(shipper, delivery, t)
                pickup_score = self._pickup_score(shipper, pickup, orders, t)
                urgent_delivery = t + self._distance(shipper.position, (delivery.ex, delivery.ey)) >= delivery.et

                if urgent_delivery or delivery_score <= pickup_score + 8.0:
                    tasks[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))
                else:
                    reserved_orders.add(pickup.id)
                    tasks[shipper.id] = ("pickup", pickup.id, (pickup.sx, pickup.sy))
                continue

            if delivery is not None:
                tasks[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))
                continue

            if pickup is not None:
                reserved_orders.add(pickup.id)
                tasks[shipper.id] = ("pickup", pickup.id, (pickup.sx, pickup.sy))

        return tasks

    def _assign_tasks_auction(self, obs: dict) -> Dict[int, Task]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        tasks: Dict[int, Task] = {}
        delivery_by_shipper: Dict[int, Order] = {}
        delivery_score_by_shipper: Dict[int, float] = {}
        assigned_orders: Set[int] = set()

        for shipper in shippers:
            delivery = self._select_delivery(shipper, orders, t)
            if delivery is None:
                continue
            delivery_by_shipper[shipper.id] = delivery
            delivery_score_by_shipper[shipper.id] = self._delivery_score(shipper, delivery, t)

            dist = self._distance(shipper.position, (delivery.ex, delivery.ey))
            if t + dist >= delivery.et:
                tasks[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))

        pickup_bids: List[Tuple[float, int, int, int, int, Shipper, Order]] = []
        for shipper in shippers:
            if shipper.id in tasks:
                continue

            delivery_score = delivery_score_by_shipper.get(shipper.id)
            for order in orders.values():
                if not shipper.can_carry(order, orders):
                    continue
                score = self._pickup_score(shipper, order, orders, t)
                if score >= INF:
                    continue
                if delivery_score is not None and delivery_score <= score + 8.0:
                    continue

                pickup_bids.append(
                    (
                        score,
                        self._distance(shipper.position, (order.sx, order.sy)),
                        -order.p,
                        order.et,
                        order.id,
                        shipper,
                        order,
                    )
                )

        pickup_bids.sort(key=lambda bid: bid[:5])
        for _, _, _, _, oid, shipper, order in pickup_bids:
            if shipper.id in tasks or oid in assigned_orders:
                continue
            tasks[shipper.id] = ("pickup", order.id, (order.sx, order.sy))
            assigned_orders.add(order.id)

        for shipper in shippers:
            if shipper.id in tasks:
                continue
            delivery = delivery_by_shipper.get(shipper.id)
            if delivery is not None:
                tasks[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))

        return tasks

    def _assign_tasks_empty_auction(self, obs: dict) -> Dict[int, Task]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        tasks: Dict[int, Task] = {}
        assigned_orders: Set[int] = set()

        for shipper in shippers:
            delivery = self._select_delivery(shipper, orders, t)
            if delivery is not None:
                delivery_dist = self._distance(shipper.position, (delivery.ex, delivery.ey))
                delivery_slack = delivery.et - (t + delivery_dist)
                free_pickup = (
                    self._free_pickup_at_current(shipper, orders, assigned_orders, t)
                    if self.env.N == 15 or self.env.N >= 20
                    else None
                )
                if free_pickup is not None and delivery_slack > 5:
                    tasks[shipper.id] = ("pickup", free_pickup.id, (free_pickup.sx, free_pickup.sy))
                    assigned_orders.add(free_pickup.id)
                    continue
                tasks[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))

        pickup_bids: List[Tuple[float, int, int, int, int, Shipper, Order]] = []
        for shipper in shippers:
            if shipper.id in tasks:
                continue
            for order in orders.values():
                if not shipper.can_carry(order, orders):
                    continue
                score = self._pickup_score(shipper, order, orders, t)
                if score >= INF:
                    continue
                pickup_bids.append(
                    (
                        score,
                        self._distance(shipper.position, (order.sx, order.sy)),
                        -order.p,
                        order.et,
                        order.id,
                        shipper,
                        order,
                    )
                )

        pickup_bids.sort(key=lambda bid: bid[:5])
        for _, _, _, _, oid, shipper, order in pickup_bids:
            if shipper.id in tasks or oid in assigned_orders:
                continue
            tasks[shipper.id] = ("pickup", order.id, (order.sx, order.sy))
            assigned_orders.add(order.id)

        return tasks

    def _assign_tasks(self, obs: dict) -> Dict[int, Task]:
        if self.env.N >= 15 or self.env.C <= 2:
            return self._assign_tasks_empty_auction(obs)
        return self._assign_tasks_sequential(obs)

    # ------------------------------------------------------------------
    # CBS-style time-expanded planning
    # ------------------------------------------------------------------
    def _find_timed_path(
        self,
        start: Position,
        goal: Position,
        blocked_vertices: Set[Tuple[int, Position]],
        blocked_edges: Set[Tuple[int, Position, Position]],
        horizon: int,
    ) -> Optional[List[Position]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        start_state = (0, start)
        queue: deque[Tuple[int, Position]] = deque([start_state])
        parent: Dict[Tuple[int, Position], Optional[Tuple[int, Position]]] = {start_state: None}

        goal_state: Optional[Tuple[int, Position]] = None
        while queue:
            current_t, current_pos = queue.popleft()
            if current_pos == goal:
                goal_state = (current_t, current_pos)
                break
            if current_t >= horizon:
                continue

            next_t = current_t + 1
            for _, nxt in self._neighbors(current_pos, include_wait=True):
                if (next_t, nxt) in blocked_vertices:
                    continue
                if (next_t, current_pos, nxt) in blocked_edges:
                    continue
                state = (next_t, nxt)
                if state in parent:
                    continue
                parent[state] = (current_t, current_pos)
                queue.append(state)

        if goal_state is None:
            return None

        rev_path: List[Position] = []
        state: Optional[Tuple[int, Position]] = goal_state
        while state is not None:
            rev_path.append(state[1])
            state = parent[state]
        return list(reversed(rev_path))

    def _reserve_path(
        self,
        path: List[Position],
        blocked_vertices: Set[Tuple[int, Position]],
        blocked_edges: Set[Tuple[int, Position, Position]],
        horizon: int,
    ) -> None:
        reserve_horizon = min(horizon, max(1, len(path) - 1)) if self.env.N == 18 else horizon
        for t in range(1, reserve_horizon + 1):
            prev = path[min(t - 1, len(path) - 1)]
            cur = path[min(t, len(path) - 1)]
            blocked_vertices.add((t, cur))
            blocked_edges.add((t, cur, prev))

    def _task_priority(self, shipper: Shipper, task: Task, orders: Dict[int, Order], t: int) -> Tuple[int, int, int, int, int]:
        kind, oid, target = task
        order = orders.get(oid)
        carrying_bonus = 0 if kind == "deliver" else 1
        deadline = order.et if order is not None else self.env.T
        priority = -order.p if order is not None else 0
        dist = self._distance(shipper.position, target)
        lateness = max(0, t + dist - deadline)
        return carrying_bonus, lateness, deadline, priority, shipper.id

    def _move_from_path(self, path: List[Position], start: Position, goal: Position) -> Move:
        if len(path) < 2:
            return "S"
        nxt = path[1]
        for move, pos in self._neighbors(start, include_wait=True):
            if pos == nxt:
                return move
        return self._next_move(start, goal)

    def _has_deliverable_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any(
            oid in orders
            and not orders[oid].delivered
            and (orders[oid].ex, orders[oid].ey) == pos
            for oid in shipper.bag
        )

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        tasks = self._assign_tasks(obs)
        self._remember_sources(orders)

        actions: Dict[int, Action] = {shipper.id: ("S", 0) for shipper in shippers}
        blocked_vertices: Set[Tuple[int, Position]] = set()
        blocked_edges: Set[Tuple[int, Position, Position]] = set()
        horizon = min(30, max(8, self.env.N * 2))

        allow_idle = self.env.N >= 20 or (self.env.N == 15 and t < self.env.T - 30)
        if allow_idle:
            claimed_idle_targets: Set[Position] = set()
            for shipper in shippers:
                if shipper.id in tasks:
                    continue
                target = self._idle_target(shipper, claimed_idle_targets)
                if target is not None:
                    tasks[shipper.id] = ("idle", -1, target)
                    claimed_idle_targets.add(target)

        active_shippers = [shipper for shipper in shippers if shipper.id in tasks]
        active_shippers.sort(key=lambda shipper: self._task_priority(shipper, tasks[shipper.id], orders, t))

        for shipper in active_shippers:
            kind, _, target = tasks[shipper.id]
            path = self._find_timed_path(
                shipper.position,
                target,
                blocked_vertices,
                blocked_edges,
                horizon,
            )

            if path is None:
                move = self._next_move(shipper.position, target)
                path = [shipper.position, valid_next_pos(shipper.position, move, self.grid)]
            else:
                move = self._move_from_path(path, shipper.position, target)

            next_pos = valid_next_pos(shipper.position, move, self.grid)
            if self._has_deliverable_at(shipper, orders, next_pos):
                actions[shipper.id] = (move, 2)
            elif next_pos != target:
                actions[shipper.id] = (move, 0)
            elif kind == "pickup":
                actions[shipper.id] = (move, 1)
            elif kind == "deliver":
                actions[shipper.id] = (move, 2)
            else:
                actions[shipper.id] = (move, 0)

            self._reserve_path(path, blocked_vertices, blocked_edges, horizon)

        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        self._seen_order_ids.clear()
        self._source_scores.clear()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        method = self.method_name if self.strategy == "best" else f"{self.method_name}-{self.strategy}"
        return self.env.result(method, elapsed_sec=time.time() - start_time)
