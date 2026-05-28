from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class GreedyBFS(Solver):
    """
    Greedy BFS baseline cho Online MAPD.

    Solver chỉ cài phần policy:
    - chọn đơn cần giao/nhặt;
    - tìm đường bằng BFS trên grid hiện tại.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        """Liệt kê các ô kề hợp lệ bằng valid_next_pos() của env."""
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        """Chạy BFS và lưu parent để lấy khoảng cách/next move."""
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {
            start: (None, "S")
        }

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
        """
        Khoảng cách đường đi ngắn nhất trên grid có vật cản.
        """
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
        """Bước đi đầu tiên trên đường BFS từ start tới goal."""
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
    # Policy: chọn đơn
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
        """
        Chọn đơn đang mang để đi giao.
        """
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None

        return min(
            carried_orders,
            key=lambda order: (
                self._distance(shipper.position, (order.ex, order.ey)),
                order.et,
                -order.p,
                order.id,
            ),
        )

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
    ) -> Optional[Order]:
        """Chọn đơn chưa nhặt có pickup gần nhất và shipper còn khả năng chở."""
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if self._distance(shipper.position, (order.sx, order.sy)) >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: (
                self._distance(shipper.position, (order.sx, order.sy)),
                -order.p,
                order.et,
                order.id,
            ),
        )

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.
        """
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)

        # Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag
        # có đích tại ô hiện tại sau khi di chuyển.
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)

        # cargo_op = 1: env/Shipper.pickup_best() sẽ nhặt một đơn tốt nhất tại ô hiện tại.
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders)
            if delivery_order is not None:
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            pickup_order = self._select_pickup(shipper, orders, reserved_pickups)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            actions[shipper.id] = ("S", 0)

        return actions

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

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
