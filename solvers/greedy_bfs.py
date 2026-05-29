from __future__ import annotations

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


class Target:
    """Một mục tiêu ngắn hạn mà greedy policy chọn cho shipper."""

    __slots__ = ("kind", "pos", "score", "order_id")

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


class GreedyBFS(Solver):
    """
    Greedy BFS cho môi trường giao hàng online.

    Ý tưởng:
    - BFS có cache theo điểm xuất phát để lấy khoảng cách/next move trên grid có vật cản.
    - Mỗi bước sinh các ứng viên giao/nhặt cho mọi shipper.
    - Chấm điểm theo reward dự kiến, deadline, priority, chi phí đường đi và mật độ đơn gần pickup.
    - Gán mục tiêu tham lam toàn cục để một đơn không bị nhiều shipper cùng đuổi.
    - Sửa move một bước bằng reservation table để giảm va chạm/cản đường tại bottleneck.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.orders: List[Order] = []
        self._bfs_cache: Dict[Position, Tuple[Dict[Position, int], Dict[Position, Move]]] = {}
        self._last_positions: Dict[int, Position] = {}
        self._stuck_ticks: Dict[int, int] = {}
        self._free_flow_steps = 0
        self._avg_free_degree = self._compute_avg_free_degree()
        self._demand_pressure = self.env.G / max(self.env.T, 1)

    # ------------------------------------------------------------------
    # BFS utilities
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
        """Trả bảng dist và first_move từ start đến mọi ô reachable."""
        if start in self._bfs_cache:
            return self._bfs_cache[start]

        dist: Dict[Position, int] = {start: 0}
        first_move: Dict[Position, Move] = {start: "S"}
        queue: deque[Position] = deque([start])

        if not is_valid_cell(start, self.grid):
            self._bfs_cache[start] = ({}, {})
            return {}, {}

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
    # Reward/scoring helpers
    # ------------------------------------------------------------------
    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _estimated_move_cost(self, shipper: Shipper, orders: Dict[int, Order], distance: int) -> float:
        if distance >= INF:
            return -INF
        w_carried = self._carried_weight(shipper, orders)
        return distance * move_cost(w_carried, shipper.W_max)

    def _order_base_value(self, order: Order) -> float:
        # Dùng reward đúng hạn ở giữa deadline làm proxy ổn định cho đơn chưa biết thời điểm giao.
        return ALPHA[order.p] * r_base(order.w)

    def _delivery_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int, T: int) -> float:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        if distance >= INF:
            return -INF

        eta = t + distance
        reward = delivery_reward(order, eta, T)
        slack = order.et - eta
        urgency = 0.0
        if slack >= 0:
            urgency = (10.0 + 5.0 * order.p) / (slack + 1.0)
        else:
            urgency = -2.0 * min(30.0, -slack) * order.p

        # Giao hàng đang mang cần ưu tiên mạnh để giải phóng capacity, nhưng vẫn chọn
        # đơn trong bag theo reward/deadline thay vì chỉ theo khoảng cách.
        return (
            2.2 * reward
            + 8.0 * order.p
            + urgency
            + 1.5 * len(shipper.bag)
            + self._estimated_move_cost(shipper, orders, distance)
            - 0.06 * distance
        )

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

    def _pickup_density_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        """Ưu tiên nhẹ các cụm pickup gần nhau để phản ứng với surge/hotspot ẩn."""
        bonus = 0.0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            manhattan = abs(order.sx - other.sx) + abs(order.sy - other.sy)
            if manhattan <= 3:
                bonus += 0.8 + 0.4 * other.p
        return min(10.0, bonus)

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int, T: int) -> float:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        d_pick = self._distance(shipper.position, pickup)
        d_drop = self._distance(pickup, dropoff)
        if d_pick >= INF or d_drop >= INF:
            return -INF

        eta = t + d_pick + d_drop
        expected_reward = delivery_reward(order, eta, T)
        slack = order.et - eta
        if expected_reward <= 0.0:
            return -INF

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

        # Nếu đang mang hàng, chỉ nhận thêm khi pickup không phá route giao hiện tại quá nặng.
        detour_penalty = 0.0
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
        density_bonus = self._pickup_density_bonus(order, orders)
        bundle_bonus = self._bundle_lookahead_bonus(shipper, order, orders, t, T)

        return (
            2.0 * expected_reward
            + priority_bonus
            + urgency
            + on_time_bonus
            + capacity_bonus
            + density_bonus
            + bundle_bonus
            + self._estimated_move_cost(shipper, orders, d_pick)
            - movement_penalty
            - detour_penalty
        ) / (1.0 + 0.08 * d_pick)

    def _bundle_lookahead_bonus(
        self,
        shipper: Shipper,
        first: Order,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> float:
        """Ước lượng lợi ích gom thêm 1 đơn gần pickup đầu tiên trong surge/hotspot."""
        if self._demand_pressure < 0.15 or self._avg_free_degree < 2.35 or self.env.N > 10:
            return 0.0
        if len(shipper.bag) + 2 > shipper.K_max:
            return 0.0

        first_pick = (first.sx, first.sy)
        first_drop = (first.ex, first.ey)
        d_to_first = self._distance(shipper.position, first_pick)
        d_first_drop = self._distance(first_pick, first_drop)
        if d_to_first >= INF or d_first_drop >= INF:
            return 0.0

        carried_weight = self._carried_weight(shipper, orders)
        baseline_eta = t + d_to_first + d_first_drop
        baseline_reward = delivery_reward(first, baseline_eta, T)
        best_bonus = 0.0

        for second in orders.values():
            if second.id == first.id or second.picked or second.delivered:
                continue
            if carried_weight + first.w + second.w > shipper.W_max:
                continue

            second_pick = (second.sx, second.sy)
            second_drop = (second.ex, second.ey)
            pickup_gap = self._distance(first_pick, second_pick)
            if pickup_gap >= INF or pickup_gap > 6:
                continue

            d_second_first = self._distance(second_pick, first_drop)
            d_first_second = self._distance(first_drop, second_drop)
            d_second_drop = self._distance(second_pick, second_drop)
            d_second_to_first_drop = self._distance(second_drop, first_drop)
            if min(d_second_first, d_second_drop) >= INF:
                continue

            # Route A: P1 -> P2 -> D1 -> D2
            reward_a = -INF
            dist_a = INF
            if d_second_first < INF and d_first_second < INF:
                eta_first = t + d_to_first + pickup_gap + d_second_first
                eta_second = eta_first + d_first_second
                dist_a = d_to_first + pickup_gap + d_second_first + d_first_second
                reward_a = delivery_reward(first, eta_first, T) + delivery_reward(second, eta_second, T)

            # Route B: P1 -> P2 -> D2 -> D1
            reward_b = -INF
            dist_b = INF
            if d_second_drop < INF and d_second_to_first_drop < INF:
                eta_second = t + d_to_first + pickup_gap + d_second_drop
                eta_first = eta_second + d_second_to_first_drop
                dist_b = d_to_first + pickup_gap + d_second_drop + d_second_to_first_drop
                reward_b = delivery_reward(second, eta_second, T) + delivery_reward(first, eta_first, T)

            route_reward = max(reward_a, reward_b)
            route_dist = min(dist_a if reward_a >= reward_b else INF, dist_b if reward_b > reward_a else INF)
            if route_reward <= baseline_reward or route_dist >= INF:
                continue

            extra_reward = route_reward - baseline_reward
            extra_distance = max(0, route_dist - (d_to_first + d_first_drop))
            bonus = 0.35 * extra_reward - 0.12 * extra_distance + 2.0 * second.p
            best_bonus = max(best_bonus, bonus)

        return min(24.0, max(0.0, best_bonus))

    def _hotspot_target(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Target]:
        """Điều shipper rảnh đến cụm đơn đang chờ, nếu có cụm đủ đáng đi."""
        best: Optional[Target] = None
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            pos = (order.sx, order.sy)
            distance = self._distance(shipper.position, pos)
            if distance >= INF:
                continue
            density = self._pickup_density_bonus(order, orders)
            value = self._order_base_value(order)
            score = 0.6 * value + 2.0 * density - 0.4 * distance
            if best is None or score > best.score:
                best = Target("hotspot", pos, score)
        return best

    # ------------------------------------------------------------------
    # Target assignment
    # ------------------------------------------------------------------
    def _can_carry(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        return shipper.can_carry(order, orders)

    def _shipper_candidates(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> List[Target]:
        candidates: List[Target] = []

        for oid in shipper.bag:
            order = orders.get(oid)
            if order is None or order.delivered:
                continue
            score = self._delivery_score(shipper, order, orders, t, T)
            if score > -INF:
                candidates.append(Target("deliver", (order.ex, order.ey), score, order.id))

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if not self._can_carry(shipper, order, orders):
                continue
            score = self._pickup_score(shipper, order, orders, t, T)
            if score > -INF:
                candidates.append(Target("pickup", (order.sx, order.sy), score, order.id))

        if not shipper.bag:
            hotspot = self._hotspot_target(shipper, orders)
            if hotspot is not None:
                candidates.append(hotspot)

        candidates.sort(key=lambda target: target.score, reverse=True)
        return candidates[:12]

    def _assign_targets(self, obs: dict) -> Dict[int, Target]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t, T = int(obs["t"]), int(obs["T"])

        all_candidates: List[Tuple[float, int, Target]] = []
        for shipper in shippers:
            for target in self._shipper_candidates(shipper, orders, t, T):
                all_candidates.append((target.score, shipper.id, target))

        all_candidates.sort(key=lambda item: (-item[0], item[1]))

        assigned: Dict[int, Target] = {}
        reserved_pickups: Set[int] = set()

        for _, sid, target in all_candidates:
            if sid in assigned:
                continue
            if target.kind == "pickup" and target.order_id in reserved_pickups:
                continue
            assigned[sid] = target
            if target.kind == "pickup" and target.order_id is not None:
                reserved_pickups.add(target.order_id)

        return assigned

    # ------------------------------------------------------------------
    # Action construction and conflict reduction
    # ------------------------------------------------------------------
    def _raw_action_for_target(self, shipper: Shipper, target: Optional[Target]) -> Action:
        if target is None:
            return "S", 0

        move = self._next_move(shipper.position, target.pos)
        next_position = valid_next_pos(shipper.position, move, self.grid)

        if target.kind == "deliver" and next_position == target.pos:
            return move, 2
        if target.kind == "pickup" and next_position == target.pos:
            return move, 1
        return move, 0

    def _maybe_pick_or_deliver_here(
        self,
        shipper: Shipper,
        action: Action,
        orders: Dict[int, Order],
        t: int,
        T: int,
        target: Optional[Target] = None,
    ) -> Action:
        """Tận dụng bước đứng yên/đang bị chặn để thao tác hàng tại ô hiện tại."""
        move, op = action
        if op != 0:
            return action

        # Nếu đã ở điểm giao của bất kỳ đơn nào trong bag, giao ngay.
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

        if self._demand_pressure < 0.15 or self._avg_free_degree < 2.35:
            return move, 1

        # Không nhặt "miễn phí" nếu đơn tại chỗ làm đầy túi nhưng route dự kiến quá xấu.
        pickup_score = self._pickup_score(shipper, env_pick, orders, t, T)
        threshold = 18.0
        if shipper.bag:
            threshold += 8.0
        if pickup_score >= threshold:
            return move, 1

        return action

    def _resolve_move_conflicts(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        targets: Dict[int, Target],
        actions: Dict[int, Action],
    ) -> Dict[int, Action]:
        """Giảm va chạm rõ ràng mà không khóa cứng hành lang hẹp."""
        old_pos = {s.id: s.position for s in shippers}
        desired = {
            s.id: valid_next_pos(s.position, actions.get(s.id, ("S", 0))[0], self.grid)
            for s in shippers
        }

        priority = {
            s.id: (
                targets.get(s.id).score if s.id in targets else -INF,
                -s.id,
            )
            for s in shippers
        }

        blocked: Set[int] = set()

        by_target: Dict[Position, List[int]] = {}
        for sid, pos in desired.items():
            by_target.setdefault(pos, []).append(sid)

        for _, sids in by_target.items():
            if len(sids) <= 1:
                continue
            winner = max(sids, key=lambda sid: priority[sid])
            for sid in sids:
                if sid != winner:
                    blocked.add(sid)

        # Swap trực diện thường bị env chặn theo thứ tự id và dễ tạo kẹt qua lại.
        ids = [s.id for s in shippers]
        for i, sid in enumerate(ids):
            for other_id in ids[i + 1:]:
                if desired[sid] == old_pos[other_id] and desired[other_id] == old_pos[sid]:
                    loser = min((sid, other_id), key=lambda x: priority[x])
                    blocked.add(loser)

        reserved = {desired[sid] for sid in ids if sid not in blocked}

        for shipper in shippers:
            sid = shipper.id
            move, op = actions.get(sid, ("S", 0))
            if sid in blocked:
                move, _ = self._best_alternative_step(shipper, targets.get(sid), reserved, old_pos)
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
        target: Optional[Target],
        reserved: Set[Position],
        old_pos: Dict[int, Position],
    ) -> Tuple[Move, Position]:
        """Chọn bước phụ hợp lệ làm giảm khoảng cách tới target, hoặc đứng yên."""
        occupied = set(old_pos.values())
        current = shipper.position
        best_move = "S"
        best_pos = current
        best_dist = self._distance(current, target.pos) if target is not None else INF

        for move, nxt in self._neighbors(current):
            if nxt in reserved:
                continue
            # Tránh chủ động chen vào ô đang có shipper khác nếu không biết chắc họ sẽ rời đi.
            if nxt in occupied and nxt != current:
                continue
            if target is None:
                return move, nxt
            dist = self._distance(nxt, target.pos)
            if dist < best_dist:
                best_dist = dist
                best_move = move
                best_pos = nxt

        if current not in reserved:
            return "S", current
        return best_move, best_pos

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        targets = self._assign_targets(obs)

        active_ids = self._active_shipper_ids(shippers, targets, int(obs["N"]))
        actions: Dict[int, Action] = {}
        for shipper in shippers:
            if shipper.id in active_ids:
                actions[shipper.id] = self._raw_action_for_target(shipper, targets.get(shipper.id))
            else:
                actions[shipper.id] = ("S", 0)
            actions[shipper.id] = self._maybe_pick_or_deliver_here(
                shipper,
                actions[shipper.id],
                orders,
                int(obs["t"]),
                int(obs["T"]),
                targets.get(shipper.id),
            )

        if self._use_free_flow(int(obs["N"])):
            return actions

        if int(obs["N"]) >= 20:
            return actions

        if len(shippers) < 5:
            return self._resolve_move_conflicts(shippers, orders, targets, actions)

        moving_targets = 0
        stuck_agents = 0
        for shipper in shippers:
            sid = shipper.id
            wants_move = actions[sid][0] != "S"
            if wants_move:
                moving_targets += 1
            if wants_move and self._last_positions.get(sid) == shipper.position:
                self._stuck_ticks[sid] = self._stuck_ticks.get(sid, 0) + 1
            else:
                self._stuck_ticks[sid] = 0
            if self._stuck_ticks[sid] >= 3:
                stuck_agents += 1
            self._last_positions[sid] = shipper.position

        if stuck_agents >= max(2, len(shippers) // 2) and moving_targets >= stuck_agents:
            self._free_flow_steps = 18

        if self._free_flow_steps > 0:
            self._free_flow_steps -= 1
            return actions

        return self._resolve_move_conflicts(shippers, orders, targets, actions)

    def _active_shipper_ids(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Target],
        N: int,
    ) -> Set[int]:
        """Giới hạn số shipper chạy trong maze/hard config để giảm deadlock."""
        active_limit = len(shippers)
        high_pressure = self._demand_pressure >= 0.15

        if high_pressure and self._avg_free_degree < 2.35:
            active_limit = min(active_limit, 2)
        elif high_pressure and N <= 10 and len(shippers) >= 3:
            active_limit = min(active_limit, 2)

        if active_limit >= len(shippers):
            return {s.id for s in shippers}

        ranked: List[Tuple[float, int]] = []
        for shipper in shippers:
            target_score = targets[shipper.id].score if shipper.id in targets else -INF
            carried_bonus = 60.0 if shipper.bag else 0.0
            ranked.append((target_score + carried_bonus, shipper.id))

        ranked.sort(reverse=True)
        return {sid for _, sid in ranked[:active_limit]}

    def _use_free_flow(self, N: int) -> bool:
        """Áp lực cao/surge mạnh: để env tự xử lý ưu tiên id thường thoát kẹt tốt hơn."""
        if self._demand_pressure >= 0.15:
            return True
        return N >= 20

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
