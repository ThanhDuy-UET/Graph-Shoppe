from __future__ import annotations

import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from env import DeliveryEnv, Order, Shipper
from solvers.greedy_bfs import GreedyBFS, INF


Position = Tuple[int, int]
Action = Tuple[str, object]


class Stop(NamedTuple):
    oid: int
    kind: str
    pos: Position


class Job(NamedTuple):
    oid: int
    stops: Tuple[Stop, ...]
    allowed: Tuple[int, ...]
    mandatory: bool
    penalty: float
    priority: int
    deadline: int


class VRPOrToolsSolver(GreedyBFS):
    """
    Self-coded online VRP heuristic inspired by OR-Tools Routing Solver.

    It does not import OR-Tools. The solver implements the same core ideas:
    vehicles, routes, pickup-delivery nodes, distance cost, capacity checks,
    disjunction-like penalties, cheapest insertion, and light local search.
    """

    method_name = "VRP-OrTools"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.max_candidate_jobs = 18
        self.local_search_rounds = 2
        self.distance_weight = 0.85
        self.late_weight = 7.0
        self.reward_weight = 0.75

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------
    def _base_reward(self, w: float) -> float:
        if w <= 0.2:
            return 4.0
        if w <= 3.0:
            return 10.0
        if w <= 10.0:
            return 15.0
        if w <= 30.0:
            return 20.0
        return 30.0

    def _estimate_reward(self, order: Order, t_delivery: int, horizon: int) -> float:
        alpha = {1: 1.0, 2: 2.0, 3: 3.0}
        beta = {1: 0.1, 2: 0.3, 3: 0.5}
        rb = self._base_reward(order.w)

        if t_delivery <= order.et:
            bonus = max(0.0, (order.et - t_delivery) / max(order.et, 1))
            return alpha[order.p] * rb * (1.0 + bonus)

        factor = max(0.0, 1.0 - (t_delivery - order.et) / max(horizon, 1))
        return beta[order.p] * rb * factor

    def _initial_load(self, shipper: Shipper, orders: Dict[int, Order]) -> Tuple[float, int, set[int]]:
        onboard = {oid for oid in shipper.bag if oid in orders}
        weight = sum(orders[oid].w for oid in onboard)
        return weight, len(onboard), onboard

    def _route_cost(
        self,
        shipper: Shipper,
        route: List[Stop],
        orders: Dict[int, Order],
        start_t: int,
        horizon: int,
    ) -> float:
        weight, count, onboard = self._initial_load(shipper, orders)
        cur = shipper.position
        t = start_t
        cost = 0.0

        for stop in route:
            order = orders.get(stop.oid)
            if order is None:
                return float("inf")

            dist = self._distance(cur, stop.pos)
            if dist >= INF:
                return float("inf")

            cost += self.distance_weight * dist * (1.0 + weight / max(shipper.W_max, 1.0))
            t += dist

            if stop.kind == "pickup":
                if stop.oid in onboard:
                    return float("inf")

                weight += order.w
                count += 1
                onboard.add(stop.oid)

                if count > shipper.K_max or weight > shipper.W_max:
                    return float("inf")

            elif stop.kind == "delivery":
                if stop.oid not in onboard:
                    return float("inf")

                reward = self._estimate_reward(order, t, horizon)
                lateness = max(0, t - order.et)
                cost += self.late_weight * lateness
                cost -= self.reward_weight * reward

                weight -= order.w
                count -= 1
                onboard.remove(stop.oid)

            else:
                return float("inf")

            cur = stop.pos

        return cost

    # ------------------------------------------------------------------
    # VRP model construction
    # ------------------------------------------------------------------
    def _make_jobs(self, obs: dict) -> List[Job]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        jobs: List[Job] = []

        for shipper in shippers:
            for oid in shipper.bag:
                order = orders.get(oid)
                if order is None or order.delivered:
                    continue

                jobs.append(
                    Job(
                        oid=order.id,
                        stops=(Stop(order.id, "delivery", (order.ex, order.ey)),),
                        allowed=(shipper.id,),
                        mandatory=True,
                        penalty=10**9,
                        priority=order.p,
                        deadline=order.et,
                    )
                )

        unpicked = [
            order
            for order in orders.values()
            if not order.picked and not order.delivered
        ]

        def candidate_key(order: Order) -> tuple:
            nearest_pickup = min(
                self._distance(shipper.position, (order.sx, order.sy))
                for shipper in shippers
            )
            return (-order.p, order.et, nearest_pickup, order.id)

        unpicked.sort(key=candidate_key)
        unpicked = unpicked[: self.max_candidate_jobs]

        for order in unpicked:
            allowed = tuple(
                shipper.id
                for shipper in shippers
                if shipper.W_max >= order.w and shipper.K_max > 0
            )
            if not allowed:
                continue

            reward_hint = self._base_reward(order.w) * order.p
            penalty = 80.0 + 40.0 * reward_hint + 250.0 * order.p

            jobs.append(
                Job(
                    oid=order.id,
                    stops=(
                        Stop(order.id, "pickup", (order.sx, order.sy)),
                        Stop(order.id, "delivery", (order.ex, order.ey)),
                    ),
                    allowed=allowed,
                    mandatory=False,
                    penalty=penalty,
                    priority=order.p,
                    deadline=order.et,
                )
            )

        jobs.sort(key=lambda job: (not job.mandatory, -job.priority, job.deadline, job.oid))
        return jobs

    def _best_insert(
        self,
        route: List[Stop],
        job: Job,
        shipper: Shipper,
        orders: Dict[int, Order],
        start_t: int,
        horizon: int,
    ) -> Tuple[float, Optional[List[Stop]]]:
        old_cost = self._route_cost(shipper, route, orders, start_t, horizon)
        best_delta = float("inf")
        best_route: Optional[List[Stop]] = None

        if len(job.stops) == 1:
            for i in range(len(route) + 1):
                candidate = route[:i] + list(job.stops) + route[i:]
                new_cost = self._route_cost(shipper, candidate, orders, start_t, horizon)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_route = candidate
            return best_delta, best_route

        pickup, delivery = job.stops
        for pickup_i in range(len(route) + 1):
            with_pickup = route[:pickup_i] + [pickup] + route[pickup_i:]
            for delivery_i in range(pickup_i + 1, len(with_pickup) + 1):
                candidate = with_pickup[:delivery_i] + [delivery] + with_pickup[delivery_i:]
                new_cost = self._route_cost(shipper, candidate, orders, start_t, horizon)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_route = candidate

        return best_delta, best_route

    def _construct_routes(self, obs: dict) -> Dict[int, List[Stop]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        routes: Dict[int, List[Stop]] = {shipper.id: [] for shipper in shippers}
        remaining_jobs = self._make_jobs(obs)

        while remaining_jobs:
            best_choice = None

            for job in remaining_jobs:
                insertions: List[Tuple[float, int, List[Stop]]] = []

                for sid in job.allowed:
                    shipper = shipper_by_id[sid]
                    delta, candidate = self._best_insert(
                        routes[sid],
                        job,
                        shipper,
                        orders,
                        obs["t"],
                        obs["T"],
                    )
                    if candidate is not None and delta < float("inf"):
                        insertions.append((delta, sid, candidate))

                if not insertions:
                    continue

                insertions.sort(key=lambda x: x[0])
                best_delta, best_sid, best_route = insertions[0]
                second_delta = insertions[1][0] if len(insertions) > 1 else job.penalty
                regret = second_delta - best_delta

                score = (
                    job.mandatory,
                    regret,
                    -best_delta,
                    job.priority,
                    -job.deadline,
                )

                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, job, best_delta, best_sid, best_route)

            if best_choice is None:
                break

            _, job, best_delta, best_sid, best_route = best_choice
            remaining_jobs.remove(job)

            if job.mandatory or best_delta <= job.penalty:
                routes[best_sid] = best_route

        return self._local_search(routes, obs)

    def _total_pair_cost(
        self,
        sid_a: int,
        route_a: List[Stop],
        sid_b: int,
        route_b: List[Stop],
        shipper_by_id: Dict[int, Shipper],
        orders: Dict[int, Order],
        obs: dict,
    ) -> float:
        return (
            self._route_cost(shipper_by_id[sid_a], route_a, orders, obs["t"], obs["T"])
            + self._route_cost(shipper_by_id[sid_b], route_b, orders, obs["t"], obs["T"])
        )

    def _try_relocate_between_routes(
        self,
        routes: Dict[int, List[Stop]],
        shipper_by_id: Dict[int, Shipper],
        orders: Dict[int, Order],
        obs: dict,
    ) -> bool:
        ids = sorted(routes)

        for sid_a in ids:
            for sid_b in ids:
                if sid_a == sid_b or not routes[sid_a]:
                    continue

                route_a = routes[sid_a]
                route_b = routes[sid_b]
                old_cost = self._total_pair_cost(sid_a, route_a, sid_b, route_b, shipper_by_id, orders, obs)

                for i, stop in enumerate(route_a):
                    reduced_a = route_a[:i] + route_a[i + 1:]

                    for j in range(len(route_b) + 1):
                        candidate_b = route_b[:j] + [stop] + route_b[j:]
                        new_cost = self._total_pair_cost(
                            sid_a,
                            reduced_a,
                            sid_b,
                            candidate_b,
                            shipper_by_id,
                            orders,
                            obs,
                        )

                        if new_cost + 1e-9 < old_cost:
                            routes[sid_a] = reduced_a
                            routes[sid_b] = candidate_b
                            return True

        return False

    def _try_two_opt_inside_route(
        self,
        routes: Dict[int, List[Stop]],
        shipper_by_id: Dict[int, Shipper],
        orders: Dict[int, Order],
        obs: dict,
    ) -> bool:
        for sid, route in routes.items():
            if len(route) < 4:
                continue

            old_cost = self._route_cost(shipper_by_id[sid], route, orders, obs["t"], obs["T"])

            for i in range(len(route) - 2):
                for j in range(i + 2, len(route)):
                    candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                    new_cost = self._route_cost(shipper_by_id[sid], candidate, orders, obs["t"], obs["T"])

                    if new_cost + 1e-9 < old_cost:
                        routes[sid] = candidate
                        return True

        return False

    def _local_search(self, routes: Dict[int, List[Stop]], obs: dict) -> Dict[int, List[Stop]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        shipper_by_id = {shipper.id: shipper for shipper in shippers}

        for _ in range(self.local_search_rounds):
            improved = False
            ids = sorted(routes)

            for sid_a_index, sid_a in enumerate(ids):
                for sid_b in ids[sid_a_index + 1:]:
                    route_a = routes[sid_a]
                    route_b = routes[sid_b]
                    if not route_a or not route_b:
                        continue

                    old_cost = (
                        self._route_cost(shipper_by_id[sid_a], route_a, orders, obs["t"], obs["T"])
                        + self._route_cost(shipper_by_id[sid_b], route_b, orders, obs["t"], obs["T"])
                    )

                    for i in range(len(route_a)):
                        for j in range(len(route_b)):
                            candidate_a = route_a[:]
                            candidate_b = route_b[:]
                            candidate_a[i], candidate_b[j] = candidate_b[j], candidate_a[i]

                            new_cost = (
                                self._route_cost(shipper_by_id[sid_a], candidate_a, orders, obs["t"], obs["T"])
                                + self._route_cost(shipper_by_id[sid_b], candidate_b, orders, obs["t"], obs["T"])
                            )

                            if new_cost + 1e-9 < old_cost:
                                routes[sid_a] = candidate_a
                                routes[sid_b] = candidate_b
                                improved = True
                                break

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

            if not improved:
                break

        for _ in range(self.local_search_rounds):
            improved = (
                self._try_relocate_between_routes(routes, shipper_by_id, orders, obs)
                or self._try_two_opt_inside_route(routes, shipper_by_id, orders, obs)
            )
            if not improved:
                break

        return routes

    # ------------------------------------------------------------------
    # Online action generation
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        routes = self._construct_routes(obs)
        actions: Dict[int, Action] = {}

        for shipper in sorted(shippers, key=lambda s: s.id):
            if any(
                oid in orders and shipper.can_deliver(orders[oid])
                for oid in shipper.bag
            ):
                actions[shipper.id] = ("S", 2)
                continue

            route = routes.get(shipper.id, [])
            if not route:
                actions[shipper.id] = ("S", 0)
                continue

            target = route[0]
            move, next_pos = self._move_towards(shipper, target.pos)

            if next_pos == target.pos:
                actions[shipper.id] = (move, 1 if target.kind == "pickup" else 2)
            else:
                actions[shipper.id] = (move, 0)

        return actions

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
