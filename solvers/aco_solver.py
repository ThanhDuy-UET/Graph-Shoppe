from __future__ import annotations
from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class ACOSolver(Solver):
    """Sinh viên cài đặt Ant Colony Optimization tại đây."""

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

    def run(self) -> dict:
        # TODO: xây dựng pheromone/heuristic trên đồ thị, mô phỏng và trả về dict kết quả.
        return default_result("ACO", self.env.config_name, self.env.G, self.orders)
