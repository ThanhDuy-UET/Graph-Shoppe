from __future__ import annotations
from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class MAPDCBSSolver(Solver):
    """Sinh viên cài đặt MAPD với Conflict-Based Search tại đây."""

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

    def run(self) -> dict:
        # TODO: sinh task, chạy CBS, mô phỏng và trả về dict kết quả.
        return default_result("MAPD-CBS", self.env.config_name, self.env.G, self.orders)
