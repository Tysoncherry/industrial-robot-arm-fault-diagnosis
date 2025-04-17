import numpy as np
import random

class DCMotor:
    def __init__(self, nominal_speed=1.0):
        self.nominal_speed = nominal_speed  # m/s at full voltage
        self.stall_counter = 0

    def simulate(self, input_voltage, fault_type=None, t=0.0, sim_time=10.0):
        # 1) Base speed with small random jitter (±5%)
        base_speed = input_voltage * self.nominal_speed
        base_speed *= 1 + np.random.normal(0, 0.05)

        speed = base_speed

        # 2) Intermittent faults only during middle 20–80% of sim_time
        active = (sim_time*0.2 < t < sim_time*0.8)

        if fault_type == "voltage_drop" and active:
            speed *= 0.5 + np.random.normal(0, 0.05)
        elif fault_type == "torque_degradation" and active:
            speed -= (0.3 + np.random.normal(0, 0.05))
        elif fault_type == "partial_stall" and active:
            # 10% chance to stall on each step
            if random.random() < 0.10:
                speed = 0.0
        elif fault_type == "overspeed" and active:
            speed *= 1.5 + np.random.normal(0, 0.1)

        # 3) Never go below zero
        return max(0.0, speed)
