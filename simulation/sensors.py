import numpy as np
import random
from collections import deque

class Encoder:
    def __init__(self):
        self.position = 0.0

    def simulate(self, velocity, dt, fault_type=None, t=0.0, sim_time=10.0):
        # nominal delta + normal jitter (±2%)
        delta = velocity * dt * (1 + np.random.normal(0, 0.02))

        # intermittent window
        active = (sim_time*0.2 < t < sim_time*0.8)

        if fault_type == "encoder_drift" and active:
            delta += 0.01 + np.random.normal(0, 0.005)
        elif fault_type == "encoder_skip" and active:
            if random.random() < 0.15:
                delta = 0.0

        self.position += delta
        return self.position

class IMU:
    def __init__(self):
        self.gyro_bias = 0.0

    def simulate(self, angular_velocity, fault_type=None, t=0.0, sim_time=10.0):
        # normal gyro noise (σ=0.05)
        omega = angular_velocity + np.random.normal(0, 0.05)
        accel = np.random.normal(0, 0.1)  # base accel noise

        active = (sim_time*0.2 < t < sim_time*0.8)

        if fault_type == "gyro_bias" and active:
            self.gyro_bias += 0.0005  # slow drift
            omega += self.gyro_bias
        elif fault_type == "gyro_noise" and active:
            omega += np.random.normal(0, 0.2)
        if fault_type == "accelerometer_offset" and active:
            accel += 0.2
        elif fault_type == "accelerometer_vibration" and active:
            accel += np.random.normal(0, 0.5)

        return {"angular_velocity": omega, "linear_acceleration": accel}
