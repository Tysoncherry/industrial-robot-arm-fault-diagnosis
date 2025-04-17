# simulation/robot_model.py

import numpy as np
from simulation.actuators import DCMotor
from simulation.sensors import Encoder, IMU

class MobileRobot:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.time = 0.0
        
        # Two independent wheel motors
        self.left_motor  = DCMotor(nominal_speed=1.0)
        self.right_motor = DCMotor(nominal_speed=1.0)
        
        # Two encoders and one IMU
        self.encoder_L = Encoder()
        self.encoder_R = Encoder()
        self.imu       = IMU()
        
        # Pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def step(self, uL, uR, fault_type, t, sim_time):
        # 1) Actuator outputs
        vL = self.left_motor.simulate(uL, fault_type, t, sim_time)
        vR = self.right_motor.simulate(uR, fault_type, t, sim_time)

        # 2) Differential-drive kinematics
        v     = (vL + vR) / 2.0
        omega = (vR - vL) / 0.3  # wheel separation = 0.3 m

        # 3) Update pose
        self.theta += omega * self.dt
        self.x     += v * np.cos(self.theta) * self.dt
        self.y     += v * np.sin(self.theta) * self.dt

        # 4) Sensor readouts
        encL = self.encoder_L.simulate(vL, self.dt, fault_type, t, sim_time)
        encR = self.encoder_R.simulate(vR, self.dt, fault_type, t, sim_time)
        imu  = self.imu.simulate(omega, fault_type, t, sim_time)

        # 5) Time update
        self.time = t + self.dt

        # 6) Return dictionary
        return {
            "time": self.time,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "encoder_l": encL,
            "encoder_r": encR,
            "angular_velocity": imu["angular_velocity"],
            "linear_acceleration": imu["linear_acceleration"]
        }
