import os, csv, random, numpy as np
from simulation.robot_model import MobileRobot

FAULTS = [
    None, "voltage_drop", "torque_degradation", "partial_stall", "overspeed",
    "encoder_drift", "encoder_skip", "gyro_bias", "gyro_noise",
    "accelerometer_offset", "accelerometer_vibration"
]

def run_simulation(sim_time=20.0, fault_type=None, fault_mix=False):
    dt = 0.02
    steps = int(sim_time / dt)
    robot = MobileRobot(dt=dt)
    logfile = []

    # random initial heading
    robot.theta = random.uniform(0, 2*np.pi)

    for i in range(steps):
        t = i * dt

        # random voltages around 6V ±10%
        uL = 6.0 * (1 + np.random.uniform(-0.1, 0.1))
        uR = 6.0 * (1 + np.random.uniform(-0.1, 0.1))

        # mixed faults: randomly pick one at halfway
        if fault_mix and i == steps//2:
            fault_type = random.choice(FAULTS[1:])

        data = robot.step(uL, uR, fault_type, t, sim_time)
        data["fault"] = fault_type or "normal"
        logfile.append(data)

    # save CSV
    fname = fault_type or "normal"
    out_dir = "dataset/raw"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logfile[0].keys())
        writer.writeheader()
        writer.writerows(logfile)

    print(f"Saved: {path}")

if __name__ == "__main__":
    # run all single faults
    for ft in FAULTS:
        if ft:
            run_simulation(sim_time=20.0, fault_type=ft, fault_mix=False)
    # run a few mixed-fault scenarios
    for _ in range(3):
        run_simulation(sim_time=20.0, fault_type=None, fault_mix=True)
