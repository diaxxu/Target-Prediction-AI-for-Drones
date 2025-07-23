import numpy as np
import matplotlib.pyplot as plt


class TargetSimulator:
    def __init__(self, dt=0.1, noise_std=0.8, random_accel=True):
        self.dt = dt 
        self.noise_std = noise_std
        self.time = 0.0

      
        self.state = np.array([0.0, 0.0, 4.0, 1.0])  
        self.random_accel = random_accel

    def step(self):
        
        if self.random_accel:
            ax = np.random.uniform(-1, 1)
            ay = np.random.uniform(-1, 1)
            self.state[2] += ax * self.dt
            self.state[3] += ay * self.dt

        
        self.state[0] += self.state[2] * self.dt
        self.state[1] += self.state[3] * self.dt
        self.time += self.dt

        
        measured_x = self.state[0] + np.random.normal(0, self.noise_std)
        measured_y = self.state[1] + np.random.normal(0, self.noise_std)

        return {
            "time": self.time,
            "true": np.array([self.state[0], self.state[1]]),
            "measured": np.array([measured_x, measured_y]),
            "velocity": np.array([self.state[2], self.state[3]])
        }

    def set_velocity(self, vx, vy):
        self.state[2] = vx
        self.state[3] = vy

    def get_true_position(self):
        return self.state[:2]

    def get_velocity(self):
        return self.state[2:]


def run_batch_simulation(steps=150, plot=True):
    sim = TargetSimulator(dt=0.1, noise_std=1.2, random_accel=True)

    true_path = []
    measured_path = []

    for _ in range(steps):
        result = sim.step()
        true_path.append(result["true"])
        measured_path.append(result["measured"])

    true_path = np.array(true_path)
    measured_path = np.array(measured_path)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(true_path[:, 0], true_path[:, 1], label="True Path", color="blue")
        plt.plot(measured_path[:, 0], measured_path[:, 1], label="Measured Path", color="orange", linestyle="--")
        plt.title("Target Motion Simulation")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_batch_simulation()
