import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulation import TargetSimulator
from kalman import KalmanFilter2D
import numpy as np


class TrackerGUI:
    def __init__(self, steps=300, dt=0.1):
        self.sim = TargetSimulator(dt=dt, noise_std=1.5)
        self.kf = KalmanFilter2D(dt=dt)
        self.dt = dt
        self.max_steps = steps

        self.true_path = []
        self.measured_path = []
        self.estimated_path = []

        self.running = True
        self.step_index = 0

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.true_line, = self.ax.plot([], [], label="True Path", color="green")
        self.measured_line, = self.ax.plot([], [], label="Measured", color="red", linestyle="--", alpha=0.5)
        self.estimated_line, = self.ax.plot([], [], label="Kalman Estimate", color="blue")
        self.status_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10,
                                        verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

        self.ax.set_title("Real-Time Target Tracking")
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True)
        self.ax.axis("equal")
        self.ax.legend()

    def toggle_run(self, event):
        self.running = not self.running

    def update(self, frame):
        if not self.running or self.step_index >= self.max_steps:
            return self.true_line, self.measured_line, self.estimated_line, self.status_text

        # Simulate target
        data = self.sim.step()
        measurement = data["measured"]
        true = data["true"]

        # Kalman Filter step
        self.kf.predict()
        self.kf.update(measurement)
        estimate = self.kf.get_current_state()

        # Store paths
        self.true_path.append(true)
        self.measured_path.append(measurement)
        self.estimated_path.append(estimate[:2])
        self.step_index += 1

        # Convert to numpy arrays for plotting
        tp = np.array(self.true_path)
        mp = np.array(self.measured_path)
        ep = np.array(self.estimated_path)

        self.true_line.set_data(tp[:, 0], tp[:, 1])
        self.measured_line.set_data(mp[:, 0], mp[:, 1])
        self.estimated_line.set_data(ep[:, 0], ep[:, 1])

        # Status message
        error = np.linalg.norm(estimate[:2] - true)
        lock_status = "LOCKED ✅" if error < 3.0 else "Tracking..."
        self.status_text.set_text(f"Step: {self.step_index}\nError: {error:.2f} m\nStatus: {lock_status}")

        # Autoscale
        self.ax.relim()
        self.ax.autoscale_view()

        return self.true_line, self.measured_line, self.estimated_line, self.status_text

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=self.dt * 1000, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.key_handler)
        plt.tight_layout()
        plt.show()

    def key_handler(self, event):
        if event.key == ' ':
            self.toggle_run(event)
        elif event.key == 'escape':
            print("Exiting...")
            plt.close()


if __name__ == "__main__":
    gui = TrackerGUI()
    gui.run()
