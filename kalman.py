import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=0.1, process_noise=1e-2, measurement_noise=1.0):
        self.dt = dt

        
        self.x = np.zeros((4, 1))

       
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])


        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])


        q = process_noise
        dt4 = (dt ** 4) / 4
        dt3 = (dt ** 3) / 2
        dt2 = dt ** 2

        self.Q = q * np.array([
            [dt4, 0,   dt3, 0],
            [0,   dt4, 0,   dt3],
            [dt3, 0,   dt2, 0],
            [0,   dt3, 0,   dt2]
        ])

    
        r = measurement_noise
        self.R = r * np.eye(2)


        self.P = np.eye(4) * 500 

    
        self.I = np.eye(4)

    def predict(self):
  
        self.x = np.dot(self.F, self.x)
 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x.copy()

    def update(self, z):
        """
        z: 2D measurement [x, y]
        """
        z = np.reshape(z, (2, 1)) 

        y = z - np.dot(self.H, self.x)  
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)

        return self.x.copy()

    def get_current_state(self):
        return self.x.flatten()

    def reset(self, initial_state):
        self.x = np.reshape(initial_state, (4, 1))
        self.P = np.eye(4) * 500



if __name__ == "__main__":
    from simulation import TargetSimulator
    import matplotlib.pyplot as plt

    sim = TargetSimulator(noise_std=2.5)
    kf = KalmanFilter2D(dt=0.1)

    est_path = []
    true_path = []
    meas_path = []

    for _ in range(100):
        data = sim.step()
        measurement = data["measured"]
        true = data["true"]

        kf.predict()
        kf.update(measurement)

        est = kf.get_current_state()

        est_path.append(est[:2])
        true_path.append(true)
        meas_path.append(measurement)

    est_path = np.array(est_path)
    true_path = np.array(true_path)
    meas_path = np.array(meas_path)

   
    plt.figure(figsize=(9, 6))
    plt.plot(true_path[:, 0], true_path[:, 1], label="True Path", color="green")
    plt.plot(meas_path[:, 0], meas_path[:, 1], label="Measured", color="red", linestyle="--", alpha=0.6)
    plt.plot(est_path[:, 0], est_path[:, 1], label="Kalman Estimate", color="blue", linewidth=2)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Kalman Filter 2D Tracking")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
