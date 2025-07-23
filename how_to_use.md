#  Target Tracking System — How To Use

This project simulates a moving target with noisy measurements, and uses a Kalman Filter to estimate the true position in real time. It also includes a GUI to visualize everything live.

---

##  Files

- `simulation.py`: Simulates target movement + noise
- `kalman.py`: The Kalman filter logic (2D position & velocity)
- `gui.py`: The live GUI using `matplotlib` animation
- `how_to_use.md`: This guide

---

##  Requirements

- Python 3.8 or higher
- Install dependencies:

```bash
pip install numpy matplotlib

```
---
## Run the Simulation

```bash
python gui.py
```
## What You’ll See
- Green line: True target path (no noise)

- Red dashed line: Noisy sensor (like GPS)

- Blue line: Kalman Filter estimate

- Stats box: Shows tracking error, lock-on status
 
