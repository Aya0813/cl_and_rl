import re
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
INPUT_FILE = RESULTS_DIR / "T_20_output.txt"

# -------- Load file --------
with open(INPUT_FILE, "r") as f:
    text = f.read()


# -------- Helper function --------
def extract_curve(section, key):
    pattern = rf"{key} \(loss vs steps\):([\s\S]*?)(?:\n\n|\Z)"
    match = re.search(pattern, section)
    if not match:
        return []

    lines = match.group(1).strip().split("\n")
    values = []
    for line in lines:
        val = float(line.split(":")[1].strip())
        values.append(val)
    return values


# -------- Split runs --------
runs = text.split("########")

curriculum_curves = []
baseline_curves = []

for run in runs:
    if "Curriculum" in run:
        curve = extract_curve(run, "Validation loss curve")
        if curve:
            curriculum_curves.append(curve)

    elif "Baseline" in run:
        curve = extract_curve(run, "Validation loss curve")
        if curve:
            baseline_curves.append(curve)


# -------- Average curves --------
import numpy as np

def average_curves(curves):
    return np.mean(np.array(curves), axis=0)

curriculum_avg = average_curves(curriculum_curves)
baseline_avg = average_curves(baseline_curves)

steps = list(range(1, len(curriculum_avg) + 1))


# -------- Plot 1: Validation loss --------
plt.figure()
plt.plot(steps, curriculum_avg, label="Curriculum")
plt.plot(steps, baseline_avg, label="Baseline")
plt.xlabel("Steps")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Steps")
plt.legend()
plt.savefig(RESULTS_DIR / "val_loss_comparison.png")
plt.close()


# -------- Plot 2: Reward (example: first curriculum run) --------
def extract_reward(section):
    pattern = r"Reward over time:([\s\S]*?)(?:\n\n|\Z)"
    match = re.search(pattern, section)
    if not match:
        return []

    lines = match.group(1).strip().split("\n")
    return [float(line.split(":")[1]) for line in lines]


for run in runs:
    if "Seed=11" in run and "Curriculum" in run:
        reward = extract_reward(run)
        break

plt.figure()
plt.plot(range(1, len(reward) + 1), reward)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Reward over Time (Curriculum)")
plt.savefig(RESULTS_DIR / "reward_curve.png")
plt.close()


# -------- Plot 3: Final test loss --------
curriculum_test = []
baseline_test = []

pattern = r"Final held-out test loss: ([0-9.]+)"

for run in runs:
    match = re.search(pattern, run)
    if match:
        val = float(match.group(1))
        if "Curriculum" in run:
            curriculum_test.append(val)
        elif "Baseline" in run:
            baseline_test.append(val)

plt.figure()
x = np.arange(len(curriculum_test))
plt.bar(x - 0.2, curriculum_test, width=0.4, label="Curriculum")
plt.bar(x + 0.2, baseline_test, width=0.4, label="Baseline")
plt.xlabel("Seed")
plt.ylabel("Test Loss")
plt.title("Final Test Loss per Seed")
plt.legend()
plt.savefig(RESULTS_DIR / "test_loss_bar.png")
plt.close()

print("Plots saved!")
