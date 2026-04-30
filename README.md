
env setup: 
uv venv --python 3.11
source .venv/bin/activate

libraries: datasets, torch, transfomers

project structure:
- `scripts/`: all python scripts (`training_loop.py`, `teacher_policy.py`, etc.)
- `results/`: run outputs and plots (`*.txt`, `*.png`)

run examples:
- training: `python3 scripts/training_loop.py --device auto`
- plotting: `python3 scripts/plot_results.py`


