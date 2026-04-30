# CL + RL (Very Basic)

Tiny experiment for curriculum learning on math data:
- Student model: `Qwen/Qwen3.5-0.8B`
- Teacher policy: REINFORCE over difficulty buckets
- Baseline: vanilla uniform sampling

## Env setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Key libraries
- `torch`
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `wandb` (optional)

## Run
```bash
python scripts/training_loop.py
```

Optional flags:
```bash
python scripts/training_loop.py --steps 20 --batch_size 8 --device auto --wandb 1
```
