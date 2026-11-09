# CL + RL

News: Research accepted for poster presentation at a NeurIPS 2026 workshop.

Small curriculum-learning experiment for math data with a trainable RL teacher:
- Student model: `Qwen/Qwen3.5-0.8B`
- Teacher policy: REINFORCE over bucket-sampling distributions
- Baseline: uniform bucket sampling

## Core idea

The project treats curriculum learning as a control problem over data buckets. Training examples are grouped by difficulty buckets, and a lightweight teacher policy decides how much probability mass to assign to each bucket before every student update.

At each step:
1. The teacher observes where we are in the episode plus running bucket-level loss statistics.
2. It outputs a distribution over buckets.
3. A training batch is sampled from those buckets using that distribution.
4. The student takes one supervised fine-tuning step on the sampled batch.
5. The teacher gets rewarded when that step improves validation loss.

## Key implementation ideas in `scripts/training_loop.py`

- **Phase-split pipeline**
  The main loop is organized into distinct modes:
  - `train_teacher`: train the teacher across episodes
  - `eval_frozen_teacher`: freeze the learned teacher and test its curriculum on a fresh student
  - `eval_uniform`: compare against the uniform-sampling baseline
  - `full_pipeline`: run all of the above end to end

- **Teacher learns across student resets**
  During teacher training, each episode starts with a newly initialized student. This makes the teacher learn a curriculum strategy that transfers across student restarts instead of overfitting to one continuous student trajectory.

- **Bucket-level observation, not raw examples**
  The teacher does not inspect individual problems. Its observation is a compact summary made of:
  - training step
  - running average loss per bucket
  - bucket-wise loss change

  This keeps the teacher small and makes the curriculum decision depend on how the student is progressing across difficulty regions.

- **Distribution-valued actions**
  The teacher does not pick a single bucket. It produces a probability distribution over buckets, and the sampler draws a mixed batch from that distribution. That lets the curriculum shift gradually instead of jumping between discrete choices.

- **Reward = immediate validation improvement**
  The reward is based on validation loss before vs. after the student update on the same freshly sampled validation batch. 

- **Two validation roles**
  The loop uses validation in two different ways:
  - a freshly sampled validation batch each step for the reward signal
  - a fixed validation batch for evaluation across steps and episodes

  Test evaluation is also done on a fixed held-out batch for consistent final comparison.

- **Curriculum compared to a real baseline**
  The learned teacher is evaluated against a uniform curriculum under the same student setup, dataset split, and step budget. The point is not just to train a teacher, but to check whether its bucket allocation improves downstream student learning.

- **Instrumentation for curriculum dynamics**
  The loop tracks teacher probabilities, sampled bucket fractions, reward history, step times, clipped-ratio statistics, and final held-out metrics. The emphasis is on making curriculum behavior observable, not only reporting final loss.

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
- `wandb` 

## Run
```bash
sbatch test.sh
```
