#!/bin/bash
#SBATCH --job-name=test

#SBATCH --chdir=/home/aya.elmir/cl_and_rl
#SBATCH --output=/home/aya.elmir/cl_and_rl/logs/17july/%x_%j.out
#SBATCH --error=/home/aya.elmir/cl_and_rl/logs/17july/%x_%j.err

#SBATCH --partition=gpuhigh2
#SBATCH --account=inception-energy-llm

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00



# Load modules
module load anaconda/2023.03
module load cuda/12.1

# Enable conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your env
conda activate cl_rl_env

# --- GPU check ---
echo "===== GPU status before run ====="
nvidia-smi


export WANDB_API_KEY="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"
export WANDB_PROJECT="cl_and_rl"
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run test - worked ;)) 
# python test_gpu.py

# python scripts/training_loop.py --steps 1 --batch_size 1

# python scripts/training_loop.py \
#   --device cuda \
#   --steps 1 \
#   --batch_size 1 \
#   --val_batch_size 8 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_run_name wandb_smoke_$(date +%Y_%m_%d_%H_%M)


# python scripts/training_loop.py \
#   --device cuda \
#   --steps 5 \
#   --batch_size 4 \
#   --val_batch_size 16 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_run_name medium_smoke_$(date +%Y_%m_%d_%H_%M)

# python scripts/training_loop.py \
#   --device cuda \
#   --steps 10 \
#   --batch_size 2 \
#   --val_batch_size 8 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_run_name oom_recovery_$(date +%Y_%m_%d_%H_%M)

# python scripts/training_loop.py \
#   --device cuda \
#   --steps 50 \
#   --batch_size 5 \
#   --val_batch_size 16 \
#   --gamma 0.99 \
#   --teacher_hidden_dim 32 \
#   --use_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name small_valid_50steps_b2_val16_$(date +%Y_%m_%d_%H_%M)

# need more episodes to see signal of change given the teacher updates once per ep. 

# python scripts/training_loop.py \
#   --device cuda \
#   --episodes 5 \
#   --steps 20 \
#   --batch_size 5 \
#   --val_batch_size 16 \
#   --gamma 0.99 \
#   --teacher_hidden_dim 32 \
#   --use_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name valid_ep5_steps20_b5_$(date +%Y_%m_%d_%H_%M)


# RUN different SEEDs at ONCE (for LATER)
# SEEDS=(11 22 33)
# SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID - 1))]}

# python3 scripts/training_loop.py \
#     --seed $SEED \
#     -- --episodes 3 \
#     .....



# python scripts/training_loop.py \
#   --device cuda \
#   --episodes 5 \
#   --steps 5 \
#   --batch_size 15 \
#   --val_batch_size 15 \
#   --gamma 0.95 \
#   --teacher_hidden_dim 32 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name smoke_ep5_steps5_b15_val15_$(date +%Y_%m_%d_%H_%M)

  # around 2 examples per bucket per step with batch_size 10, which is a good starting point for interpretability. With batch_size 5, we had around 1 example per bucket, which made it hard to interpret the teacher's behavior and the reward signal. With batch_size 10, we can start to see clearer patterns in how the teacher is sampling from the buckets and how that affects the student's learning and validation performance.


# --episodes 10 \  Enough to see whether teacher updates are doing anything.
# steps -> Good short horizon. Credit assignment is not too hard.
# batch_size -> Because with K=5, this gives roughly 5 examples per bucket under uniform sampling. Much more interpretable than B=5.
# val_batch_size = 32 -> More stable reward than 16.
# gamma -> Better than 0.99 for short noisy PoC episodes. With T=20, 0.99 makes rewards across the whole episode almost equally weighted.
# teacher_hidden_dim -> Good starting point. Keep teacher simple.


# NEXT MOVE TO -> But for interpreting learning behavior, I would move quickly to:
  # batch_size = 25 or 50
  # val_batch_size = 32 or 64
  # episodes = 10+


  # later change the code so direct the name of the err and output files contain the main changes per run 



# python scripts/training_loop.py \
#   --method curriculum \
#   --seed 11 \
#   --episodes 10 \
#   --steps 20 \
#   --batch_size 15 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name curriculum_seed11

# # and:

# python scripts/training_loop.py \
#   --method baseline \
#   --seed 11 \
#   --episodes 10 \
#   --steps 20 \
#   --batch_size 15 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name baseline_seed11

# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --episodes 2 \
#   --steps_per_episode 10 \
#   --eval_steps 10 \
#   --batch_size 10 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name test


# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --episodes 5 \
#   --steps_per_episode 10 \
#   --eval_steps 10 \
#   --batch_size 10 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name first_meaningful_full_pipeline_seed11



# COMPARISON 

# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 20 \
#   --batch_size 25 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name cl_full_pipeline_seed11_b25_steps20



# python scripts/training_loop.py \
#   --mode eval_uniform \
#   --device cuda \
#   --seed 11 \
#   --eval_steps 20 \
#   --batch_size 25 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name baseline_uniform_seed11_b25_steps20


# 14 juillet 

# check wandb for details 

# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 5 \
#   --steps_per_episode 5 \
#   --eval_steps 2 \
#   --batch_size 4 \
#   --val_batch_size 8 \
#   --test_batch_size 8 \
#   --max_length 256 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name smoke_eval_step_added_seed11


# ended up being an issue with the frozen teacher and eval uniform modes with properly logging into WANDB 


# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 1 \
#   --steps_per_episode 2 \
#   --eval_steps 2 \
#   --batch_size 4 \
#   --val_batch_size 8 \
#   --test_batch_size 8 \
#   --max_length 256 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name smoke_all_namespaces_offset_fix_seed11


# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 5 \
#   --steps_per_episode 2 \
#   --eval_steps 2 \
#   --batch_size 4 \
#   --val_batch_size 8 \
#   --test_batch_size 8 \
#   --max_length 256 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name smoke_eval_step_log_seed11


# logging should be fine now... 


# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 5 \
#   --steps_per_episode 10 \
#   --eval_steps 10 \
#   --batch_size 10 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed11_ep5_steps10_b10_rerun



##### REAL RUNS START FROM HERE #####


# Since batch_size > 10 is what seems to blow up, I would not push batch size next. The safest next move is to increase the things that improve the experiment signal without changing peak memory much:
# keep batch_size=10
# keep val_batch_size=32
# keep max_length=512
# increase episodes
# increase steps_per_episode

# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 11 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 20 \
#   --batch_size 10 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed11_ep10_steps20_b10_rerun_proper_logging


# just following ce que GPT recommend to run next ... 

# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 33 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 20 \
#   --batch_size 15 \
#   --val_batch_size 32 \
#   --test_batch_size 32 \
#   --max_length 256 \
#   --student_lr 1e-5 \
#   --teacher_lr 1e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed33_ep10_steps20_b15_len256


# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 33 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 20 \
#   --batch_size 15 \
#   --val_batch_size 16 \
#   --test_batch_size 16 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 3e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed33_ep10_steps20_b15_val16_changed_teacher_lr


# changing the TEACHER LR to 3e-3 and 3e-4 to see if it helps the teacher learn better.
# 3e-3 ended up being way better - so validating across seed 22 and 11 as well 
# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 22 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 20 \
#   --batch_size 15 \
#   --val_batch_size 16 \
#   --test_batch_size 16 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 3e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed22_ep10_steps20_b15_val16_len512_tlr3e-3


# ONLY valid if you see SAME pattern with seed 22 and 11 
# the seed-33 3e-3 curve is still separating at step 20 rather than flattening so more room for improvement - so increased to 100 eval steps instead of 20 to see if the teacher is still learning and improving the student..... 
# python scripts/training_loop.py \
#   --mode full_pipeline \
#   --device cuda \
#   --seed 22 \
#   --episodes 10 \
#   --steps_per_episode 20 \
#   --eval_steps 100 \
#   --batch_size 15 \
#   --val_batch_size 16 \
#   --test_batch_size 16 \
#   --max_length 512 \
#   --student_lr 1e-5 \
#   --teacher_lr 3e-3 \
#   --teacher_hidden_dim 32 \
#   --gamma 0.95 \
#   --use_advantage_baseline 1 \
#   --baseline_type mean_return \
#   --normalize_advantage 1 \
#   --lambda_cost 0.0 \
#   --alpha_stability 0.0 \
#   --wandb 1 \
#   --wandb_project cl-rl \
#   --wandb_entity CL-and-RL \
#   --wandb_run_name fp_seed22_ep10_steps20_eval100_b15_val16_len512_tlr3e-3


# indeed all 3 seeds behave as expected :)) - now trying 2 more seeds on same 100 eval steps to see if the pattern holds !! 


for SEED in 44 55; do
  python scripts/training_loop.py \
    --mode full_pipeline \
    --device cuda \
    --seed ${SEED} \
    --episodes 10 \
    --steps_per_episode 20 \
    --eval_steps 100 \
    --batch_size 15 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --max_length 512 \
    --student_lr 1e-5 \
    --teacher_lr 3e-3 \
    --teacher_hidden_dim 32 \
    --gamma 0.95 \
    --use_advantage_baseline 1 \
    --baseline_type mean_return \
    --normalize_advantage 1 \
    --lambda_cost 0.0 \
    --alpha_stability 0.0 \
    --wandb 1 \
    --wandb_project cl-rl \
    --wandb_entity CL-and-RL \
    --wandb_run_name fp_seed${SEED}_ep10_steps20_eval100_b15_val16_len512_tlr3e-3
done


# add de date of the current day l fou9 

# suggestion de Kirill 
  # call it reinforce with a baseline  