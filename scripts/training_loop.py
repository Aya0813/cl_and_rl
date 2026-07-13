
import argparse
import gc
import random
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:  # allow running even if wandb is not installed
    import wandb  # experiment tracking client
except ImportError:  # handle missing wandb dependency
    wandb = None  # disable wandb features when unavailable

from bucket_stats import update_bucket_loss_stats
from dataset_prep import prepare_math_dataset, sample_batch_from_buckets
from student_step import build_solution_only_batch, resolve_device, student_train_step
from teacher_policy import (
    build_teacher_observation,
    compute_compute_cost,
    compute_stability_penalty_kl,
    discounted_returns,
    init_bucket_probs,
    init_teacher_model,
    reinforce_teacher_update,
    sample_teacher_action,
    teacher_policy_probs,
)

def clear_device_memory():
    gc.collect() # what is gc - Python's garbage collection module - it's a way to free up memory by cleaning up unused objects in Python. In the context of training loops, especially when running multiple episodes or iterations, you can end up with a lot of intermediate tensors and models that consume GPU memory. By calling gc.collect(), you can help ensure that any unused objects are cleaned up and their memory is freed, which can help prevent out-of-memory errors and keep your training loop running smoothly. It's especially important in a loop where you're creating and deleting models and optimizers, as it helps manage memory more effectively.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sample_uniform_batch(dataset_split, m, seed): 
    # [ TO DIVE DEEPLER LATER ]
    """
    This function is used to sample a batch of examples uniformly at random from a given dataset split (e.g., validation or test set). 
    It's called "uniform" because it samples examples with equal probability, without any weighting or curriculum. 
    """
    rng = random.Random(seed)
    n = min(m, len(dataset_split)) # change this later cause not precise - technically dataset split should be larger than m, but we add this check just in case to avoid errors when the dataset is smaller than the requested batch size. If m is larger than the dataset size, we simply sample as many examples as are available in the dataset.
    indices = rng.sample(range(len(dataset_split)), n)
    return [dataset_split[i] for i in indices]


def clamp_batch_size(dataset_len, batch_size):
    # [ TO DIVE DEEPLER LATER ] -> (machi grande priority) do we really need this function seems extra... 
    """
    Clamp batch_size to not exceed dataset_len. If batch_size > dataset_len,
    warn the user that we're sampling fewer examples than requested.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0.")
    
    clamped = min(batch_size, dataset_len)
    
    # Warn if clamping occurred (batch_size was larger than available data)
    if clamped < batch_size:
        print(f"[WARNING] batch_size ({batch_size}) exceeds available dataset size ({dataset_len}). "
              f"Clamping to {clamped}. This may hurt validation/test performance.")
    
    return clamped


def evaluate_loss_accuracy(student_model, student_tokenizer, batch_examples, device, max_length=1024): 
    student_model.eval()
    with torch.no_grad():
        input_ids, attention_mask, labels, *_  = build_solution_only_batch(
            batch_examples=batch_examples,
            tokenizer=student_tokenizer,
            device=device,
            max_length=max_length,
        )
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = float(outputs.loss.item()) # why do we need to convert it to float ? - outputs.loss is a tensor, and by calling .item() we get the Python scalar value from that tensor. Then we wrap it in float() to ensure it's a standard Python float type, which can be easier to work with for logging, comparisons, and other operations outside of PyTorch. So the combination of .item() and float() gives us a simple scalar value for the loss that we can easily print or use in calculations.
        preds = outputs.logits.argmax(dim=-1) # [ TO DIVE DEEPER LATER ] outputs.logits has shape (batch_size, seq_len, vocab_size). By calling argmax(dim=-1), we get the index of the highest logit for each position in the sequence, which corresponds to the predicted token ID. This gives us a tensor of shape (batch_size, seq_len) containing the predicted token IDs for each example in the batch. 
        valid = labels != -100
        acc = 0.0 if valid.sum().item() == 0 else float((preds[valid] == labels[valid]).float().mean().item()) # This line computes the token-level accuracy for the predicted tokens compared to the true labels, but only for the valid positions (where labels != -100). Here's a breakdown:
        
        # 1. valid = labels != -100: This creates a boolean mask that is True for positions where the label is not -100 (i.e., valid tokens) and False for positions where the label is -100 (which are ignored in loss computation).
        # 2. preds[valid] == labels[valid]: This compares the predicted token IDs to the true token IDs, but only at the valid positions. It returns a boolean tensor that is True where the prediction is correct and False where it is incorrect.
        # 3. (preds[valid] == labels[valid]).float().mean(): This converts the boolean tensor to float (1.0 for correct predictions, 0.0 for incorrect) and then takes the mean to compute the average accuracy across all valid tokens.
        # 4. The if-else condition handles the case where there are no valid tokens (valid.sum().item() == 0). In that case, we set accuracy to 0.0 to avoid division by zero. Otherwise, we compute the accuracy as described.
    
        # NOTE: This is a token-level accuracy, which may not perfectly reflect the correctness of the solution in cases where there are multiple valid token sequences that can represent the same correct answer (e.g., "2+2" vs "4"). For a more semantic evaluation, we might want to implement a custom accuracy metric that can account for equivalent solutions, but for the purpose of this PoC, we'll keep it as token-level accuracy for simplicity.

        # [ TO DIVE DEEPER LATER ] 
        # the data we have is in the format of {"problem": ..., "solution": ...} so the model is given the problem as a prompt and is expected to generate the solution, and the loss and accuracy are computed based on how well the model generates the solution tokens (ignoring the prompt tokens).
        # BUT since theses are math equations the solution can be correct even if the tokens are not exactly the same (e.g. 2+2 and 4 are both correct solutions to the problem "What is 2+2?") so this token-level accuracy might not be the best metric for evaluating the model's performance on this task, and we might want to consider a more semantic evaluation metric in the future that can capture the correctness of the solution even if the exact tokens differ.
    return loss, acc


def compute_per_example_losses(student_model, student_tokenizer, batch_examples, device, max_length=1024):
    """
    Compute loss for each training example individually (one at a time).
    
    VISUAL EXAMPLE:
    ===============
    Suppose training_batch has 8 examples from mixed buckets:
    
    training_batch = [
        {"problem": "2+2", "solution": "4", "bucket": 1},       # easy
        {"problem": "15*17", "solution": "255", "bucket": 3},   # hard
        {"problem": "5+3", "solution": "8", "bucket": 1},       # easy
        {"problem": "99*99", "solution": "9801", "bucket": 3},  # hard
        {"problem": "1+1", "solution": "2", "bucket": 1},       # easy
        {"problem": "7*8", "solution": "56", "bucket": 2},      # medium
        {"problem": "100/5", "solution": "20", "bucket": 2},    # medium
        {"problem": "123*456", "solution": "56088", "bucket": 3},# hard
    ]
    
    STEP 1: Evaluate each example alone
    ===================================
    for ex in training_batch:
        loss_1 = model([example_1])  →  0.45  (easy, low loss)
        loss_2 = model([example_2])  →  1.23  (hard, high loss)
        loss_3 = model([example_3])  →  0.38  (easy, low loss)
        loss_4 = model([example_4])  →  1.89  (hard, very high loss)
        loss_5 = model([example_5])  →  0.22  (easy, low loss)
        loss_6 = model([example_6])  →  0.78  (medium)
        loss_7 = model([example_7])  →  0.65  (medium)
        loss_8 = model([example_8])  →  2.10  (hard, highest loss)
    
    RETURN:
    =======
    per_example_losses = [0.45, 1.23, 0.38, 1.89, 0.22, 0.78, 0.65, 2.10]
    
    STEP 2: Group by bucket (in update_bucket_loss_stats)
    ======================================================
    bucket_1 losses: [0.45, 0.38, 0.22]  →  avg = 0.35
    bucket_2 losses: [0.78, 0.65]        →  avg = 0.71
    bucket_3 losses: [1.23, 1.89, 2.10]  →  avg = 1.74
    
    STEP 3: Teacher sees this and decides
    ======================================
    o_t includes:
        running_avg_loss = {0: 0.35, 1: 0.71, 2: 1.74}  ← bucket 3 (idx 2) is hardest!
    
    Teacher's decision:
        "Bucket 3 has avg loss 1.74 (much higher than easy bucket at 0.35).
         I should sample MORE from bucket 3 to help the student improve."
    
    Next step's action a_t might become:
        a_t = [0.20, 0.20, 0.60]  ← increased from uniform [0.33, 0.33, 0.33]
    """
    losses = []
    student_model.eval()
    with torch.no_grad():
        for ex in batch_examples:
            loss, _ = evaluate_loss_accuracy(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=[ex],
                device=device,
                max_length=max_length,
            )
            losses.append(loss)
    return losses


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefixed_metrics(prefix, metrics): # the prefix is the phase we're at
    if not prefix: # if there is no prefix, return metrics as is
        return metrics
    return {f"{prefix}/{key}": value for key, value in metrics.items()} # if there is a metric prefix, we prepend it to each metric key for better organization in logging (e.g., "train/loss" instead of just "loss").


def create_student(student_model_name, student_lr, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=dtype,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=student_lr)
    return tokenizer, model, optimizer


def run_episode(
    D_tr,
    D_val_curve,
    D_test,
    buckets,
    seed,
    episode_id,
    fixed_val_curve_batch,
    global_test_batch,
    T,
    B,
    student_model,
    student_tokenizer,
    student_optimizer,
    student_lr,
    max_length,
    teacher_model,
    teacher_optimizer,
    teacher_lr,
    teacher_hidden_dim,
    gamma,
    lambda_cost,
    alpha_stability,
    use_advantage_baseline,
    baseline_type,
    normalize_advantage,
    val_batch_size,
    device,
    enable_wandb=False,
    use_teacher=True,
    update_teacher=True,
    run_phase="",
    step_offset=0,
):
    # del -> does not delete but just removes the local reference of those variables
    del D_tr, D_test, student_lr, teacher_lr, teacher_hidden_dim

    episode_seed = seed + episode_id
    set_random_seed(episode_seed)
    clear_device_memory()


    try:
        print(f"Starting episode {episode_id} with student seed {episode_seed}.\n")
        if use_teacher and update_teacher:
            print("Teacher is active and will be updated with REINFORCE at the end of this episode.\n")
        elif use_teacher:
            print("Teacher is active but frozen for this run (no REINFORCE update).\n")
        else:
            print("Uniform baseline run: teacher is disabled.\n")

        #  p_ref is the reference (uniform) probability distribution over buckets. It's used as a baseline for the teacher's policy
        num_buckets = len(buckets)
        p_ref = init_bucket_probs(num_buckets) # creates a uniform distribution—each bucket gets equal probability 1/num_buckets. This represents "no curriculum learning," just random sampling. -> p_ref defines what "neutral/no-curriculum" looks like, and it's used to regularize the teacher's learned policy.
        print(f"Reference bucket probabilities (p_ref): {p_ref}\n") # should be uniform distribution over buckets, e.g., [0.2, 0.2, 0.2, 0.2, 0.2] for 5 buckets.
        stats = None

        episode_stepwise_fixed_val_curve_losses = []
        episode_stepwise_fixed_val_curve_accs = []
        reward_history = []
        teacher_probs_history = []
        step_time_history = []
        trajectory_rewards = []
        trajectory_log_probs = []
        batch_clipped_ratios = []
        clipped_ratios_per_bucket = [[] for _ in range(len(buckets))] # DIVE DEEPER LATER 
        
        print(
            f"Using global fixed validation batch of size {len(fixed_val_curve_batch)} "
            "(sampled once per seed). This batch will be used for the validation curve throughout this run.\n"
        )
        print(
            f"Using global fixed test batch of size {len(global_test_batch)} "
            "(sampled once per seed). This batch will be used for final test evaluation throughout this run.\n"
        )

        for t in range(T):
            global_step = step_offset + (episode_id * T) + t
            step_start = time.perf_counter()
            print(f"\n*** Step {t + 1}/{T} | {'Curriculum' if use_teacher else 'Baseline'} ***")

            print("\n----- (1) Construct o_t -----")
            o_t = build_teacher_observation( # o_t = (current step, running avg loss per bucket, running change per bucket)
                step_t=t,
                stats=stats,
                K=num_buckets,
                T=T,
                device=device,
            )

            if not torch.isfinite(o_t).all():
                print("[WARNING] Non-finite values found in o_t before teacher forward pass.")
                print("Raw o_t values:", o_t)
                o_t = torch.nan_to_num(o_t, nan=0.0, posinf=0.0, neginf=0.0)
                print("Fixed o_t values:", o_t)

            print("o_t is a tensor of shape:", o_t.shape)
            print("o_t values:", o_t.tolist())

            print("\n----- (2) Teacher action sampling -----")
            if use_teacher:
                if update_teacher: # if we re not in the eval run 
                    # keep gradients, because teacher is learning
                    a_t, log_prob_t, teacher_probs = sample_teacher_action(
                        teacher_model=teacher_model,
                        observation=o_t,
                    )
                    trajectory_log_probs.append(log_prob_t)
                else: # use_teacher is true -> when teacher is frozen 
                    # PyTorch will still: run the teacher forward pass + compute probabilities + sample the action
                    # BUT will NOT: track those operations for backprop + store intermediate tensors needed for gradients + let this action contribute to a future teacher update
                    with torch.no_grad():  # teacher is only being used for inference/sampling
                        a_t, _, teacher_probs = sample_teacher_action(
                            teacher_model=teacher_model,
                            observation=o_t,
                        )
                    log_prob_t = None # DIVE DEEPER LATER
            else: # baseline run
                a_t = p_ref[:]
                teacher_probs = a_t[:]
                log_prob_t = None
                print("Baseline run: using uniform bucket probabilities (no teacher).")
                print(f"Uniform bucket probabilities (a_t): {a_t}\n")
                print(f"Reference bucket probabilities (p_ref): {p_ref}\n")

            print("\n----- (3-4) Sample training batch from buckets -----")
            training_batch, sampled_bucket_indices = sample_batch_from_buckets(
                buckets=buckets,
                a_t=a_t,
                B=B,
                seed=seed + (episode_id * 1000) + t,
            )

            bucket_counts = {
                k: sampled_bucket_indices.count(k)
                for k in range(num_buckets)
            }
            sampled_bucket_fractions = {
                k: bucket_counts[k] / len(training_batch) if training_batch else 0.0
                for k in range(num_buckets)
            }

            print("Training batch:")
            print(f"  Number of examples: {len(training_batch)}")
            if training_batch:
                print(f"  First example: {training_batch[0]}")
            print("\n")
            print("Sampled bucket counts:", bucket_counts)

            print("\n----- (5) Sample validation batch -----")
            val_batch = sample_uniform_batch( # note -> sampled EVERY step
                D_val_curve,
                m=clamp_batch_size(len(D_val_curve), val_batch_size),
                seed=seed + 30_000 + t, # now set so that -> ep1 step 2 and ep2 step 2 will have the same val_batch.... 
            )

            print(
                f"Seed {seed}: Sampled validation batch of size {len(val_batch)} at step {t + 1} "
                f"This batch is freshly sampled each step to compute the reward signal based on immediate validation improvement.\n"
            )

            print(f"Validation batch of length {len(val_batch)}:")
            if val_batch:
                print(f"  First example: {val_batch[0]}")
            print("\n")

            print("\n----- (6) Validation loss BEFORE learner update -----")
            val_loss_before, _ = evaluate_loss_accuracy(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch,
                device=device,
                max_length=max_length,         
                    # do we need max_length here if we're just computing losses? - yes because the model's forward pass might still require it, even if we're not doing any generation. 
                    # It's safer to include it to ensure that the input tensors are properly shaped and that we don't run into unexpected issues with sequence lengths during evaluation.
            )
            print(f"Validation loss before student update: {val_loss_before:.6f}\n")

            print("\n----- (7-8) Train student on one SFT step -----")
            _, train_loss, batch_clipped_ratio, clipped_ratios = student_train_step(
                model=student_model,
                tokenizer=student_tokenizer,
                optimizer=student_optimizer,
                batch_examples=training_batch,
                device=device,
                max_length=max_length,
            )
            # _ means we ignore the first return value (which is the updated model) -> since we're updating in-place, we don't need to capture it)
            print(f"Student train loss on this training batch: {train_loss:.6f}")

            batch_clipped_ratios.append(batch_clipped_ratio)
            for idx, bucket_idx in enumerate(sampled_bucket_indices): # this is to keep track of the clipped ratios per bucket, so that we can analyze how much clipping occurred for each bucket across the training steps. This information can be useful for understanding the training dynamics and for debugging purposes.
                clipped_ratios_per_bucket[bucket_idx].append(clipped_ratios[idx])

            print("\n----- (9)  Update running bucket statistics -----")

            # PURPOSE: Track which buckets are "easy" vs "hard" based on their current loss values.
                # This helps the teacher learn which buckets the student is struggling with.
 
            # WORKFLOW:
                # 1. Compute per-example loss: For each training example, evaluate the loss after training on that single example
                # 2. Group losses by bucket: Aggregate per-example losses by their bucket label
                # 3. Update running average: Maintain exponential moving average of loss per bucket (with smoothing alpha=0.3)
                # 4. Compute loss change: Track how much each bucket's loss changed since last step

            # WHY: 
                # The teacher's observation (o_t) includes running_avg_loss and loss_change per bucket.
                # The teacher uses this info to decide: "If bucket 1 had high loss last step, maybe I should sample from it more
                # to help the student improve on those harder examples."

            per_example_losses = compute_per_example_losses(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=training_batch,
                device=device,
                max_length=max_length,
            ) # what is per_example_losses? -> It's a list of loss values, one for each example in the training batch, computed by evaluating the student model on each example individually. This allows us to see how well the student is performing on each specific example, which we can then aggregate by bucket to update our running statistics for the teacher's observation. By having per-example losses, we can compute the average loss for each bucket and track how it changes over time, which is crucial information for the teacher to make informed decisions about curriculum sampling.
            
            # take the examples we just trained on, look at which bucket each one came from, look at each example’s loss, then summarize that information per bucket.
            stats = update_bucket_loss_stats(
                batch_examples=training_batch,
                bucket_labels=sampled_bucket_indices,
                losses=per_example_losses,  # losses for each training example (one loss per example)
                stats=stats,
                alpha=0.3,  # smoothing factor for exponential moving average (0.3 = 30% new info, 70% old info) [ is also a hyper parameter to tune - for now PoC]
            )
            # format of stats 
            # stats is a dictionary where keys are bucket indices and values are dictionaries containing:
            #   - 'running_avg_loss': the exponential moving average of losses for that bucket
            #   - 'loss_change': the change in loss since the last step for that bucket

            print("\n----- (10) Validation loss AFTER update on SAME val batch -----")
            val_loss_after, _ = evaluate_loss_accuracy(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch,
                device=device,
                max_length=max_length,
            )

            # This computes the validation loss on the freshly sampled (per step) validation batch (val_batch) right after the student model has been updated in the current step.
            # This value is used to compute the reward for the teacher (i.e., how much the validation loss improved after the update).

            print(f"Validation loss after student update on SAME val batch: {val_loss_after:.6f}\n")

            print("\n----- (11) Reward computation -----")
            validation_improvement = val_loss_before - val_loss_after
            compute_cost = compute_compute_cost(training_batch)
            stability_penalty = compute_stability_penalty_kl(a_t, p_ref)
            
            reward = validation_improvement - lambda_cost * compute_cost - alpha_stability * stability_penalty
            # alpha_stability and lambda_cost are CLI arguments that control how much we penalize the teacher for compute cost and for deviating from uniform sampling (stability).
            # for now we're ignoring them (default 0) and focusing on the val improv. 

            print(f"\nValidation improvement: {validation_improvement:.6f}\n")
            print(
                f"\nCompute cost: {compute_cost:.6f} || Stability penalty: {stability_penalty:.6f} || Final reward: {reward:.6f}\n"
            )

            trajectory_rewards.append(float(reward)) # why float(reward) ? -> reward is a scalar tensor, and we convert it to a Python float for easier logging, storage, and analysis. This ensures that when we append it to the trajectory_rewards list, we're storing a simple numeric value rather than a tensor object, which can be more cumbersome to work with outside of PyTorch.
            reward_history.append(float(reward))
            teacher_probs_history.append(teacher_probs)

            print("\n----- (12) Validation curve evaluation (on fixed batch across episodes) -----")
            # Validation curve on global fixed batch (consistent across all episodes)

            # This computes the validation loss and accuracy on a global fixed validation batch (fixed_val_curve_batch) that was sampled once at the start of the seed (before all episodes) and stays the same for all steps across all episodes.
            # This value is used to build a consistent validation curve across all episodes, so you can plot how the model improves on the same set of examples over time and across episodes.
            # val_loss_after → reward calculation, fresh batch each step
            # stepwise_fixed_val_curve_loss → learning curve plotting, global fixed batch for consistency across episodes

            stepwise_fixed_val_curve_loss, stepwise_fixed_val_curve_acc = evaluate_loss_accuracy(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=fixed_val_curve_batch, # We want to track the validation loss and accuracy over time on a consistent set of examples to see how the student's performance evolves throughout ALL episodes. By using a global fixed validation batch (sampled once per seed at the beginning), we can plot a meaningful validation curve that reflects the student's learning progress on the same data points across episodes, rather than having the validation metrics fluctuate due to different random samples per episode. This allows us to better visualize the impact of the teacher's curriculum decisions on the student's learning trajectory across the entire training run.
                device=device,
                max_length=max_length,
            )
            print(f"Validation on global fixed batch (same across all episodes) for current step {t + 1}:")
            print(f"   Loss: {stepwise_fixed_val_curve_loss:.6f}")
            print(f"   Accuracy: {stepwise_fixed_val_curve_acc:.6f}\n")

            episode_stepwise_fixed_val_curve_losses.append(stepwise_fixed_val_curve_loss)
            episode_stepwise_fixed_val_curve_accs.append(stepwise_fixed_val_curve_acc)

            ######## ######## ######## ######## ######## ########
            
            step_time = time.perf_counter() - step_start
            step_time_history.append(step_time)
            print(f"Step time: {step_time:.2f} || Batch clipped ratio: {batch_clipped_ratio:.4f}\n")
            # This is the wall-clock time it took to complete the entire step, from constructing the observation to computing the reward. Tracking step time can help us understand the computational efficiency of our training loop and identify any bottlenecks. If we see that certain steps are taking much longer than others, we can investigate further to optimize those parts of the code. Additionally, comparing step times between the curriculum and baseline runs can give us insight into whether the teacher's decisions are leading to more efficient learning (e.g., if the curriculum run has shorter step times due to better sampling, that could be a positive sign).
            # The batch clipped ratio is a metric that indicates the proportion of training examples in the current batch that had their loss values clipped during the student training step. Clipping is often used to prevent excessively large loss values from destabilizing training, especially when using techniques like gradient clipping or when dealing with outliers. A high batch clipped ratio might indicate that many examples in the batch are difficult for the student model, which could be a sign that the teacher's curriculum is sampling harder examples. Conversely, a low batch clipped ratio might suggest that the sampled examples are easier for the student. Tracking this ratio can provide insight into how the difficulty of the training batches evolves over time and how it correlates with the teacher's actions and the student's learning progress. 
                # for example an batch_clipped_ratio of 0.1 - what does it mean? - It means that 10% of the examples in the current training batch were clipped 

            print_gpu_memory(tag=f"after step {t + 1}/{T}") # tells whether memory is growing step by step

            # SUMMARY of what's GREAT To log: 
                # Student learning: train_loss, val_loss, val_improvement
                # Teacher behavior: reward, advantage, return, teacher_loss, entropy_teacher_policy
                # Curriculum dynamics [ MOST IMPORTANT to visualize CL dynamics ] : bucket_0_prob, bucket_1_prob, ..., bucket_k_prob
            
            if enable_wandb and wandb is not None and wandb.run is not None: # log step metrics only when wandb is active
                step_metrics = {
                    "train/loss": train_loss,
                    "reward/value": reward,
                    "reward/validation_improvement": validation_improvement,
                    "reward_val_batch/loss_before": val_loss_before,
                    "reward_val_batch/loss_after": val_loss_after,
                    "stepwise_fixed_val_curve/loss": stepwise_fixed_val_curve_loss,
                    "stepwise_fixed_val_curve/acc": stepwise_fixed_val_curve_acc,
                    "episode/id": episode_id,
                    "step/id": t,
                    # could add step_time, batch_clipped_ratio, etc. LATER 
                }
                # One line per bucket in W&B:
                # - teacher_prob_* shows the policy's intended sampling distribution
                # - sampled_fraction_* shows the actual batch composition at this step
                # - uniform_reference_* provides the flat baseline line (1 / K per bucket)
                step_metrics.update({f"bucket/teacher_prob_{i}": prob for i, prob in enumerate(teacher_probs)})
                step_metrics.update({f"bucket/sampled_fraction_{i}": frac for i, frac in sampled_bucket_fractions.items()})
                step_metrics.update({f"bucket/uniform_reference_{i}": prob for i, prob in enumerate(p_ref)})
                wandb.log(prefixed_metrics(run_phase, step_metrics), step=global_step)  # run_phase -> alredy passed as argument to run_episode 
                    # after prefixing teacher probs for each bucket -> teacher_train/bucket/teacher_prob_0 ... 

            clear_device_memory() 
            ######## ######## ######## ######## ######## ########

        print(f"\n===== Running {T} steps is done! =====\n")

        final_ep_test_loss, final_ep_test_acc = evaluate_loss_accuracy(
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            batch_examples=global_test_batch, # sampled once -> fixed across all episodes for consistency in final evaluation
            device=device,
            max_length=max_length,
        )
        print("\nFinal test evaluation on global fixed held-out batch (sampled once per seed):")
        print(f"Test loss: {final_ep_test_loss:.4f}, Test accuracy: {final_ep_test_acc:.4f}")

        general_clipped_ratio = sum(batch_clipped_ratios) / len(batch_clipped_ratios) if batch_clipped_ratios else 0.0
        clipped_ratio_per_bucket = [sum(ratios) / len(ratios) if ratios else 0.0 for ratios in clipped_ratios_per_bucket]

        teacher_loss = None

        # trajectory_rewards is a list of rewards collected at each step of the episode. 
        # We compute the discounted returns from these rewards to use in the REINFORCE update for the teacher model. 
        # The discounted return at each time step t is calculated as the sum of future rewards, discounted by a factor gamma raised to the power of the number of steps into the future. 
        # This gives more weight to immediate rewards and less weight to distant future rewards, which is a common practice in RL to encourage the agent (teacher) to focus on short-term gains while still considering long-term outcomes.
        returns = discounted_returns(trajectory_rewards, gamma=gamma) # [ TO DIVE DEEPLER LATER ]
        print("Trajectory rewards:", trajectory_rewards)
        print("Discounted returns:", returns)
        print_gpu_memory(tag="before teacher update")

        if use_teacher and update_teacher: # case of active teacher (teacher is used for sampling and updated at the end of the episode)
            if teacher_optimizer is None:
                raise ValueError("teacher_optimizer is required when update_teacher=True")
             
            teacher_loss = reinforce_teacher_update( # [ TO DIVE DEEPLER LATER ]
                optimizer=teacher_optimizer,
                log_probs=trajectory_log_probs,
                rewards=trajectory_rewards,
                gamma=gamma,
                use_advantage_baseline=use_advantage_baseline,
                baseline_type=baseline_type,
                normalize_advantage=normalize_advantage,
            )

            final_obs = build_teacher_observation(
                step_t=T - 1,
                stats=stats,
                K=num_buckets,
                T=T,
                device=device,
            )

            updated_probs = teacher_policy_probs( # this is to see what the teacher's policy would be after the update, given the final observation. This is useful for logging and analysis, to understand how the teacher's behavior has changed as a result of the REINFORCE update.
                teacher_model=teacher_model,
                observation=final_obs,
            ).detach().cpu().tolist()

        
            print("\n=========== TEACHER UPDATE once at end of episode ===========")
            print(f"Teacher loss: {teacher_loss:.6f}")
            print("Final teacher observation (o_T):", final_obs)
            print("Updated teacher bucket probabilities after REINFORCE update:", updated_probs)
        
        elif use_teacher: # case of frozen teacher (teacher is used for sampling but not updated)
            with torch.no_grad(): 
                 # [ TO DIVE DEEPLER LATER ] - why do we need build_teacher_observation() and teacher_policy_probs() here? 
                final_obs = build_teacher_observation(
                    step_t=T - 1,
                    stats=stats,
                    K=num_buckets,
                    T=T,
                    device=device,
                )
                updated_probs = teacher_policy_probs(
                    teacher_model=teacher_model,
                    observation=final_obs,
                ).detach().cpu().tolist()
            print("Frozen teacher run -> policy used for sampling, but no update was applied.")
            print("Teacher bucket probabilities at end of run:", updated_probs)
        
        else: # case of baseline run (no teacher, uniform sampling)
            updated_probs = p_ref[:]
            print(f"Baseline run -> no teacher update, probabilities remain uniform: {updated_probs}")

        
        ########## convergence_step  ##########
        # to know how QUICKLY the student model improved during training → whether the teacher's curriculum decisions led to FASTER convergence against baseline.
        # find the smallest loss in the episode_stepwise_fixed_val_curve_losses (which has T losses -> 1 per step) -> and save that step as the convergence step 
        # note -> in this context convergence means -> there is no loss that is lower
        convergence_step = 1 + min(range(len(episode_stepwise_fixed_val_curve_losses)), key=lambda i: episode_stepwise_fixed_val_curve_losses[i])
        print(
            f"\nConvergence step (lowest validation loss): Step {convergence_step} "
            f"with val_loss {episode_stepwise_fixed_val_curve_losses[convergence_step - 1]:.6f}"
        )

        # [ TO DIVE DEEPLER LATER ]
        global_step_end = step_offset + (episode_id * T) + (T - 1)
        if enable_wandb and wandb is not None and wandb.run is not None:
            summary_metrics = {
                "final_test_loss": final_ep_test_loss,
                "final_test_acc": final_ep_test_acc,
                "convergence_step": convergence_step,
            }
            if teacher_loss is not None:
                summary_metrics["teacher_loss"] = teacher_loss
            wandb.log(prefixed_metrics(run_phase, summary_metrics), step=global_step_end)

        # [ TO DIVE DEEPLER LATER ] find the best way to PLOT the end of episode metrics in the WANDB UI .... 

        return {
            "method": "curriculum" if use_teacher else "baseline",
            "seed": seed,
            "episode_id": episode_id,
            "T": T,
            "episode_stepwise_fixed_val_curve_losses": episode_stepwise_fixed_val_curve_losses,
            "episode_stepwise_fixed_val_curve_accs": episode_stepwise_fixed_val_curve_accs,
            "reward_history": reward_history,
            "teacher_probs_history": teacher_probs_history,
            "step_time_history": step_time_history,
            "avg_step_time": mean(step_time_history),
            "returns": returns,
            "teacher_loss": teacher_loss,
            "updated_teacher_probs": updated_probs,
            "final_ep_test_loss": final_ep_test_loss,
            "final_ep_test_acc": final_ep_test_acc,
            "convergence_step": convergence_step,
            "bucket_stats": stats,
            "general_clipped_ratio": general_clipped_ratio,
            "batch_clipped_ratios": batch_clipped_ratios,
            "clipped_ratio_per_bucket": clipped_ratio_per_bucket,
            "global_step_end": global_step_end,
        }
    finally:
        clear_device_memory()


def summarize_runs(label, runs):
    print(f"\n===== {label} Summary of across all runs =====")
    print(f"Runs: {len(runs)}")
    print(f"Avg final held-out test loss: {mean([r['final_ep_test_loss'] for r in runs]):.6f}")
    print(f"Avg final held-out test accuracy: {mean([r['final_ep_test_acc'] for r in runs]):.6f}")
    print(f"Avg convergence step: {mean([r['convergence_step'] for r in runs]):.2f}")
    print(f"Avg step time (seconds): {mean([r['avg_step_time'] for r in runs]):.2f}")


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(
            f"[GPU MEMORY] {tag} | "
            f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
            f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB | "
            f"max_allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )


def setup_experiment(
    seed,
    student_model_name,
    student_lr,
    teacher_lr,
    teacher_hidden_dim,
    val_batch_size,
    test_batch_size,
    device,
):
    del student_model_name, student_lr, teacher_lr, teacher_hidden_dim

    set_random_seed(seed)

    D_tr, D_val_curve, D_test, buckets_raw = prepare_math_dataset(
        seed=seed,
        train_ratio=0.8,
        val_ratio=0.1,
        holdout_ratio=0.1,
    )

    bucket_keys = sorted(buckets_raw.keys())
    buckets = {i: buckets_raw[k] for i, k in enumerate(bucket_keys)}

    print(f"Bucket keys: {bucket_keys}")
    print("Bucket sizes:")
    for i, k in enumerate(bucket_keys):
        print(f"  {i} ({k}): {len(buckets[i])} examples")

    m_val = clamp_batch_size(len(D_val_curve), val_batch_size)
    m_test = clamp_batch_size(len(D_test), test_batch_size)

    if device == "mps":
        m_val = min(m_val, 4)
        m_test = min(m_test, 4)

    fixed_val_curve_batch = sample_uniform_batch(
        D_val_curve,
        m=m_val,
        seed=seed + 10_000,
    )
    fixed_test_batch = sample_uniform_batch(
        D_test,
        m=m_test,
        seed=seed + 20_000,
    )
    # 10_000 and 20_000 -> The goal is keeping the random streams logically separated and easier to reason about.

    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    print(f"Validation curve batch size: {len(fixed_val_curve_batch)}")
    print(f"Fixed test batch size: {len(fixed_test_batch)}")

    return {
        "D_tr": D_tr,
        "D_val_curve": D_val_curve,
        "D_test": D_test,
        "buckets": buckets, # note = of the train set !! 
        "fixed_val_curve_batch": fixed_val_curve_batch,
        "fixed_test_batch": fixed_test_batch,
        "dtype": dtype,
    }


def create_teacher(num_buckets, device, teacher_hidden_dim, teacher_lr):
    teacher_model = init_teacher_model(
        K=num_buckets,
        device=device,
        hidden_dim=teacher_hidden_dim,
    )
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=teacher_lr)
    print(f"Initialized teacher model with {sum(p.numel() for p in teacher_model.parameters())} parameters.")
    return teacher_model, teacher_optimizer


def save_teacher_checkpoint(teacher_model, teacher_checkpoint, teacher_hidden_dim):
    checkpoint_path = Path(teacher_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "teacher_state_dict": teacher_model.state_dict(),
            "teacher_hidden_dim": teacher_hidden_dim,
        },
        checkpoint_path,
    )
    print(f"Saved trained teacher checkpoint to {checkpoint_path}")


def load_teacher_checkpoint(teacher_checkpoint, num_buckets, device, teacher_hidden_dim):
    checkpoint_path = Path(teacher_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_hidden_dim = checkpoint.get("teacher_hidden_dim", teacher_hidden_dim)
    teacher_model = init_teacher_model(
        K=num_buckets,
        device=device,
        hidden_dim=checkpoint_hidden_dim,
    )
    teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
    print(f"Loaded teacher checkpoint from {checkpoint_path}")
    return teacher_model


def freeze_teacher_model(teacher_model):
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)


def train_teacher_phase(
    experiment,
    seed,
    episodes,
    steps_per_episode,
    B,
    val_batch_size,
    student_model_name,
    student_lr,
    teacher_lr,
    teacher_hidden_dim,
    gamma,
    lambda_cost,
    alpha_stability,
    use_advantage_baseline,
    baseline_type,
    normalize_advantage,
    max_length,
    device,
    enable_wandb,
    teacher_checkpoint,
):
    print("\n========== Teacher Training Phase ==========")
    teacher_model, teacher_optimizer = create_teacher(
        num_buckets=len(experiment["buckets"]),
        device=device,
        teacher_hidden_dim=teacher_hidden_dim,
        teacher_lr=teacher_lr,
    )

    teacher_runs = []
    for ep in range(episodes):
        print(f"\n\n######## Teacher Training Episode={ep} ########\n")
        set_random_seed(seed)
        student_tokenizer, student_model, student_optimizer = create_student(
            student_model_name=student_model_name,
            student_lr=student_lr,
            device=device,
            dtype=experiment["dtype"],
        )
        episode_run = run_episode(
            D_tr=experiment["D_tr"],
            D_val_curve=experiment["D_val_curve"],
            D_test=experiment["D_test"],
            buckets=experiment["buckets"],
            T=steps_per_episode,
            B=B,
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            student_optimizer=student_optimizer,
            teacher_model=teacher_model,
            teacher_optimizer=teacher_optimizer,
            val_batch_size=val_batch_size,
            seed=seed,
            episode_id=ep,
            fixed_val_curve_batch=experiment["fixed_val_curve_batch"],
            global_test_batch=experiment["fixed_test_batch"],
            student_lr=student_lr,
            teacher_lr=teacher_lr,
            teacher_hidden_dim=teacher_hidden_dim,
            gamma=gamma,
            lambda_cost=lambda_cost,
            alpha_stability=alpha_stability,
            use_advantage_baseline=use_advantage_baseline,
            baseline_type=baseline_type,
            normalize_advantage=normalize_advantage,
            device=device,
            max_length=max_length,
            use_teacher=True, # IMPORTANT -> MAKE SURE THAT when use_teacher = True -> it is NOT the case for the baseline... 
            update_teacher=True,    # IMPORTANT -> MAKE SURE THAT it alligns with the rest of the code...
            enable_wandb=enable_wandb,
            run_phase="teacher_train",
        )
        teacher_runs.append(episode_run)
        del student_model, student_optimizer, student_tokenizer
        clear_device_memory()

    summarize_runs("teacher_train", teacher_runs)
    save_teacher_checkpoint(teacher_model, teacher_checkpoint, teacher_hidden_dim)
    return teacher_model, teacher_runs


def evaluate_frozen_teacher_phase(
    experiment,
    seed,
    eval_steps,
    B,
    val_batch_size,
    student_model_name,
    student_lr,
    teacher_lr,
    teacher_hidden_dim,
    gamma,
    lambda_cost,
    alpha_stability,
    use_advantage_baseline,
    baseline_type,
    normalize_advantage,
    max_length,
    device,
    enable_wandb,
    teacher_checkpoint,
    teacher_model=None,
    step_offset=0,
):
    print("\n========== Frozen Teacher Evaluation Phase ==========")
    if teacher_model is None:
        teacher_model = load_teacher_checkpoint(
            teacher_checkpoint=teacher_checkpoint,
            num_buckets=len(experiment["buckets"]),
            device=device,
            teacher_hidden_dim=teacher_hidden_dim,
        )
    freeze_teacher_model(teacher_model)

    set_random_seed(seed)
    student_tokenizer, student_model, student_optimizer = create_student(
        student_model_name=student_model_name,
        student_lr=student_lr,
        device=device,
        dtype=experiment["dtype"],
    )
    result = run_episode(
        D_tr=experiment["D_tr"],
        D_val_curve=experiment["D_val_curve"],
        D_test=experiment["D_test"],
        buckets=experiment["buckets"],
        T=eval_steps,
        B=B,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        student_optimizer=student_optimizer,
        teacher_model=teacher_model,
        teacher_optimizer=None,
        val_batch_size=val_batch_size,
        seed=seed,
        episode_id=0,
        fixed_val_curve_batch=experiment["fixed_val_curve_batch"],
        global_test_batch=experiment["fixed_test_batch"],
        student_lr=student_lr,
        teacher_lr=teacher_lr,
        teacher_hidden_dim=teacher_hidden_dim,
        gamma=gamma,
        lambda_cost=lambda_cost,
        alpha_stability=alpha_stability,
        use_advantage_baseline=use_advantage_baseline,
        baseline_type=baseline_type,
        normalize_advantage=normalize_advantage,
        device=device,
        max_length=max_length,
        use_teacher=True,
        update_teacher=False,
        enable_wandb=enable_wandb,
        run_phase="eval_frozen_teacher",
        step_offset=step_offset,
    )
    del student_model, student_optimizer, student_tokenizer
    clear_device_memory()
    return result


def evaluate_uniform_phase(
    experiment,
    seed,
    eval_steps,
    B,
    val_batch_size,
    student_model_name,
    student_lr,
    teacher_lr,
    teacher_hidden_dim,
    gamma,
    lambda_cost,
    alpha_stability,
    use_advantage_baseline,
    baseline_type,
    normalize_advantage,
    max_length,
    device,
    enable_wandb,
    step_offset=0,
):
    del teacher_lr, teacher_hidden_dim

    print("\n========== Uniform Baseline Evaluation Phase ==========")
    set_random_seed(seed)
    student_tokenizer, student_model, student_optimizer = create_student(
        student_model_name=student_model_name,
        student_lr=student_lr,
        device=device,
        dtype=experiment["dtype"],
    )
    result = run_episode(
        D_tr=experiment["D_tr"],
        D_val_curve=experiment["D_val_curve"],
        D_test=experiment["D_test"],
        buckets=experiment["buckets"],
        T=eval_steps,
        B=B,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        student_optimizer=student_optimizer,
        teacher_model=None,
        teacher_optimizer=None,
        val_batch_size=val_batch_size,
        seed=seed,
        episode_id=0,
        fixed_val_curve_batch=experiment["fixed_val_curve_batch"],
        global_test_batch=experiment["fixed_test_batch"],
        student_lr=student_lr,
        teacher_lr=0.0,
        teacher_hidden_dim=0,
        gamma=gamma,
        lambda_cost=lambda_cost,
        alpha_stability=alpha_stability,
        use_advantage_baseline=use_advantage_baseline,
        baseline_type=baseline_type,
        normalize_advantage=normalize_advantage,
        device=device,
        max_length=max_length,
        use_teacher=False,
        update_teacher=False,
        enable_wandb=enable_wandb,
        run_phase="eval_uniform",
        step_offset=step_offset,
    )
    del student_model, student_optimizer, student_tokenizer
    clear_device_memory()
    return result


def print_final_comparison(frozen_result, uniform_result):
    print("\n========== Final Comparison ==========")
    print(f"{'Run':<24} {'Final Test Loss':<18} {'Final Test Acc':<18}")
    print(f"{'Frozen Teacher':<24} {frozen_result['final_ep_test_loss']:<18.6f} {frozen_result['final_ep_test_acc']:<18.6f}")
    print(f"{'Uniform Baseline':<24} {uniform_result['final_ep_test_loss']:<18.6f} {uniform_result['final_ep_test_acc']:<18.6f}")
    print("=====================================\n")


def run_pipeline(
    mode,
    seed,
    episodes,
    steps_per_episode,
    eval_steps,
    B,
    val_batch_size,
    test_batch_size,
    student_model_name,
    student_lr,
    teacher_lr,
    teacher_hidden_dim,
    gamma,
    lambda_cost,
    alpha_stability,
    use_advantage_baseline,
    baseline_type,
    normalize_advantage,
    max_length,
    device,
    enable_wandb,
    wandb_project,
    wandb_entity,
    wandb_run_name,  # remember wandb_run_name is a CLI argument (i set it myself in the sbatch file for each experiment)
    teacher_checkpoint,
):
    clear_device_memory()
    device = resolve_device(device)
    eval_steps = eval_steps if eval_steps is not None else steps_per_episode

    import datetime
    run_start_time = datetime.datetime.now()
    run_name = wandb_run_name or f"{mode}_seed_{seed}"

    print("\n========== RUN CONFIGURATION ==========")
    print(f"[RUN START] {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"mode: {mode}")
    print(f"seed: {seed}")
    print(f"episodes: {episodes}")
    print(f"device: {device}")
    print(f"enable_wandb: {enable_wandb}")
    print(f"wandb_project: {wandb_project}")
    print(f"wandb_entity: {wandb_entity}")
    print(f"wandb_run_name: {run_name}")
    print(f"steps_per_episode: {steps_per_episode}")
    print(f"eval_steps: {eval_steps}")
    print(f"batch_size (B): {B}")
    print(f"val_batch_size: {val_batch_size}")
    print(f"test_batch_size: {test_batch_size}")
    print(f"student_model_name: {student_model_name}")
    print(f"student_lr: {student_lr}")
    print(f"max_length: {max_length}")
    print(f"teacher_lr: {teacher_lr}")
    print(f"teacher_hidden_dim: {teacher_hidden_dim}")
    print(f"gamma: {gamma}")
    print(f"lambda_cost: {lambda_cost}")
    print(f"alpha_stability: {alpha_stability}")
    print(f"use_advantage_baseline: {use_advantage_baseline}")
    print(f"baseline_type: {baseline_type}")
    print(f"normalize_advantage: {normalize_advantage}")
    print(f"teacher_checkpoint: {teacher_checkpoint}")
    print("=======================================\n")

    if enable_wandb and wandb is None:
        raise ImportError("wandb is not installed. Please run: pip install wandb")
    if enable_wandb and wandb is not None:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            group=f"seed_{seed}",
            config={
                "mode": mode,
                "seed": seed,
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "eval_steps": eval_steps,
                "batch_size": B,
                "val_batch_size": val_batch_size,
                "test_batch_size": test_batch_size,
                "student_model_name": student_model_name,
                "student_lr": student_lr,
                "teacher_lr": teacher_lr,
                "teacher_hidden_dim": teacher_hidden_dim,
                "gamma": gamma,
                "lambda_cost": lambda_cost,
                "alpha_stability": alpha_stability,
                "use_advantage_baseline": use_advantage_baseline,
                "baseline_type": baseline_type,
                "normalize_advantage": normalize_advantage,
                "max_length": max_length,
                "device": device,
                "teacher_checkpoint": teacher_checkpoint,
            },
        )

    if device == "mps":
        if B > 2:
            print(f"[MPS] Reducing B from {B} to 2 to avoid OOM.")
            B = 2
        if max_length > 256:
            print(f"[MPS] Reducing max_length from {max_length} to 256 to avoid OOM.")
            max_length = 256

    experiment = setup_experiment(
        seed=seed,
        student_model_name=student_model_name,
        student_lr=student_lr,
        teacher_lr=teacher_lr,
        teacher_hidden_dim=teacher_hidden_dim,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        device=device,
    )

    results = {}
    teacher_model = None
    teacher_phase_steps = episodes * steps_per_episode if mode in {"train_teacher", "full_pipeline"} else 0

    if mode in {"train_teacher", "full_pipeline"}: 
        teacher_model, teacher_runs = train_teacher_phase(
            experiment=experiment,
            seed=seed,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            B=B,
            val_batch_size=val_batch_size,
            student_model_name=student_model_name,
            student_lr=student_lr,
            teacher_lr=teacher_lr,
            teacher_hidden_dim=teacher_hidden_dim,
            gamma=gamma,
            lambda_cost=lambda_cost,
            alpha_stability=alpha_stability,
            use_advantage_baseline=use_advantage_baseline,
            baseline_type=baseline_type,
            normalize_advantage=normalize_advantage,
            max_length=max_length,
            device=device,
            enable_wandb=enable_wandb,
            teacher_checkpoint=teacher_checkpoint,
        )
        results["teacher_train"] = teacher_runs

    if mode in {"eval_frozen_teacher", "full_pipeline"}:
        frozen_result = evaluate_frozen_teacher_phase(
            experiment=experiment,
            seed=seed,
            eval_steps=eval_steps,
            B=B,
            val_batch_size=val_batch_size,
            student_model_name=student_model_name,
            student_lr=student_lr,
            teacher_lr=teacher_lr,
            teacher_hidden_dim=teacher_hidden_dim,
            gamma=gamma,
            lambda_cost=lambda_cost,
            alpha_stability=alpha_stability,
            use_advantage_baseline=use_advantage_baseline,
            baseline_type=baseline_type,
            normalize_advantage=normalize_advantage,
            max_length=max_length,
            device=device,
            enable_wandb=enable_wandb,
            teacher_checkpoint=teacher_checkpoint,
            teacher_model=teacher_model,
            step_offset=teacher_phase_steps if mode == "full_pipeline" else 0,
        )
        results["eval_frozen_teacher"] = frozen_result

    if mode in {"eval_uniform", "full_pipeline"}:
        uniform_result = evaluate_uniform_phase(
            experiment=experiment,
            seed=seed,
            eval_steps=eval_steps,
            B=B,
            val_batch_size=val_batch_size,
            student_model_name=student_model_name,
            student_lr=student_lr,
            teacher_lr=teacher_lr,
            teacher_hidden_dim=teacher_hidden_dim,
            gamma=gamma,
            lambda_cost=lambda_cost,
            alpha_stability=alpha_stability,
            use_advantage_baseline=use_advantage_baseline,
            baseline_type=baseline_type,
            normalize_advantage=normalize_advantage,
            max_length=max_length,
            device=device,
            enable_wandb=enable_wandb,
            step_offset=(teacher_phase_steps + eval_steps) if mode == "full_pipeline" else 0,
        )
        results["eval_uniform"] = uniform_result

    if mode == "full_pipeline":  
        print_final_comparison(results["eval_frozen_teacher"], results["eval_uniform"])

    run_end_time = datetime.datetime.now()
    run_duration = run_end_time - run_start_time
    print("\n========== RUN END ==========")
    print(f"[RUN END] {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run duration: {str(run_duration).split('.')[0]}")
    print("============================\n")

    if enable_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning with REINFORCE Teacher - phase-split pipeline")

    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility")
    parser.add_argument(
        "--mode",
        type=str,
        default="full_pipeline",
        choices=["train_teacher", "eval_frozen_teacher", "eval_uniform", "full_pipeline"],
        help="Which phase of the pipeline to run.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of reset-student episodes for teacher training")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"], help="Compute device")

    parser.add_argument("--steps_per_episode", type=int, default=20, help="Number of training steps per teacher-training episode")
    parser.add_argument("--eval_steps", type=int, default=None, help="Number of steps for the long persistent evaluation run")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (B)")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Test batch size. If None, use val_batch_size.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")

    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B", help="Student model identifier (HuggingFace)")
    parser.add_argument("--student_lr", type=float, default=1e-5, help="Student optimizer learning rate (AdamW)")

    parser.add_argument("--teacher_lr", type=float, default=1e-3, help="Teacher optimizer learning rate (Adam)")
    parser.add_argument("--teacher_hidden_dim", type=int, default=32, help="Hidden dimension of teacher MLP")
    parser.add_argument("--teacher_checkpoint", type=str, default="teacher_final.pt", help="Path to save/load the trained teacher checkpoint")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for REINFORCE")
    parser.add_argument("--lambda_cost", type=float, default=0.0, help="Weight on compute-cost penalty in reward")
    parser.add_argument("--alpha_stability", type=float, default=0.0, help="Weight on stability penalty (KL from uniform)")

    parser.add_argument("--use_advantage_baseline", type=int, default=1, help="Whether to use baseline in advantage computation (0 or 1)")
    parser.add_argument("--baseline_type", type=str, default="mean_return", help="Type of baseline: 'mean_return' or other")
    parser.add_argument("--normalize_advantage", type=int, default=1, help="Whether to normalize advantages (0 or 1)") 

    parser.add_argument("--wandb", type=int, default=0, help="Enable wandb logging (0 or 1)")
    parser.add_argument("--wandb_project", type=str, default="cl-rl", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B team/entity name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B custom run name (optional)") 

    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        seed=args.seed,
        episodes=args.episodes, # number of teacher updates
        steps_per_episode=args.steps_per_episode,  # length of each trajectory used to compute one teacher update (number of student udpates)
        eval_steps=args.eval_steps,
        B=args.batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size if args.test_batch_size is not None else args.val_batch_size,
        student_model_name=args.student_model_name,
        student_lr=args.student_lr,
        teacher_lr=args.teacher_lr,
        teacher_hidden_dim=args.teacher_hidden_dim,
        gamma=args.gamma,
        lambda_cost=args.lambda_cost,
        alpha_stability=args.alpha_stability,
        use_advantage_baseline=bool(args.use_advantage_baseline),
        baseline_type=args.baseline_type,
        normalize_advantage=bool(args.normalize_advantage),
        max_length=args.max_length,
        device=args.device,
        enable_wandb=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        teacher_checkpoint=args.teacher_checkpoint,
    )
