
import argparse
import gc
import random
import time
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


def run_episode(
    # ===== DATASET & EXPERIMENT SETUP (not CLI args) =====
    D_tr,                  # Training dataset (80% of data)
    D_val_curve,           # Validation dataset for reward signal (10% of data)
    D_test,                # Test dataset for final evaluation (10% of data)
    buckets,               # Dict mapping bucket indices to example lists
    seed,                  # Random seed (from loop, not direct CLI arg)
    episode_id,            # Episode number within seed (from loop, not direct CLI arg)
    global_val_curve_batch,# Global fixed validation batch (sampled once per seed, used for all episodes)
    global_test_batch,     # Global fixed test batch (sampled once per seed, used for all episodes)

    T,                     # Number of training steps (from --steps CLI arg)
    B,                     # Training batch size (from --batch_size CLI arg)
    
    # ===== STUDENT MODEL (persistent across episodes) =====
    student_model,         # Pre-initialized student model (persists across episodes)
    student_tokenizer,     # Pre-initialized student tokenizer (persists across episodes)
    student_optimizer,     # Pre-initialized student optimizer (persists across episodes)
    
    # ===== STUDENT MODEL CONFIGURATION (not CLI args) =====
    student_lr,            # Student learning rate (hardcoded: 1e-5)
    max_length,            # Max sequence length (hardcoded: 512)
    
    # ===== TEACHER MODEL (persistent across episodes) =====
    teacher_model,         # Pre-initialized teacher model (persists and learns across episodes)
    teacher_optimizer,     # Pre-initialized teacher optimizer (persists across episodes)
    
    # ===== TEACHER MODEL CONFIGURATION (CLI args) =====
    teacher_lr,            # Teacher learning rate (hardcoded: 1e-2)
    teacher_hidden_dim,    # [CLI arg] Hidden dimension of teacher MLP (type: int, default=32) -> Start small (e.g., 32), then tune (16/32/64/128).
    
    # ===== TEACHER POLICY PARAMETERS (CLI args) =====
    gamma,                 # [CLI arg] Discount factor for REINFORCE (type: float, default=0.99) - determines weight of future vs immediate rewards
    lambda_cost,           # [CLI arg] Weight on compute-cost penalty in reward (type: float, default=0.0)
    alpha_stability,       # [CLI arg] Weight on stability penalty (KL div from uniform) (type: float, default=0.0)
    
    # ===== REINFORCE BASELINE CONFIGURATION (CLI args) =====
    use_advantage_baseline,          # [CLI arg] Whether to use baseline in advantage computation (type: bool via int, default=1)
    baseline_type,         # [CLI arg] Type of baseline: "mean_return" or other (type: str, default="mean_return")
    normalize_advantage,   # [CLI arg] Whether to normalize advantages (type: bool via int, default=1)
    
    # ===== VALIDATION CONFIGURATION (CLI arg) =====
    val_batch_size,        # [CLI arg] Batch size for validation sampling (type: int, default=8) - sampled fresh at each step from D_val_curve
    
    # ===== DEVICE & RUNTIME (CLI arg) =====
    device,                # [CLI arg] Compute device: "auto", "cpu", "mps", "cuda" (type: str, default="auto")
    enable_wandb=False,    # whether to emit metrics to wandb
    
    # ===== EXPERIMENT CONDITION (not CLI arg, controlled in loop) =====
    use_teacher=True,      # Whether to use curriculum learning (teacher) vs baseline - False = vanilla uniform sampling
):
    episode_seed = seed + episode_id  # ensure different seed per episode for reproducibility
    random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    torch.cuda.manual_seed_all(episode_seed)
    clear_device_memory()
    
    # Student and teacher models are passed in as persistent across episodes
    # They contain the state learned from previous episodes (if any)

    try:
        print(f"Using persistent student model (Episode {episode_id}). Student has learned from prior episodes.\n")
        if use_teacher:
            print(f"Using persistent teacher model (Episode {episode_id}). Teacher has learned from prior episodes.\n")

        num_buckets = len(buckets)

        # p_ref is the reference (uniform) probability distribution over buckets. It's used as a baseline for the teacher's policy.
        p_ref = init_bucket_probs(num_buckets) # creates a uniform distribution—each bucket gets equal probability 1/num_buckets. This represents "no curriculum learning," just random sampling. -> p_ref defines what "neutral/no-curriculum" looks like, and it's used to regularize the teacher's learned policy.
        print (f"Reference bucket probabilities (p_ref): {p_ref}\n") # should be uniform distribution over buckets, e.g., [0.2, 0.2, 0.2, 0.2, 0.2] for 5 buckets.
        stats = None

        per_ep_val_loss_curve = []
        per_ep_val_acc_curve = []
        reward_history = []
        teacher_probs_history = []
        step_time_history = []

        trajectory_rewards = []
        trajectory_log_probs = []

        # Use global fixed validation and test batches that were sampled once per seed
        # This ensures consistent convergence tracking across all episodes
        print(f"Using global fixed validation batch of size {len(global_val_curve_batch)} (sampled once per seed). This batch will be used for the validation curve throughout this run.\n")
        print(f"Using global fixed test batch of size {len(global_test_batch)} (sampled once per seed). This batch will be used for final test evaluation throughout this run.\n")


        for t in range(T): 
            
            global_step = episode_id * T + t
            step_start = time.perf_counter()
            print(f"\n*** Step {t + 1}/{T} | {'Curriculum' if use_teacher else 'Baseline'} ***")

            # 1) Construct observation
            print("\n----- (1) Construct o_t -----")
            o_t = build_teacher_observation(
                step_t=t,
                stats=stats,
                K=num_buckets,
                T=T,
                device=device,
            )

            # Safety check: teacher input must never contain NaN/inf values.
            if not torch.isfinite(o_t).all():
                print("[WARNING] Non-finite values found in o_t before teacher forward pass.")
                print("Raw o_t values:", o_t)

                # Replace:
                # NaN  -> 0.0
                # +inf -> 0.0
                # -inf -> 0.0

                o_t = torch.nan_to_num(o_t, nan=0.0, posinf=0.0, neginf=0.0)
                print("Fixed o_t values:", o_t)


            print("o_t is a tensor of shape:", o_t.shape)  # should be (1, num_features) - a single observation vector for the current step
            print("o_t values:", o_t.tolist())


            # 2) Teacher chooses action a_t (distribution over buckets)
            print("\n----- (2) Teacher action sampling -----")
            if use_teacher:
                a_t, log_prob_t, teacher_probs = sample_teacher_action(
                    teacher_model=teacher_model,
                    observation=o_t,
                )
                trajectory_log_probs.append(log_prob_t)
            else:   # vanilla SFT baseline = uniform buckets
                # If you did that, both a_t and p_ref would point to the same list object. 
                # If you later modified a_t, it would also modify p_ref. By using [:], you create a separate copy, so modifications to a_t won't affect p_ref.
                # In the baseline run (use_teacher=False), you want to use the uniform distribution without accidentally modifying it.
                a_t             = p_ref[:]  # creates a shallow copy of the list p_ref  - e.g. Create a copy: [0.2, 0.2, 0.2, 0.2, 0.2]
                teacher_probs   = a_t[:]  # Another copy (for logging)
                log_prob_t      = None # log_prob_t is only relevant when using the teacher, since it's needed for the REINFORCE update. In the baseline case, we won't be doing a policy update, so we can set it to None or just ignore it.
                print ("Baseline run: using uniform bucket probabilities (no teacher).")
                print(f"Uniform bucket probabilities (a_t): {a_t}\n")
                print(f"Reference bucket probabilities (p_ref): {p_ref}\n")


            # 3-4) Sample training batch from buckets via categorical(a_t)
            print("\n----- (3-4) Sample training batch from buckets -----")
            training_batch, sampled_bucket_indices = sample_batch_from_buckets(
                buckets=buckets,
                a_t=a_t,
                B=B,
                seed=seed + (episode_id * 1000) + t, # ensure different sampling each step and episode (note that it is t dependant - unlike test sampling)
            )
            # sampled_bucket_indices = “which pool each training example came from”

            bucket_counts = {
                k: sampled_bucket_indices.count(k)
                for k in range(num_buckets)
            }

            # print info about training_batch 
            print("Training batch:")
            print(f"  Number of examples: {len(training_batch)}")
            if training_batch:
                print(f"  First example: {training_batch[0]}")
            print ("\n")
            print("Sampled bucket counts:", bucket_counts) # VERY crutial to see BECAUSE -> If the batch composition barely changes, then the curriculum policy is not really affecting training.


            # 5) Sample one validation batch from fixed D_val -> AT EACH STEP
            print("\n----- (5) Sample validation batch -----")
            val_batch = sample_uniform_batch(D_val_curve, m=clamp_batch_size(len(D_val_curve), val_batch_size), seed=seed + 30_000 + t)
            # notice = we re not including episode_id in the validation sampling seed -> which makes episode-to-episode comparisons cleaner [ can be modified later to be more statistically robust]
            print(f"Sampled validation batch of size {len(val_batch)} at step {t + 1} for reward evaluation (seed {seed}). This batch is freshly sampled each step to compute the reward signal based on immediate validation improvement.\n")
            print (f"Validation batch of length {len(val_batch)}:")
            if val_batch:
                print(f"  First example: {val_batch[0]}")
            print("\n")

            # 6) Validation loss BEFORE learner update
            print("\n----- (6) Validation loss BEFORE learner update -----")
            val_loss_before, _ = evaluate_loss_accuracy( 
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch, # <-- Uses the sampled val_batch 
                device=device,
                max_length=max_length, 
                    # do we need max_length here if we're just computing losses? - yes because the model's forward pass might still require it, even if we're not doing any generation. 
                    # It's safer to include it to ensure that the input tensors are properly shaped and that we don't run into unexpected issues with sequence lengths during evaluation.
            )
            print(f"Validation loss before student update: {val_loss_before:.6f}\n")

            # 7-8) Train learner one SFT step
            print("\n----- (7-8) Train student on one SFT step -----")
            # Train step returns loss and clipping stats
            _, train_loss, batch_clipped_ratio, clipped_ratios = student_train_step(
                model=student_model,
                tokenizer=student_tokenizer,
                optimizer=student_optimizer,
                batch_examples=training_batch,
                device=device,
                max_length=max_length,
            )
            print (f"Student train loss on this training batch: {train_loss:.6f}")


            # Track batch clipped ratio for this step
            if 'batch_clipped_ratios' not in locals():
                batch_clipped_ratios = []  # List to store batch clipped ratios
            batch_clipped_ratios.append(batch_clipped_ratio)
            
            # Track per-bucket clipped ratios for this step
            if 'clipped_ratios_per_bucket' not in locals():
                clipped_ratios_per_bucket = [[] for _ in range(len(buckets))]  # List of lists
            
            # Assign each example's clipped ratio to its bucket
            for idx, bucket_idx in enumerate(sampled_bucket_indices):
                clipped_ratios_per_bucket[bucket_idx].append(clipped_ratios[idx])

            # 9) Update running bucket statistics
            # PURPOSE: Track which buckets are "easy" vs "hard" based on their current loss values.
            # This helps the teacher learn which buckets the student is struggling with.
            # 
            # WORKFLOW:
            # 1. Compute per-example loss: For each training example, evaluate the loss after training on that single example
            # 2. Group losses by bucket: Aggregate per-example losses by their bucket label
            # 3. Update running average: Maintain exponential moving average of loss per bucket (with smoothing alpha=0.3)
            # 4. Compute loss change: Track how much each bucket's loss changed since last step
            # 
            # WHY: The teacher's observation (o_t) includes running_avg_loss and loss_change per bucket.
            # The teacher uses this info to decide: "If bucket 1 had high loss last step, maybe I should sample from it more
            # to help the student improve on those harder examples."
            
            print("\n----- (9)  Update running bucket statistics -----")
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
                alpha=0.3,  # smoothing factor for exponential moving average (0.3 = 30% new info, 70% old info)
            )

            # 10) Validation loss AFTER update on SAME val batch
            print("\n----- (10) Validation loss AFTER update on SAME val batch -----")

            val_loss_after, _ = evaluate_loss_accuracy( # note -> this is for the reward signal and reflects immediate improvement on a random sample (which can fluctuate).
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch, # <-- SAME val_batch as the one used for BEFORE the update (not resampled)
                device=device,
                max_length=max_length,
            )
            # This computes the validation loss on the freshly sampled (per step) validation batch (val_batch) right after the student model has been updated in the current step.
            # This value is used to compute the reward for the teacher (i.e., how much the validation loss improved after the update).

            print (f"Validation loss after student update: {val_loss_after:.6f}\n")

            # 11) Reward
            print("\n----- (11) Reward computation -----")
            validation_improvement = val_loss_before - val_loss_after
            compute_cost = compute_compute_cost(training_batch)
            stability_penalty = compute_stability_penalty_kl(a_t, p_ref)
            reward = validation_improvement - lambda_cost * compute_cost - alpha_stability * stability_penalty

            print(f"\nValidation improvement: {validation_improvement:.6f}\n")
            print(f"\nCompute cost: {compute_cost:.6f} || Stability penalty: {stability_penalty:.6f} || Final reward: {reward:.6f}\n")

            trajectory_rewards.append(float(reward)) # for student/validation values, be careful. Store Python floats, not tensors with graphs. -> This prevents accidental tensor graph storage later.
            reward_history.append(float(reward))
            teacher_probs_history.append(teacher_probs)

            ####### 

            # Validation curve on global fixed batch (consistent across all episodes)
            per_step_val_curve_loss, per_step_val_curve_acc = evaluate_loss_accuracy( # note -> this is for plotting a smooth, comparable learning curve across steps and episodes, since it always uses the same data.
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=global_val_curve_batch, # We want to track the validation loss and accuracy over time on a consistent set of examples to see how the student's performance evolves throughout ALL episodes. By using a global fixed validation batch (sampled once per seed at the beginning), we can plot a meaningful validation curve that reflects the student's learning progress on the same data points across episodes, rather than having the validation metrics fluctuate due to different random samples per episode. This allows us to better visualize the impact of the teacher's curriculum decisions on the student's learning trajectory across the entire training run.
                device=device,
                max_length=max_length,
            )
            
            # This computes the validation loss and accuracy on a global fixed validation batch (global_val_curve_batch) that was sampled once at the start of the seed (before all episodes) and stays the same for all steps across all episodes.
            # This value is used to build a consistent validation curve across all episodes, so you can plot how the model improves on the same set of examples over time and across episodes.
            # val_loss_after → reward calculation, fresh batch each step
            # per_step_val_curve_loss → learning curve plotting, global fixed batch for consistency across episodes

            print (f"Validation on global fixed batch (sampled once per seed) for current step {t + 1}:")
            print (f"   Loss: {per_step_val_curve_loss:.6f}")
            print (f"   Accuracy: {per_step_val_curve_acc:.6f}\n")
            
            per_ep_val_loss_curve.append(per_step_val_curve_loss)
            per_ep_val_acc_curve.append(per_step_val_curve_acc)

            ####### 

            step_time = time.perf_counter() - step_start
            step_time_history.append(step_time)

            print(f"Step time: {step_time:.2f} || Batch clipped ratio: {batch_clipped_ratio:.4f}\n") 
            # This is the wall-clock time it took to complete the entire step, from constructing the observation to computing the reward. Tracking step time can help us understand the computational efficiency of our training loop and identify any bottlenecks. If we see that certain steps are taking much longer than others, we can investigate further to optimize those parts of the code. Additionally, comparing step times between the curriculum and baseline runs can give us insight into whether the teacher's decisions are leading to more efficient learning (e.g., if the curriculum run has shorter step times due to better sampling, that could be a positive sign).
            # The batch clipped ratio is a metric that indicates the proportion of training examples in the current batch that had their loss values clipped during the student training step. Clipping is often used to prevent excessively large loss values from destabilizing training, especially when using techniques like gradient clipping or when dealing with outliers. A high batch clipped ratio might indicate that many examples in the batch are difficult for the student model, which could be a sign that the teacher's curriculum is sampling harder examples. Conversely, a low batch clipped ratio might suggest that the sampled examples are easier for the student. Tracking this ratio can provide insight into how the difficulty of the training batches evolves over time and how it correlates with the teacher's actions and the student's learning progress. 
                # for example an batch_clipped_ratio of 0.1 - what does it mean? - It means that 10% of the examples in the current training batch were clipped 

            print_gpu_memory(tag=f"after step {t + 1}/{T}") # tells whether memory is growing step by step

            if enable_wandb and wandb is not None and wandb.run is not None:  # log step metrics only when wandb is active
                wandb.log({  # send one metrics payload per step
                    "train/loss": train_loss,
                    "reward/value": reward,
                    "reward/validation_improvement": validation_improvement,
                    "reward_val_batch/loss_before": val_loss_before,
                    "reward_val_batch/loss_after": val_loss_after,
                    "per_step_val_curve/loss": per_step_val_curve_loss,
                    "per_step_val_curve/acc": per_step_val_curve_acc,
                    "episode/id": episode_id,
                    **{f"bucket/prob_{i}": prob for i, prob in enumerate(teacher_probs)},
                    # ** -> unpack the dictionary of bucket probabilities into individual key-value pairs for logging
                }, step = global_step)  # close step-level wandb payload

                # SUMMARY of what's GREAT To log: 
                    # Student learning
                        # train_loss
                        # val_loss
                        # val_improvement
                    # Teacher behavior
                        # reward
                        # advantage
                        # return
                        # teacher_loss
                        # entropy_teacher_policy
                    # Curriculum dynamics [ MOST IMPORTANT to visualize CL dynamics ]
                        # bucket_0_prob
                        # bucket_1_prob
                        # ...
                        # bucket_k_prob

            clear_device_memory()

        ####### ####### ####### episode done ####### ####### ####### 
        
        print (f"\n===== Running {T} steps is done! =====\n")

        ##### 

        # Using global fixed test batch (sampled once per seed, used for all episodes)
        final_ep_test_loss, final_ep_test_acc = evaluate_loss_accuracy(
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            batch_examples=global_test_batch,  # <-- Global fixed test evaluation
            device=device,
            max_length=max_length,
        ) 
        print (f"\nFinal test evaluation on global fixed held-out batch (sampled once per seed):")
        print (f"Test loss: {final_ep_test_loss:.4f}, Test accuracy: {final_ep_test_acc:.4f}")

        # difference between val and test
            # Val batch during training: Resampled fresh every step (seed + 20_000 + t changes each step)
            # Test batch at the end: Sampled once after episode is complete (fixed seed)
        
        # HOW TO INTERPRET Validation VS test metrics:  
            # The validation loss and accuracy that we track at each step during the episode are meant to provide a real-time signal of the student's performance and learning progress on a consistent set of examples (the fixed_val_batch). This allows us to see how the student's performance evolves over time in response to the teacher's curriculum decisions. The validation metrics can fluctuate from step to step based on the training updates and the teacher's actions, and they serve as a feedback signal for the teacher's policy learning.
            # The final test loss and accuracy are meant to provide an overall evaluation of the student's performance on held-out data after the entire curriculum learning process has been completed. These metrics are computed on a separate test set that was not used during training or for the validation curve, so they serve as an unbiased assessment of how well the student has learned to solve the problems after being guided by the teacher's curriculum. When interpreting these final test metrics, you can compare them between the curriculum and baseline runs to see if the teacher's guidance led to better generalization performance on unseen data. Additionally, you can analyze how the final test performance correlates with the validation curves and the teacher's actions throughout the episode to gain insights into which curriculum strategies were most effective in improving the student's learning outcomes.

        ##### 

        
        # Compute overall clipped ratio for the episode
        general_clipped_ratio = sum(batch_clipped_ratios) / len(batch_clipped_ratios) if batch_clipped_ratios else 0.0
        # Compute per-bucket average clipped ratio for the episode
        clipped_ratio_per_bucket = [sum(ratios) / len(ratios) if ratios else 0.0 for ratios in clipped_ratios_per_bucket]
        
        
        # REINFORCE update once per episode
        teacher_loss = None
        returns = discounted_returns(trajectory_rewards, gamma=gamma)
        
        print("Trajectory rewards:", trajectory_rewards)
        print("Discounted returns:", returns)
        print_gpu_memory(tag="before teacher update")

        
        if use_teacher:
            teacher_loss = reinforce_teacher_update(
                optimizer=teacher_optimizer,
                log_probs=trajectory_log_probs,
                rewards=trajectory_rewards,
                gamma=gamma,
                use_advantage_baseline=use_advantage_baseline,
                baseline_type=baseline_type,
                normalize_advantage=normalize_advantage,
            )
            final_obs = build_teacher_observation( # why final? - answer: We want to see what the teacher's observation looks like at the end of the episode, after all the training steps and updates have been done. This final observation can give us insight into what information the teacher had access to when it made its last decision, and how that might have influenced its policy update. By looking at the final observation, we can also analyze how the bucket statistics evolved over the course of the episode and how that might have impacted the teacher's learning and decision-making process. It's a way to introspect on the teacher's perspective at the conclusion of the episode.
                step_t=T - 1,
                stats=stats,
                K=num_buckets,
                T=T,
                device=device,
            )
    
            updated_probs = teacher_policy_probs( # after we perform the REINFORCE update on the teacher model, we want to compute the new action probabilities (teacher_probs) based on the final observation. This allows us to see how the teacher's policy has changed as a result of the update. By passing the final observation through the updated teacher model, we can get the new distribution over buckets that the teacher would use if it were to sample an action again at this point. This is useful for understanding how the teacher's learning has progressed and how its curriculum strategy has evolved by the end of the episode.
                teacher_model=teacher_model, 
                observation=final_obs,
            ).detach().cpu().tolist()

            # note = The final_obs represents the teacher's observation at the end of the episode, which includes the most up-to-date information about the student's performance and the bucket statistics after all training steps have been completed. By computing the updated_probs based on this final observation, we can see how the teacher's policy would respond to the current state of the student and the buckets after learning has occurred. This gives us insight into how the teacher has adapted its curriculum strategy in response to the student's progress throughout the episode. If we were to compute updated_probs based on an earlier observation, it might not reflect the full impact of the learning that took place during the episode, so using the final observation allows us to capture the cumulative effect of all updates and training steps on the teacher's policy.

            print("\n=========== TEACHER UPDATE once at end of episode ===========")
            print(f"Teacher loss: {teacher_loss:.6f}")
            print ("Final teacher observation (o_T):", final_obs)
            print ("Updated teacher bucket probabilities after REINFORCE update:", updated_probs)
        else:
            updated_probs = p_ref[:] # what does this do? -> In the baseline case where use_teacher=False, we don't have a teacher model or policy updates. So we just set updated_probs to be the reference uniform distribution (p_ref) that we used for sampling during the episode. This is just for logging purposes, to show that in the baseline run, the "updated" probabilities remain the same as the uniform distribution since there is no learning happening on a teacher model.   
            print (f"Baseline run -> no teacher update, probabilities remain uniform: {updated_probs}")

        
        # [ TO DIVE INTO IT DEEPLY LATER ]

        ########## to know how QUICKLY the student model improved during training → whether the teacher's curriculum decisions led to FASTER convergence against baseline.  ##########
        # find the smallest loss in the per_ep_val_loss_curve (which has T losses -> 1 per step) -> and save that step as the convergence step 
        # note -> in this context convergence meanas -> there is no loss that is lower
        convergence_step = 1 + min(range(len(per_ep_val_loss_curve)), key=lambda i: per_ep_val_loss_curve[i]) # this line finds the index of the step where the validation loss was lowest, which we can interpret as the point of convergence or best performance during training. By adding 1, we convert from a 0-based index to a 1-based step number for easier interpretation (e.g., "convergence at step 5" instead of "convergence at index 4"). This metric can help us understand how quickly the student model improved during training and whether the teacher's curriculum decisions led to faster convergence compared to the baseline.
        print (f"\nConvergence step (lowest validation loss): Step {convergence_step} with val_loss {per_ep_val_loss_curve[convergence_step - 1]:.6f}")

        # Calculate global_step_end: the final step of this episode across all episodes
        global_step_end = episode_id * T + (T - 1)  # 0-indexed, so last step is T-1

        if enable_wandb and wandb is not None and wandb.run is not None:  # log final episode metrics and curves to wandb
            # Log final episode summary at the global step end
            wandb.log({
                # "episode/id": episode_id,
                "final_ep_test/loss": final_ep_test_loss,
                "final_ep_test/acc": final_ep_test_acc,
                "episode/convergence_step": convergence_step,
            }, step=global_step_end)  # log at the final global step of this episode
            # LATER THINK OF LOGGING ALSO ->             "bucket_0_prob": action[0], "bucket_1_prob": action[1],... 

        return {
            "method": "curriculum" if use_teacher else "baseline",
            "seed": seed,
            "episode_id": episode_id,
            "T": T,  # Track steps per episode for global_step calculations
            "per_ep_val_loss_curve": per_ep_val_loss_curve,
            "per_ep_val_acc_curve": per_ep_val_acc_curve,
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
            "general_clipped_ratio": general_clipped_ratio,  # Add general clipped ratio to output
            "batch_clipped_ratios": batch_clipped_ratios,    # Add batch clipped ratios to output
            "clipped_ratio_per_bucket": clipped_ratio_per_bucket,  # Add per-bucket ratios
            "global_step_end": global_step_end,  # Final global step of this episode
        }
    finally: # try-finally because No matter what happens (error or success), the finally block always executes and cleans up temporary objects to free up memory.
        # Note: student_model, student_tokenizer, student_optimizer, teacher_model, and teacher_optimizer
        # are passed in as persistent across episodes and are managed by run_final_evaluation, so we don't delete them here.
        clear_device_memory()


def summarize_runs(label, runs): # label is whether it's curriculum or baseline run 
    print(f"\n===== {label} Summary of across all seeds =====")
    print(f"Runs (total number of episodes across all seeds): {len(runs)}")
    print(f"Avg final held-out test loss: {mean([r['final_ep_test_loss'] for r in runs]):.6f}")
    print(f"Avg final held-out test accuracy: {mean([r['final_ep_test_acc'] for r in runs]):.6f}")
    print(f"Avg convergence step: {mean([r['convergence_step'] for r in runs]):.2f}") # what does this do? -> This line calculates the average convergence step across all runs. The convergence step is defined as the step at which the validation loss was lowest during training. By taking the mean of these convergence steps across multiple runs, we can get an overall sense of how quickly the student model tended to reach its best performance on the validation set under the given training conditions (curriculum vs baseline). A lower average convergence step would suggest that, on average, the student model reached its optimal performance earlier in the training process, which could indicate more efficient learning potentially due to the teacher's curriculum guidance.
    print(f"Avg step time (seconds): {mean([r['avg_step_time'] for r in runs]):.2f}") # what does this do? -> This line calculates the average step time in seconds across all runs. The step time is the wall-clock duration it took to complete each training step during the episode. By averaging this metric across multiple runs, we can get an overall sense of the computational efficiency of the training process under the given conditions (curriculum vs baseline). If one method consistently has a lower average step time, it could indicate that it is more computationally efficient, which is an important factor to consider alongside final performance metrics like test loss and accuracy when evaluating the effectiveness of the curriculum learning approach. 

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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    m_val = clamp_batch_size(len(D_val_curve), val_batch_size) # basically m_val = val_batch_size 
    m_test = clamp_batch_size(len(D_test), test_batch_size)
    
    if device == "mps":
        m_val = min(m_val, 4)
        m_test = min(m_test, 4)

    
    # SAMPLE A FIXED validation batch and a FIXED test batch # 
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

    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    
    # ####### DEFINE STUDENT ####### # 
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=dtype,
    ).to(device)
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=student_lr)

    # ####### DEFINE TEACHER ####### #
    teacher_model = init_teacher_model(
        K=len(buckets),
        device=device,
        hidden_dim=teacher_hidden_dim,
    )
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=teacher_lr)

    ##

    print(f"Loaded student model '{student_model_name}' on device '{device}' with dtype '{dtype}'.")
    print(f"Student optimizer: AdamW with lr={student_lr}.")
    print(f"Validation curve batch size: {len(fixed_val_curve_batch)}")
    print(f"Fixed test batch size: {len(fixed_test_batch)}")
    print(f"Initialized teacher model with {sum(p.numel() for p in teacher_model.parameters())} parameters.")

    return {
        "D_tr": D_tr,
        "D_val_curve": D_val_curve,
        "D_test": D_test,
        "buckets": buckets,
        "fixed_val_curve_batch": fixed_val_curve_batch,
        "fixed_test_batch": fixed_test_batch,
        "student_model": student_model,
        "student_tokenizer": student_tokenizer,
        "student_optimizer": student_optimizer,
        "teacher_model": teacher_model,
        "teacher_optimizer": teacher_optimizer,
    }

def run_final_evaluation(
    method,
    seed,
    episodes,
    T,
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
    wandb_run_name,
):
    clear_device_memory()
    device = resolve_device(device)
    use_teacher = method == "curriculum"

    import datetime
    run_start_time = datetime.datetime.now()
    run_name = wandb_run_name or f"{method}_seed_{seed}"

    print("\n========== RUN CONFIGURATION ==========")
    print(f"[RUN START] {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"method: {method}")
    print(f"seed: {seed}")
    print(f"episodes: {episodes}")
    print(f"device: {device}")
    print(f"enable_wandb: {enable_wandb}")
    print(f"wandb_project: {wandb_project}")
    print(f"wandb_entity: {wandb_entity}")
    print(f"wandb_run_name: {run_name}")
    print(f"steps (T): {T}")
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
                "method": method,
                "seed": seed,
                "episodes": episodes,
                "steps": T,
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

    method_runs = []
    for ep in range(episodes):
        print(f"\n\n######## Seed={seed} Episode={ep} | {method} ########\n")
        episode_run = run_episode(
            D_tr=experiment["D_tr"],
            D_val_curve=experiment["D_val_curve"],
            D_test=experiment["D_test"],
            buckets=experiment["buckets"],
            T=T,
            B=B,
            student_model=experiment["student_model"],
            student_tokenizer=experiment["student_tokenizer"],
            student_optimizer=experiment["student_optimizer"],
            teacher_model=experiment["teacher_model"],
            teacher_optimizer=experiment["teacher_optimizer"],
            val_batch_size=val_batch_size,
            seed=seed,
            episode_id=ep,
            global_val_curve_batch=experiment["fixed_val_curve_batch"],
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
            use_teacher=use_teacher,
            enable_wandb=enable_wandb,
        )
        method_runs.append(episode_run)
        clear_device_memory()

    summarize_runs(method, method_runs)

    run_end_time = datetime.datetime.now()
    run_duration = run_end_time - run_start_time
    print("\n========== RUN END ==========")
    print(f"[RUN END] {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run duration: {str(run_duration).split('.')[0]}")
    print("============================\n")

    if enable_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()
    return {"method": method, "runs": method_runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning with REINFORCE Teacher - Fully configurable from CLI")
    
    # === EXPERIMENT & REPRODUCIBILITY ===
    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility")
    parser.add_argument("--method", type=str, choices=["curriculum", "baseline"], required=True, help="Training method to run: curriculum uses the REINFORCE teacher; baseline uses uniform bucket sampling.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per seed")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"], help="Compute device")
    
    # === TRAINING LOOP ===
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps per episode (T)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (B)")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Test batch size. If None, use val_batch_size.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    
    # === STUDENT MODEL ===
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B", help="Student model identifier (HuggingFace)")
    parser.add_argument("--student_lr", type=float, default=1e-5, help="Student optimizer learning rate (AdamW)")
    
    # === TEACHER MODEL ===
    parser.add_argument("--teacher_lr", type=float, default=1e-3, help="Teacher optimizer learning rate (Adam)")
    parser.add_argument("--teacher_hidden_dim", type=int, default=32, help="Hidden dimension of teacher MLP")
    
    # === TEACHER POLICY (REINFORCE) ===
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for REINFORCE")
    parser.add_argument("--lambda_cost", type=float, default=0.0, help="Weight on compute-cost penalty in reward")
    parser.add_argument("--alpha_stability", type=float, default=0.0, help="Weight on stability penalty (KL from uniform)")
    
    # === REINFORCE BASELINE ===
    parser.add_argument("--use_advantage_baseline", type=int, default=1, help="Whether to use baseline in advantage computation (0 or 1)")
    parser.add_argument("--baseline_type", type=str, default="mean_return", help="Type of baseline: 'mean_return' or other")
    parser.add_argument("--normalize_advantage", type=int, default=1, help="Whether to normalize advantages (0 or 1)")
    
    # === WANDB LOGGING ===
    parser.add_argument("--wandb", type=int, default=0, help="Enable wandb logging (0 or 1)")
    parser.add_argument("--wandb_project", type=str, default="cl-rl", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B team/entity name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B custom run name (optional)")
    
    args = parser.parse_args()

    run_final_evaluation(
        method=args.method,
        seed=args.seed,
        episodes=args.episodes,
        T=args.steps,
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
        max_length=args.max_length, # PLAY AROUND with diff. values - TO DIVE DEEPER LATER -> how does max_length compare with the max_length of Qwen itself and what happens when our max_length is smaller than the Qwen's max_length? -> The max_length parameter in our training loop defines the maximum sequence length that we will use for tokenization and input to the student model during training. If our max_length is smaller than the model's inherent max_length, it means we are truncating the input sequences to a shorter length than what the model can technically handle. This can help reduce memory usage and speed up training, especially if our dataset contains long sequences that exceed our chosen max_length. However, it also means that we may lose some information from the input if important tokens are truncated away. It's a trade-off between computational efficiency and retaining full context from the input data. We should experiment with different max_length values to find a good balance for our specific dataset and model.
        device=args.device,
        enable_wandb=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

# in WANBD interface: 
    # check TABLE (very helfpul to see ALL runs at once)
    # COMPARE ACROSS GROUPS 
 