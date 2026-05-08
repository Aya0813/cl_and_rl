
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

def log_comparison_to_wandb(curriculum_run, baseline_run, seed, episode_id):
    """
    Log both curriculum and baseline validation curves to the same wandb run for direct comparison.
    """
    if wandb is None or wandb.run is None:
        return
    steps = list(range(1, len(curriculum_run["val_loss_curve"]) + 1))
    # Validation loss
    wandb.log({
        f"val_loss_comparison/seed_{seed}_ep_{episode_id}": wandb.plot.line_series(
            xs=steps,
            ys=[curriculum_run["val_loss_curve"], baseline_run["val_loss_curve"]],
            keys=["curriculum", "baseline"],
            title=f"Validation Loss (Seed {seed} Ep {episode_id})",
            xname="Step"
        )
    })
    # Validation accuracy
    wandb.log({
        f"val_acc_comparison/seed_{seed}_ep_{episode_id}": wandb.plot.line_series(
            xs=steps,
            ys=[curriculum_run["val_acc_curve"], baseline_run["val_acc_curve"]],
            keys=["curriculum", "baseline"],
            title=f"Validation Accuracy (Seed {seed} Ep {episode_id})",
            xname="Step"
        )
    })


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
        input_ids, attention_mask, labels = build_solution_only_batch(
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

    T,                     # Number of training steps (from --steps CLI arg)
    B,                     # Training batch size (from --batch_size CLI arg)
    
    # ===== STUDENT MODEL CONFIGURATION (not CLI args) =====
    student_model_name,    # Model name (hardcoded in run_final_evaluation)
    student_lr,            # Student learning rate (hardcoded: 1e-5)
    max_length,            # Max sequence length (hardcoded: 512)
    
    # ===== TEACHER MODEL CONFIGURATION (CLI args) =====
    teacher_lr,            # Teacher learning rate (hardcoded: 1e-2)
    teacher_hidden_dim,    # [CLI arg] Hidden dimension of teacher MLP (type: int, default=32) -> Start small (e.g., 32), then tune (16/32/64/128).
    
    # ===== TEACHER POLICY PARAMETERS (CLI args) =====
    gamma,                 # [CLI arg] Discount factor for REINFORCE (type: float, default=0.99) - determines weight of future vs immediate rewards
    lambda_cost,           # [CLI arg] Weight on compute-cost penalty in reward (type: float, default=0.0)
    alpha_stability,       # [CLI arg] Weight on stability penalty (KL div from uniform) (type: float, default=0.0)
    
    # ===== REINFORCE BASELINE CONFIGURATION (CLI args) =====
    use_baseline,          # [CLI arg] Whether to use baseline in advantage computation (type: bool via int, default=1)
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
    
    random.seed(seed)
    torch.manual_seed(seed) 
    clear_device_memory() 

    dtype = torch.float16 if device == "mps" else torch.float32 
    student_tokenizer = None
    student_model = None
    student_optimizer = None
    teacher_model = None
    teacher_optimizer = None

    try:
        student_tokenizer   = AutoTokenizer.from_pretrained(student_model_name)
        student_model       = AutoModelForCausalLM.from_pretrained( # Explaining further what does AutoModelForCausalLM do: It's a class from the Hugging Face Transformers library that provides a convenient way to load pre-trained language models that are designed for causal language modeling tasks. Causal language modeling is a type of language modeling where the model is trained to predict the next token in a sequence given all the previous tokens. This is in contrast to masked language modeling (like BERT), where the model predicts masked tokens based on their context. By using AutoModelForCausalLM, you can easily load a variety of pre-trained models (like GPT-2, GPT-3, etc.) that are suitable for generating text in an autoregressive manner, which is what we want for our student model in this curriculum learning setup. The student model will take a problem as input and generate a solution, so we need a causal language model architecture for that.)
            student_model_name, torch_dtype=dtype
        ).to(device)
        student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=student_lr)  # POST POC: play around with optimizer

        num_buckets = len(buckets)
        # p_ref is the reference (uniform) probability distribution over buckets. It's used as a baseline for the teacher's policy.
        p_ref = init_bucket_probs(num_buckets) # creates a uniform distribution—each bucket gets equal probability 1/num_buckets. This represents "no curriculum learning," just random sampling. -> p_ref defines what "neutral/no-curriculum" looks like, and it's used to regularize the teacher's learned policy.
        stats = None

        if use_teacher:
            teacher_model = init_teacher_model(
                K=num_buckets,
                device=device,
                hidden_dim=teacher_hidden_dim, 
            )
            teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=teacher_lr)

        val_loss_curve = []
        val_acc_curve = []
        reward_history = []
        teacher_probs_history = []
        step_time_history = []

        trajectory_rewards = []
        trajectory_log_probs = []

        # define validation batch sizes for both val and test 
        m_val = clamp_batch_size(len(D_val_curve), val_batch_size) # clamp val batch size to not exceed dataset size, with a warning if clamping occurs
        m_test = clamp_batch_size(len(D_test), val_batch_size)
        # technically m_val and m_test are not necessarily the same as val_batch_size if the datasets are smaller than val_batch_size, but for simplicity we use the same val_batch_size argument for both validation and test batch sizes. In a more refined implementation, we might want to have separate CLI arguments for val_batch_size and test_batch_size to allow more flexibility in case the validation and test datasets have different sizes or if we want to use different batch sizes for them.
        
        if device == "mps": # to deal with it later - change cause not precise 
            m_val = min(m_val, 4)
            m_test = min(m_test, 4)

        print(f"Validation batch size: {m_val}")
        print(f"Held-out batch size: {m_test}")

        fixed_val_batch = sample_uniform_batch(D_val_curve, m=m_val, seed=seed + 10_000 + episode_id) # samples once at the beginning and stays fixed for the entire episode—that's used for the validation curve you plot.


        for t in range(T):
            step_start = time.perf_counter()
            print(f"\n=== Step {t + 1}/{T} | {'Curriculum' if use_teacher else 'Baseline'} ===")

            # 1) Construct observation
            o_t = build_teacher_observation(
                step_t=t,
                stats=stats,
                K=num_buckets,
                T=T,
                device=device,
            )

            # 2) Teacher chooses action a_t (distribution over buckets)
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

            # 3-4) Sample training batch from buckets via categorical(a_t)
            training_batch, sampled_bucket_indices = sample_batch_from_buckets(
                buckets=buckets,
                a_t=a_t,
                B=B,
                seed=seed + (episode_id * 1000) + t, # ensure different sampling each step and episode (note that it is t dependant - unlike test sampling)
            )

            # 5) Sample one validation batch from fixed D_val -> AT EACH STEP
            val_batch = sample_uniform_batch(D_val_curve, m=m_val, seed=seed + 20_000 + t)

            # 6) Validation loss BEFORE learner update
            val_loss_before, _ = evaluate_loss_accuracy( 
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch, # <-- Uses the sampled val_batch
                device=device,
                max_length=max_length, 
                    # do we need max_length here if we're just computing losses? - yes because the model's forward pass might still require it, even if we're not doing any generation. 
                    # It's safer to include it to ensure that the input tensors are properly shaped and that we don't run into unexpected issues with sequence lengths during evaluation.
            )

            # 7-8) Train learner one SFT step
            _, train_loss = student_train_step(
                model=student_model,
                tokenizer=student_tokenizer,
                optimizer=student_optimizer,
                batch_examples=training_batch,
                device=device,
                max_length=max_length,
            )

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
            
            per_example_losses = compute_per_example_losses(
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=training_batch,
                device=device,
                max_length=max_length, 
            )

            stats = update_bucket_loss_stats(
                batch_examples=training_batch,
                bucket_labels=sampled_bucket_indices,
                losses=per_example_losses,  # losses for each training example (one loss per example)
                stats=stats,
                alpha=0.3,  # smoothing factor for exponential moving average (0.3 = 30% new info, 70% old info)
            )

            # 10) Validation loss AFTER update on SAME val batch
            val_loss_after, _ = evaluate_loss_accuracy( # note -> this is for the reward signal and reflects immediate improvement on a random sample (which can fluctuate).
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=val_batch, # <-- SAME val_batch (not resampled)
                device=device,
                max_length=max_length,
            )
            # This computes the validation loss on a freshly sampled validation batch (val_batch) right after the student model has been updated in the current step.
            # This value is used to compute the reward for the teacher (i.e., how much the validation loss improved after the update).


            ####### 

            # 11) Reward
            validation_improvement = val_loss_before - val_loss_after
            compute_cost = compute_compute_cost(training_batch)
            stability_penalty = compute_stability_penalty_kl(a_t, p_ref)
            reward = validation_improvement - lambda_cost * compute_cost - alpha_stability * stability_penalty

            trajectory_rewards.append(reward)
            reward_history.append(reward)
            teacher_probs_history.append(teacher_probs)

            ####### 

            print("\n================ STEP ================")
            print("Step t:", t)

            print("Observation o_t:")
            print(o_t.detach().cpu().tolist())

            print("\nTeacher mean probs:")
            print(teacher_probs)

            print("\nSampled Dirichlet action a_t:")
            print(a_t)

            print("\nBucket counts:") # VERY crutial to see BECAUSE -> If the batch composition barely changes, then the curriculum policy is not really affecting training.
            print({
                k: sampled_bucket_indices.count(k)
                for k in range(num_buckets)
            })

            print("\nLog prob:")
            if log_prob_t is not None:
                print(float(log_prob_t.detach().cpu()))
            else:
                print("log_prob_t is None (baseline run)") 

            print("\nValidation improvement:")
            print(validation_improvement)

            print("\nCompute cost:")
            print(compute_cost)

            print("\nStability penalty:")
            print(stability_penalty)

            print("\nFinal reward:")
            print(reward)

            ####### 

            # Validation curve on fixed batch
            curve_loss, curve_acc = evaluate_loss_accuracy( # note -> this is for plotting a smooth, comparable learning curve across steps and episodes, since it always uses the same data.
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                batch_examples=fixed_val_batch, # We want to track the validation loss and accuracy over time on a consistent set of examples to see how the student's performance evolves throughout the episode. By using a fixed validation batch (sampled once at the beginning of the episode), we can plot a meaningful validation curve that reflects the student's learning progress on the same data points, rather than having the validation metrics fluctuate due to different random samples at each step. This allows us to better visualize the impact of the teacher's curriculum decisions on the student's learning trajectory.
                device=device,
                max_length=max_length,
            )
            # This computes the validation loss and accuracy on a fixed validation batch (fixed_val_batch) that was sampled once at the start of the episode and stays the same for all steps.
            # This value is used to build a consistent validation curve over the episode, so you can plot how the model improves on the same set of examples over time.

            
            # val_loss_after → reward calculation, fresh batch each step
            # curve_loss → learning curve plotting, fixed batch for consistency


            val_loss_curve.append(curve_loss)
            val_acc_curve.append(curve_acc)

            ####### 

            step_time = time.perf_counter() - step_start
            step_time_history.append(step_time)

            print(f"Train loss: {train_loss:.6f}")
            print(f"Val loss before: {val_loss_before:.6f}")
            print(f"Val loss after: {val_loss_after:.6f}")
            print(f"Reward: {reward:.6f}")
            print(f"Step time: {step_time:.2f}s")
            print(f"Teacher bucket probs: {teacher_probs}")
            if enable_wandb and wandb is not None and wandb.run is not None:  # log step metrics only when wandb is active
                wandb.log({  # send one metrics payload per step
                    "step": t + 1,  # 1-based step index for readability
                    "method": "curriculum" if use_teacher else "baseline",  # tag run branch in logs
                    "seed": seed,  # track random seed in each record
                    "episode_id": episode_id,  # track episode index in each record
                    "train_loss": train_loss,  # student train loss at this step
                    "val_loss_before": val_loss_before,  # validation loss before update
                    "val_loss_after": val_loss_after,  # validation loss after update
                    "reward": reward,  # scalar reward used by teacher
                    "curve_loss": curve_loss,  # fixed-batch validation loss curve value
                    "curve_acc": curve_acc,  # fixed-batch validation accuracy curve value
                    "step_time_sec": step_time,  # wall-clock step duration
                })  # close step-level wandb payload

            clear_device_memory()

        ####### ####### ####### episode done ####### ####### ####### 
        
        # REINFORCE update once per episode
        teacher_loss = None
        returns = discounted_returns(trajectory_rewards, gamma=gamma)
        
        
        
        
        if use_teacher:
            teacher_loss = reinforce_teacher_update(
                optimizer=teacher_optimizer,
                log_probs=trajectory_log_probs,
                rewards=trajectory_rewards,
                gamma=gamma,
                use_baseline=use_baseline,
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

            print(f"\nTeacher updated once at end of episode. Teacher loss: {teacher_loss:.6f}")
            print(f"Updated teacher bucket probabilities: {updated_probs}")
        else:
            updated_probs = p_ref[:]

        # Sampled ONCE at the end of the entire episode (after all T steps)
        test_batch = sample_uniform_batch(D_test, m=m_test, seed=seed + 30_000 + episode_id)
        final_test_loss, final_test_acc = evaluate_loss_accuracy(
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            batch_examples=test_batch,  # <-- Single test evaluation
            device=device,
            max_length=max_length,
        ) 
        
        # difference between val and test
            # Val batch during training: Resampled fresh every step (seed + 20_000 + t changes each step)
            # Test batch at the end: Sampled once after episode is complete (fixed seed)
        
        # HOW TO INTERPRET Validation VS test metrics:  
            # The validation loss and accuracy that we track at each step during the episode are meant to provide a real-time signal of the student's performance and learning progress on a consistent set of examples (the fixed_val_batch). This allows us to see how the student's performance evolves over time in response to the teacher's curriculum decisions. The validation metrics can fluctuate from step to step based on the training updates and the teacher's actions, and they serve as a feedback signal for the teacher's policy learning.
            # The final test loss and accuracy are meant to provide an overall evaluation of the student's performance on held-out data after the entire curriculum learning process has been completed. These metrics are computed on a separate test set that was not used during training or for the validation curve, so they serve as an unbiased assessment of how well the student has learned to solve the problems after being guided by the teacher's curriculum. When interpreting these final test metrics, you can compare them between the curriculum and baseline runs to see if the teacher's guidance led to better generalization performance on unseen data. Additionally, you can analyze how the final test performance correlates with the validation curves and the teacher's actions throughout the episode to gain insights into which curriculum strategies were most effective in improving the student's learning outcomes.
    
        
        print("\n====================================")
        print("END OF EPISODE")
        print("====================================")

        print("Trajectory rewards:")
        print(trajectory_rewards)

        print("\nDiscounted returns:")
        print(returns)

        # print("\nAdvantages:")
        # print(advantages)


        
        # [ TO DIVE INTO IT DEEPLY LATER ]
        convergence_step = 1 + min(range(len(val_loss_curve)), key=lambda i: val_loss_curve[i]) # this line finds the index of the step where the validation loss was lowest, which we can interpret as the point of convergence or best performance during training. By adding 1, we convert from a 0-based index to a 1-based step number for easier interpretation (e.g., "convergence at step 5" instead of "convergence at index 4"). This metric can help us understand how quickly the student model improved during training and whether the teacher's curriculum decisions led to faster convergence compared to the baseline.
        if enable_wandb and wandb is not None and wandb.run is not None:  # log final episode metrics and curves to wandb
            # Log final episode summary
            wandb.log({
                "method": "curriculum" if use_teacher else "baseline",  # tag run branch in logs
                "seed": seed,  # include seed for grouping
                "episode_id": episode_id,  # include episode identifier
                "final_test_loss": final_test_loss,  # held-out final test loss
                "final_test_acc": final_test_acc,  # held-out final test accuracy
                "convergence_step": convergence_step,  # best validation-loss step index
                "avg_step_time": mean(step_time_history),  # average per-step runtime
            })

            # Log validation loss and accuracy curves as wandb plots
            # These will appear as line plots in the wandb dashboard, one for each run (curriculum/baseline)
            wandb.log({
                "val_loss_curve": wandb.plot.line_series(
                    xs=list(range(1, len(val_loss_curve) + 1)),
                    ys=[val_loss_curve],
                    keys=["val_loss"],
                    title=f"Validation Loss Curve ({'curriculum' if use_teacher else 'baseline'})",
                    xname="Step"
                ),
                "val_acc_curve": wandb.plot.line_series(
                    xs=list(range(1, len(val_acc_curve) + 1)),
                    ys=[val_acc_curve],
                    keys=["val_acc"],
                    title=f"Validation Accuracy Curve ({'curriculum' if use_teacher else 'baseline'})",
                    xname="Step"
                ),
            })

            # Log teacher's probability distribution over time (only for curriculum runs)
            if use_teacher and teacher_probs_history and len(teacher_probs_history) > 0:
                # teacher_probs_history: list of lists, shape [steps, num_buckets]
                num_buckets = len(teacher_probs_history[0])
                xs = list(range(1, len(teacher_probs_history) + 1))
                ys = list(zip(*teacher_probs_history))  # shape: [num_buckets][steps]
                keys = [f"bucket_{i}" for i in range(num_buckets)]
                wandb.log({
                    "teacher_probs_curve": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=keys,
                        title="Teacher Bucket Probabilities (curriculum)",
                        xname="Step"
                    )
                })

        return {
            "method": "curriculum" if use_teacher else "baseline",
            "seed": seed,
            "episode_id": episode_id,
            "val_loss_curve": val_loss_curve,
            "val_acc_curve": val_acc_curve,
            "reward_history": reward_history,
            "teacher_probs_history": teacher_probs_history,
            "step_time_history": step_time_history,
            "avg_step_time": mean(step_time_history),
            "returns": returns,
            "teacher_loss": teacher_loss,
            "updated_teacher_probs": updated_probs,
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
            "convergence_step": convergence_step,
            "bucket_stats": stats,
        }
    finally: # try-finally because No matter what happens (error or success), the finally block always executes and cleans up the models and optimizers to free up memory, which is especially important when running multiple episodes in a loop.
        if teacher_optimizer is not None:
            del teacher_optimizer # del -> delete 
        if teacher_model is not None:
            del teacher_model
        if student_optimizer is not None:
            del student_optimizer
        if student_model is not None:
            del student_model
        if student_tokenizer is not None:
            del student_tokenizer
        clear_device_memory()


def summarize_runs(label, runs):
    print(f"\n===== {label} Summary =====")
    print(f"Runs: {len(runs)}")
    print(f"Avg final held-out test loss: {mean([r['final_test_loss'] for r in runs]):.6f}")
    print(f"Avg final held-out test accuracy: {mean([r['final_test_acc'] for r in runs]):.6f}")
    print(f"Avg convergence step: {mean([r['convergence_step'] for r in runs]):.2f}")
    print(f"Avg step time (seconds): {mean([r['avg_step_time'] for r in runs]):.2f}")


def run_final_evaluation(
    seeds=(11, 22, 33),
    episodes=1,
    T=20,
    B=8,
    val_batch_size=8,
    student_model_name="Qwen/Qwen3.5-0.8B",
    student_lr=1e-5,
    teacher_lr=1e-2,
    teacher_hidden_dim=32,
    gamma=0.99,
    lambda_cost=0.0,
    alpha_stability=0.0,
    use_baseline=True,
    baseline_type="mean_return", # "mean_return" = use the average of all returns in the episode as the baseline value. This helps reduce variance by centering the advantages around zero, which can lead to more stable and efficient learning. The teacher's policy is updated based on how much better or worse each action's return was compared to this average baseline.
    normalize_advantage=True,
    max_length=512,
    device="auto",
    enable_wandb=False,
    wandb_project="cl-rl",
    wandb_entity=None,
    wandb_run_name=None,
):
    clear_device_memory()
    device = resolve_device(device)

    import datetime
    run_start_time = datetime.datetime.now()
    print("\n========== RUN CONFIGURATION ==========")
    print(f"[RUN START] {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n# === GENERAL RUN SETUP ===")
    print(f"seeds: {seeds}")
    print(f"episodes: {episodes}")
    print(f"device: {device}")
    print(f"enable_wandb: {enable_wandb}")
    print(f"wandb_project: {wandb_project}")
    print(f"wandb_entity: {wandb_entity}")
    print(f"wandb_run_name: {wandb_run_name}")
    print("\n# === STUDENT MODEL CONFIGURATION ===")
    print(f"student_model_name: {student_model_name}")
    print(f"student_lr: {student_lr}")
    print(f"max_length: {max_length}")
    print(f"batch_size (B): {B}")
    print(f"val_batch_size: {val_batch_size}")
    print("\n# === TEACHER MODEL CONFIGURATION ===")
    print(f"teacher_lr: {teacher_lr}")
    print(f"teacher_hidden_dim: {teacher_hidden_dim}")
    print("\n# === TEACHER POLICY PARAMETERS ===")
    print(f"gamma: {gamma}")
    print(f"lambda_cost: {lambda_cost}")
    print(f"alpha_stability: {alpha_stability}")
    print("\n# === REINFORCE BASELINE CONFIGURATION ===")
    print(f"use_baseline: {use_baseline}")
    print(f"baseline_type: {baseline_type}")
    print(f"normalize_advantage: {normalize_advantage}")
    print("\n# === TRAINING LOOP ===")
    print(f"steps (T): {T}")
    print("=======================================\n")
    print(f"Running on device: {device}")
    if enable_wandb and wandb is None:  # fail fast if user requested wandb without dependency
        raise ImportError("wandb is not installed. Please run: pip install wandb")  # clear install guidance
    if enable_wandb and wandb is not None:  # initialize wandb once for this execution
        wandb.init(  # create a new wandb run
            project=wandb_project,  # set wandb project name
            entity=wandb_entity,  # optional wandb team/user
            name=wandb_run_name,  # optional custom run title
            config={  # store key hyperparameters in run config
                "steps": T,  # number of train steps per episode
                "batch_size": B,  # train batch size
                "val_batch_size": val_batch_size,  # validation batch size
                "student_model_name": student_model_name,  # student model identifier
                "student_lr": student_lr,  # student optimizer learning rate
                "teacher_lr": teacher_lr,  # teacher optimizer learning rate
                "teacher_hidden_dim": teacher_hidden_dim,  # teacher MLP width
                "gamma": gamma,  # return discount factor
                "lambda_cost": lambda_cost,  # compute-cost penalty weight
                "alpha_stability": alpha_stability,  # stability penalty weight
                "use_baseline": use_baseline,  # whether baseline is used in reinforce
                "baseline_type": baseline_type,  # baseline strategy name
                "normalize_advantage": normalize_advantage,  # advantage normalization toggle
                "max_length": max_length,  # sequence truncation cap
                "device": device,  # resolved runtime device
            },  # close config payload
        )  # finish wandb initialization

    # Keep MPS safer by default 
    # TO DIVE FURTHER LATER - does not make sense to have different max_length and batch_size for MPS vs other devices, but this is just a quick fix to avoid OOM errors on MPS during testing. Ideally we would implement proper dynamic batching and/or gradient accumulation to handle larger batches on MPS without OOM.
    if device == "mps":
        if B > 2:
            print(f"[MPS] Reducing B from {B} to 2 to avoid OOM.")
            B = 2
        if max_length > 256:
            print(f"[MPS] Reducing max_length from {max_length} to 256 to avoid OOM.")
            max_length = 256

    curriculum_runs = []
    baseline_runs = []

    for seed in seeds:
        # Split once per seed: 80% train / 10% val / 10% holdout
        D_tr, D_val_curve, D_test, buckets_raw = prepare_math_dataset( 
            seed=seed,
            train_ratio=0.8,        # training
            val_ratio=0.1,          # validation (used during training for reward)
            holdout_ratio=0.1,      # test (final evaluation)
        )

        # TO DIVE FURTHER LATER - cause there could be overlap between the val sampling 

        # note the distinction between val_ratio and val_batch_size
            # val_ratio      → defines how big the validation dataset is (how much data is reserved for validation)
            # val_batch_size → defines how many samples you use per step

        # buckets_raw is the original bucket structure coming from your dataset preprocessing
        # buckets_raw = {
        #     "easy":   [example1, example2, ...],
        #     "medium": [...],
        #     "hard":   [...]
        # } 
        # keys = semantic labels of buckets (e.g. easy/medium/hard)


        
        bucket_keys = sorted(buckets_raw.keys()) 
        buckets = {i: buckets_raw[k] for i, k in enumerate(bucket_keys)}
        print(f"Bucket keys: {bucket_keys}")
        print("Bucket sizes:")
        for i, k in enumerate(bucket_keys):
            print(f"  {i} ({k}): {len(buckets[i])} examples")
        

        # - `bucket_keys = ['1','2','3']`
        # - `buckets = {0: buckets_raw['1'], 1: buckets_raw['2'], 2: buckets_raw['3']}`
        # Teacher action `a_t[i]` corresponds directly to integer bucket `i`.


        for ep in range(episodes):
            print(f"\n\n######## Seed={seed} Episode={ep} | Curriculum ########")
            curriculum_run = run_episode(
                D_tr=D_tr,
                D_val_curve=D_val_curve,
                D_test=D_test,
                buckets=buckets,
                T=T,
                B=B,
                val_batch_size=val_batch_size,
                seed=seed,
                episode_id=ep,
                student_model_name=student_model_name,
                student_lr=student_lr,
                teacher_lr=teacher_lr,
                teacher_hidden_dim=teacher_hidden_dim, 
                gamma=gamma, 
                lambda_cost=lambda_cost, 
                alpha_stability=alpha_stability, 
                use_baseline=use_baseline,
                baseline_type=baseline_type,
                normalize_advantage=normalize_advantage,
                device=device,
                max_length=max_length,
                use_teacher=True, # MAIN difference between curriculum and baseline runs - whether teacher is used or not
                enable_wandb=enable_wandb,  # forward wandb toggle to episode runner
            )
            curriculum_runs.append(curriculum_run)

            print(f"\n\n######## Seed={seed} Episode={ep} | Baseline ########")
            baseline_run = run_episode(
                D_tr=D_tr,
                D_val_curve=D_val_curve,
                D_test=D_test,
                buckets=buckets,
                T=T,
                B=B,
                val_batch_size=val_batch_size, 
                seed=seed,
                episode_id=ep,
                student_model_name=student_model_name,
                student_lr=student_lr,
                teacher_lr=teacher_lr,
                teacher_hidden_dim=teacher_hidden_dim,
                gamma=gamma,
                lambda_cost=lambda_cost,
                alpha_stability=alpha_stability,
                use_baseline=use_baseline, 
                baseline_type=baseline_type,
                normalize_advantage=normalize_advantage,
                device=device,
                max_length=max_length,
                use_teacher=False, ## 
                enable_wandb=enable_wandb,  # forward wandb toggle to episode runner
            )
            baseline_runs.append(baseline_run)





            # Log both curves to wandb for direct comparison
            if enable_wandb and wandb is not None and wandb.run is not None:
                log_comparison_to_wandb(curriculum_run, baseline_run, seed, ep)

            clear_device_memory()

    summarize_runs("Curriculum (REINFORCE Teacher)", curriculum_runs)
    summarize_runs("Vanilla SFT Baseline", baseline_runs)
    run_end_time = datetime.datetime.now()
    run_duration = run_end_time - run_start_time
    print("\n========== RUN END ==========")
    print(f"[RUN END] {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run duration: {str(run_duration).split('.')[0]}")
    print("============================\n")
    if enable_wandb and wandb is not None and wandb.run is not None:  # close wandb run cleanly
        wandb.finish()  # flush metrics and finalize run
    return {"curriculum_runs": curriculum_runs, "baseline_runs": baseline_runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--teacher_hidden_dim", type=int, default=32)
    parser.add_argument("--lambda_cost", type=float, default=0.0)
    parser.add_argument("--alpha_stability", type=float, default=0.0)
    parser.add_argument("--use_baseline", type=int, default=1)
    parser.add_argument("--baseline_type", type=str, default="mean_return")
    parser.add_argument("--normalize_advantage", type=int, default=1)
    parser.add_argument("--wandb", type=int, default=0)  # set to 1 to enable wandb logging
    parser.add_argument("--wandb_project", type=str, default="cl-rl")  # wandb project name
    parser.add_argument("--wandb_entity", type=str, default=None)  # optional wandb account/team -> FOR ME IT WILL BE Omar and I's shared team name on wandb -> "CL-and-RL"
    parser.add_argument("--wandb_run_name", type=str, default=None)  # optional custom run name
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    run_final_evaluation(
        seeds=(11, 22, 33),
        episodes=args.episodes,
        T=args.steps,
        B=args.batch_size,
        val_batch_size=args.val_batch_size,
        student_model_name="Qwen/Qwen3.5-0.8B",
        student_lr=1e-5,
        teacher_lr=1e-3,
        teacher_hidden_dim=args.teacher_hidden_dim,
        gamma=args.gamma,
        lambda_cost=args.lambda_cost,
        alpha_stability=args.alpha_stability,
        use_baseline=bool(args.use_baseline),
        baseline_type=args.baseline_type,
        normalize_advantage=bool(args.normalize_advantage),
        max_length=512, # MAKE SUERE IDEALLY max_length is a bit less or equal to the model's max context length to avoid OOM, but for simplicity we keep it fixed at 512 for now since Qwen3.5-0.8B can handle that. (play around with different values!!)
        device=args.device,
        enable_wandb=bool(args.wandb),  # pass wandb on/off flag
        wandb_project=args.wandb_project,  # pass wandb project name
        wandb_entity=args.wandb_entity,  # pass optional wandb entity
        wandb_run_name=args.wandb_run_name,  # pass optional wandb run name
    )
