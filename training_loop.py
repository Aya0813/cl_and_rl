import gc
import random
import time
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bucket_stats import update_bucket_loss_stats
from dataset_prep import prepare_math_dataset, sample_batch_from_buckets
from student_step import (
    build_solution_only_batch,
    compute_reward,
    resolve_gpu_device,
    student_train_step,
    validation_batch_loss,
)
from teacher_policy import init_bucket_probs, sample_a_t, update_bucket_probs


def sample_uniform_batch(dataset_split, m, seed):
    rng = random.Random(seed)
    n = min(m, len(dataset_split))
    indices = rng.sample(range(len(dataset_split)), n)
    return [dataset_split[i] for i in indices]


def evaluate_loss_accuracy(model, tokenizer, batch_examples, device, max_length=1024):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, labels = build_solution_only_batch(
            batch_examples=batch_examples,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = float(outputs.loss.item())

        preds = outputs.logits.argmax(dim=-1)
        valid = labels != -100
        if valid.sum().item() == 0:
            acc = 0.0
        else:
            acc = float((preds[valid] == labels[valid]).float().mean().item())

    return loss, acc             


def compute_per_example_losses(model, tokenizer, batch_examples, device, max_length=1024):
    losses = []
    model.eval()
    with torch.no_grad():
        for ex in batch_examples:
            input_ids, attention_mask, labels = build_solution_only_batch(
                batch_examples=[ex],
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(float(outputs.loss.item()))
    return losses      


def run_training_loop(
    T=3,
    B=8,
    m=8,
    seed=42,
    model_name="Qwen/Qwen3.5-0.8B",
    lr=1e-5,
    max_length=512,
    use_teacher=True,
    episode_id=0,
):
    random.seed(seed)
    torch.manual_seed(seed)

    D_tr, D_val, buckets = prepare_math_dataset(seed=seed)
    print(f"D_tr size: {len(D_tr)}")
    print(f"D_val size: {len(D_val)}")
    print(f"Number of buckets K: {len(buckets)}")

    # Make a held-out test split from D_val for final evaluation.
    if len(D_val) >= 2:
        vt = D_val.train_test_split(test_size=0.5, seed=seed)
        D_val_curve = vt["train"]
        D_test = vt["test"]
    else:
        D_val_curve = D_val
        D_test = D_val

    device = resolve_gpu_device()
    dtype = torch.float16
    print(f"Device: {device}")

    if device == "mps":
        if B > 2:
            print(f"[MPS] Reducing B from {B} to 2 to avoid OOM.")
            B = 2
        if m > 4:
            print(f"[MPS] Reducing m from {m} to 4 to avoid OOM.")
            m = 4
        if max_length > 256:
            print(f"[MPS] Reducing max_length from {max_length} to 256 to avoid OOM.")
            max_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    K = len(buckets)
    bucket_probs = init_bucket_probs(K)
    uniform_probs = [1.0 / K] * K
    stats = None

    val_loss_curve = []
    val_acc_curve = []
    reward_history = []
    probs_history = []
    step_time_history = []

    fixed_val_batch = sample_uniform_batch(D_val_curve, m=m, seed=seed + 10_000 + episode_id)

    for t in range(T):
        step_start = time.perf_counter()
        print(f"\n=== Step {t + 1}/{T} | {'Curriculum' if use_teacher else 'Baseline'} ===")

        # 1) teacher outputs a_t
        a_t = sample_a_t(bucket_probs) if use_teacher else uniform_probs

        # 2) sample training batch using a_t
        train_batch, sampled_bucket_indices = sample_batch_from_buckets(
            buckets=buckets,
            a_t=a_t,
            B=B,
            seed=seed + (episode_id * 1000) + t,
        )

        # 3) sample validation batch
        # 4) compute val_loss_before
        val_batch, val_loss_before = validation_batch_loss(
            model=model,
            tokenizer=tokenizer,
            D_val=D_val_curve,
            m=m,
            device=device,
            max_length=max_length,
            seed=seed + 20_000 + t,
        )

        # 5) train student on batch
        _, train_loss = student_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            batch_examples=train_batch,
            device=device,
            max_length=max_length,
        )

        # 6) compute val_loss_after (same validation batch)
        _, val_loss_after = validation_batch_loss(
            model=model,
            tokenizer=tokenizer,
            D_val=D_val_curve,
            device=device,
            max_length=max_length,
            validation_batch=val_batch,
        )

        # 7) compute reward
        reward = compute_reward(
            val_loss_before=val_loss_before,
            val_loss_after=val_loss_after,
        )

        # 8) update teacher (curriculum only)
        if use_teacher:
            bucket_probs = update_bucket_probs(
                bucket_probs=bucket_probs,
                used_bucket_indices=sampled_bucket_indices,
                reward=reward,
                step_size=0.05,
            )
        else:
            print(f"Current bucket probabilities after update: {uniform_probs}")
            bucket_probs = uniform_probs

        # 9) update bucket statistics
        example_losses = compute_per_example_losses(
            model=model,
            tokenizer=tokenizer,
            batch_examples=train_batch,
            device=device,
            max_length=max_length,
        )
        stats = update_bucket_loss_stats(
            batch_examples=train_batch,
            bucket_labels=sampled_bucket_indices,
            losses=example_losses,
            stats=stats,
            alpha=0.3,
        )

        # Validation curve (loss + accuracy) on a fixed validation batch.
        curve_loss, curve_acc = evaluate_loss_accuracy(
            model=model,
            tokenizer=tokenizer,
            batch_examples=fixed_val_batch,
            device=device,
            max_length=max_length,
        )

        step_time = time.perf_counter() - step_start
        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss before: {val_loss_before:.6f}")
        print(f"Val loss after: {val_loss_after:.6f}")
        print(f"Validation curve point -> loss: {curve_loss:.6f}, acc: {curve_acc:.6f}")
        print(f"Reward: {reward:.6f}")
        print(f"Step time: {step_time:.2f}s")

        val_loss_curve.append(curve_loss)
        val_acc_curve.append(curve_acc)
        reward_history.append(reward)
        probs_history.append(bucket_probs[:])
        step_time_history.append(step_time)

        if device == "mps":
            torch.mps.empty_cache()

    test_batch = sample_uniform_batch(D_test, m=m, seed=seed + 30_000 + episode_id)
    final_test_loss, final_test_acc = evaluate_loss_accuracy(
        model=model,
        tokenizer=tokenizer,
        batch_examples=test_batch,
        device=device,
        max_length=max_length,
    )

    convergence_step = 1 + min(range(len(val_loss_curve)), key=lambda i: val_loss_curve[i])

    print("\nValidation loss curve (loss vs steps):")
    for i, v in enumerate(val_loss_curve, start=1):
        print(f"  step {i}: {v:.6f}")

    print("\nValidation accuracy curve (acc vs steps):")
    for i, a in enumerate(val_acc_curve, start=1):
        print(f"  step {i}: {a:.6f}")

    print("\nReward over time:")
    for i, r in enumerate(reward_history, start=1):
        print(f"  step {i}: {r:.6f}")

    print("\nTeacher bucket probabilities over time:")
    for i, probs in enumerate(probs_history, start=1):
        print(f"  step {i}: {probs}")

    print("\nStep times:")
    for i, s in enumerate(step_time_history, start=1):
        print(f"  step {i}: {s:.2f}s")

    print(f"\nFinal held-out test loss: {final_test_loss:.6f}")
    print(f"Final held-out test accuracy: {final_test_acc:.6f}")
    print(f"Convergence step (best val loss): {convergence_step}")

    result = {
        "method": "curriculum" if use_teacher else "baseline",
        "seed": seed,
        "episode_id": episode_id,
        "val_loss_curve": val_loss_curve,
        "val_acc_curve": val_acc_curve,
        "reward_history": reward_history,
        "probs_history": probs_history,
        "step_time_history": step_time_history,
        "avg_step_time": mean(step_time_history),
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "convergence_step": convergence_step,
        "bucket_stats": stats,
    }

    del model
    if device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return result


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
    T=3,
    B=8,
    m=8,
    model_name="Qwen/Qwen3.5-0.8B",
    lr=1e-5,
    max_length=512,
):
    curriculum_runs = []
    baseline_runs = []

    for seed in seeds:
        for ep in range(episodes):
            print(f"\n\n######## Seed={seed} Episode={ep} | Curriculum ########")
            curriculum_runs.append(
                run_training_loop(
                    T=T,
                    B=B,
                    m=m,
                    seed=seed,
                    model_name=model_name,
                    lr=lr,
                    max_length=max_length,
                    use_teacher=True,
                    episode_id=ep,
                )
            )

            print(f"\n\n######## Seed={seed} Episode={ep} | Baseline ########")
            baseline_runs.append(
                run_training_loop(
                    T=T,
                    B=B,
                    m=m,
                    seed=seed,
                    model_name=model_name,
                    lr=lr,
                    max_length=max_length,
                    use_teacher=False,
                    episode_id=ep,
                )
            )

    summarize_runs("Curriculum", curriculum_runs)
    summarize_runs("Vanilla SFT Baseline", baseline_runs)

    return {
        "curriculum_runs": curriculum_runs,
        "baseline_runs": baseline_runs,
    }


if __name__ == "__main__":
    run_final_evaluation(
        seeds=(11, 22, 33),
        episodes=1,
        T=3,
        B=8,
        m=8,
        model_name="Qwen/Qwen3.5-0.8B",
        lr=1e-5,
        max_length=512,
    )
