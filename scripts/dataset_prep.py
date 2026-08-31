from collections import defaultdict
import random

from datasets import load_dataset


def prepare_math_dataset(
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1, 
    holdout_ratio: float = 0.1,
):
    """
    Returns:
        D_tr: training split
        D_val: validation split
        D_holdout: held-out/test split
        buckets: dict[level, list[example]] built from D_tr
    """
    if abs((train_ratio + val_ratio + holdout_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + holdout_ratio must equal 1.0")

    # Load the full dataset split. 
    dataset = load_dataset("EleutherAI/hendrycks_math", 'algebra', split="train")

    # Keep only rows that have a "level" label.
    dataset = dataset.filter(
        lambda ex: ex["level"] is not None and str(ex["level"]).strip() != ""
    )

    # Try stratified 80/10/10 split by level.
    try:
        encoded = dataset.class_encode_column("level")
        first = encoded.train_test_split(
            test_size=(1.0 - train_ratio),  # temp = val + holdout
            seed=seed,
            stratify_by_column="level", # stratify_by_column - preserves class proportions of `level` across splits (approximately), so train/val/test have similar level distribution.
        )
        temp = first["test"]
        second = temp.train_test_split(
            test_size=holdout_ratio / (val_ratio + holdout_ratio), # This fraction picks the holdout share within `temp`. Example: if val=0.1 and holdout=0.1, temp=0.2, holdout fraction in temp = 0.1/0.2 = 0.5.
            seed=seed,
            stratify_by_column="level",
        )
        D_tr = first["train"]
        D_val = second["train"]
        D_holdout = second["test"]
    
    except Exception:
        # Fallback to non-stratified split.
        first = dataset.train_test_split(test_size=(1.0 - train_ratio), seed=seed)
        temp = first["test"]
        second = temp.train_test_split(
            test_size=holdout_ratio / (val_ratio + holdout_ratio),
            seed=seed,
        )
        D_tr = first["train"]
        D_val = second["train"]
        D_holdout = second["test"]


    # To know whether the statification happened or NOT
    print("Stratified split successful." if 'encoded' in locals() else "Stratified split failed, used non-stratified split.")

    # TO SHOW THE DISTRIBUTION OF LEVELS IN D_tr, D_val, AND D_holdout TO VERIFY STRATIFICATION (OR LACK THEREOF)
    def print_level_distribution(split_name, split_data):
        level_counts = defaultdict(int)
        for ex in split_data:
            level_counts[ex["level"]] += 1
        print(f"{split_name} level distribution:")
        for level, count in sorted(level_counts.items()):
            print(f"  Level {level}: {count} examples")

    print_level_distribution("D_tr", D_tr)
    print_level_distribution("D_val", D_val)
    print_level_distribution("D_holdout", D_holdout)


    # Build level buckets from D_tr.
    buckets = defaultdict(list)
    for ex in D_tr:
        buckets[ex["level"]].append(ex)

    buckets = dict(sorted(buckets.items(), key=lambda x: str(x[0]))) # buckets looks like: {'1': [ex1, ex7], '2': [ex3, ex9, ex11], '3': [ex2]}...
    return D_tr, D_val, D_holdout, buckets # KEY INFO here to keep in mind -> the buckets are built from D_tr only, not from D_val or D_holdout. This is important for training and evaluation consistency.


def sample_batch_from_buckets(buckets, a_t, B, seed=None):
    """
    Args:
        buckets: dict[bucket_name, list[example]]
        a_t: probability vector over buckets (same order as list(buckets.keys()))
        B: batch size
        seed: optional random seed for deterministic sampling

    Returns:
        sampled_examples: list of examples
        sampled_bucket_indices: list of sampled bucket indices
    """
    bucket_names = list(buckets.keys())
    if len(bucket_names) != len(a_t):
        raise ValueError("len(a_t) must match number of buckets")
    if B <= 0:
        raise ValueError("B must be > 0")
    if sum(a_t) <= 0:
        raise ValueError("a_t must have positive total probability")
    if any(len(buckets[name]) == 0 for name in bucket_names):
        raise ValueError("all buckets must be non-empty")

    rng = random.Random(seed)
    sampled_examples = []
    sampled_bucket_indices = []

    for _ in range(B):
        bucket_idx = rng.choices(range(len(bucket_names)), weights=a_t, k=1)[0]
        bucket_name = bucket_names[bucket_idx]
        example = rng.choice(buckets[bucket_name])
        sampled_bucket_indices.append(bucket_idx)
        sampled_examples.append(example)

    return sampled_examples, sampled_bucket_indices


if __name__ == "__main__":
    D_tr, D_val, D_holdout, buckets = prepare_math_dataset()

    print(f"D_tr size: {len(D_tr)}")
    print(f"D_val size: {len(D_val)}")
    print(f"D_holdout size: {len(D_holdout)}")
    print("Bucket sizes:")
    for level, examples in buckets.items():
        print(f"  {level}: {len(examples)}")

    # Task 2 demo: sample one batch and print sampled-bucket distribution.
    B = 32
    a_t = [1.0 / len(buckets)] * len(buckets)  # uniform over buckets for demo
    _, sampled_bucket_indices = sample_batch_from_buckets(
        buckets=buckets,
        a_t=a_t,
        B=B,
        seed=42,
    )

    bucket_count = defaultdict(int)
    for idx in sampled_bucket_indices:
        bucket_count[idx] += 1

    print("Sampled bucket distribution (one batch):")
    for idx in range(len(buckets)):
        print(f"  bucket_idx {idx}: {bucket_count[idx]}")
