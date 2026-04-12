from collections import defaultdict
import random

from datasets import load_dataset


def prepare_math_dataset(seed: int = 42):
    """
    Returns:
        D_tr: training split (90%)
        D_val: validation split (10%)
        buckets: dict[level, list[example]] built from D_tr
    """
    # Load the full dataset split.
    dataset = load_dataset("EleutherAI/hendrycks_math", 'algebra', split="train")

    # Keep only rows that have a difficulty label.
    dataset = dataset.filter(
        lambda ex: ex["level"] is not None and str(ex["level"]).strip() != ""
    )

    # Try stratified split by level.
    # This requires a ClassLabel column, so we encode "level" first.
    try:
        encoded = dataset.class_encode_column("level")
        split = encoded.train_test_split(
            test_size=0.10,
            seed=seed,
            stratify_by_column="level",
        )
    except Exception:
        # Fallback to a normal random split if stratification is not possible.
        split = dataset.train_test_split(test_size=0.10, seed=seed)

    D_tr = split["train"]
    D_val = split["test"]

    # Build level buckets from D_tr.
    buckets = defaultdict(list)
    for ex in D_tr:
        buckets[ex["level"]].append(ex)

    buckets = dict(sorted(buckets.items(), key=lambda x: str(x[0])))
    return D_tr, D_val, buckets


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
    D_tr, D_val, buckets = prepare_math_dataset()

    print(f"D_tr size: {len(D_tr)}")
    print(f"D_val size: {len(D_val)}")
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
