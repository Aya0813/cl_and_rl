from collections import Counter


def init_bucket_probs(K):
    """
    Initialize a uniform probability vector over K buckets.
    """
    # K = number of buckets (e.g., easy, medium, hard, ...)
    if K <= 0:
        raise ValueError("K must be > 0")

    # Start with equal probability for each bucket
    # Example: K=4 → [0.25, 0.25, 0.25, 0.25]
    return [1.0 / K] * K


def sample_a_t(bucket_probs):
    """
    Return current bucket distribution a_t.
    (In this simple teacher, a_t is just the current probability vector.)
    """
    # Ensure probabilities sum to something positive
    total = sum(bucket_probs)
    if total <= 0:
        raise ValueError("bucket_probs must have positive sum")

    # Normalize to ensure it is a valid probability distribution
    # (sum = 1)
    return [p / total for p in bucket_probs]


def update_bucket_probs(bucket_probs, used_bucket_indices, reward, step_size=0.05):
    """
    Simple reward-based update:
    - reward > 0  -> slightly increase probs of used buckets
    - reward <= 0 -> slightly decrease probs of used buckets
    Then normalize.
    """

    # Step size controls how aggressively we update probabilities
    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    # Ensure we have at least one bucket
    if len(bucket_probs) == 0:
        raise ValueError("bucket_probs must be non-empty")

    # Normalize input probabilities (safety)
    total = sum(bucket_probs)
    if total <= 0:
        raise ValueError("bucket_probs must have positive sum")

    probs = [p / total for p in bucket_probs]

    # If no buckets were used in this batch, nothing to update
    if len(used_bucket_indices) == 0:
        print(f"Current bucket probabilities after update: {probs}")
        return probs

    # Count how many times each bucket appeared in the batch
    # Example: [0,0,2,2,2,3] → {0:2, 2:3, 3:1}
    counts = Counter(used_bucket_indices)

    # Total number of samples in batch
    used_total = sum(counts.values())

    # Determine update direction:
    # reward > 0 → increase probabilities
    # reward <= 0 → decrease probabilities
    direction = 1.0 if reward > 0 else -1.0

    # Update probabilities for each used bucket
    for idx, c in counts.items():

        # Ensure bucket index is valid
        if idx < 0 or idx >= len(probs):
            raise ValueError(f"bucket index out of range: {idx}")

        # Update rule:
        # - proportional to how often the bucket appeared (c / used_total)
        # - scaled by step_size
        # - direction depends on reward
        probs[idx] += direction * step_size * (c / used_total)

    # Ensure probabilities don't go negative
    eps = 1e-8
    probs = [max(eps, p) for p in probs]

    # Normalize again so they sum to 1
    norm = sum(probs)
    probs = [p / norm for p in probs]

    # Print updated distribution for debugging/monitoring
    print(f"Current bucket probabilities after update: {probs}")

    return probs


if __name__ == "__main__":
    # Example usage

    K = 4  # number of buckets
    bucket_probs = init_bucket_probs(K)

    # Get current distribution a_t
    a_t = sample_a_t(bucket_probs)
    print(f"Initial a_t: {a_t}")

    # Simulate which buckets were used in a batch
    used_bucket_indices = [0, 0, 2, 2, 2, 3]

    # Example reward (positive → good step)
    reward = 0.12

    # Update teacher probabilities based on reward
    updated = update_bucket_probs(
        bucket_probs=a_t,
        used_bucket_indices=used_bucket_indices,
        reward=reward,
        step_size=0.05,
    )

    print(f"Updated bucket_probs: {updated}")