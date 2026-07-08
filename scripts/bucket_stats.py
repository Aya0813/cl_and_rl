from collections import defaultdict


def update_bucket_loss_stats(
    batch_examples,
    bucket_labels,
    losses,
    stats=None,
    alpha=0.3, 
):
    """
    Update running loss statistics per bucket.

    Args:
        batch_examples: batch of examples (kept for interface completeness; not used) -> will be the training_batch 
        bucket_labels: list of bucket labels (same length as losses) -> will be sampled_bucket_indices
        losses: list of per-example losses 
        stats: running stats dict to update in-place
        alpha: EMA factor (0 < alpha <= 1) -> 

    Returns:
        stats dict with keys:
          - running_avg_loss : dict[bucket -> float] 
          - loss_change      : dict[bucket -> float]
          - batch_avg_loss   : dict[bucket -> float] (for buckets seen this batch) -> for this current training batch only, what was the avg loss of examples from each bucket 
    
    Further explanation: what does each one of the above tell us???? 
          - running_avg_loss : how hard each bucket currently is? [ memory across steps unlike batch_avg_loss for only this current training step ]
          - loss_change      : whether that bucket is improving or worsening? 

          this compressed summary per bucket is what later gets turned into the teacher observation in build_teacher_observation() function 


    - batch_avg_loss
        - average loss for a bucket in just the current batch
        - noisy but very fresh
    - running_avg_loss
        - smoothed version across many steps
        - more stable, less noisy
    - And loss_change
        - how much running_avg_loss moved this step
        - tells the teacher not just “how hard is this bucket?” but also “is it improving or getting worse?”
    
    """
    _ = batch_examples      # not needed now, but included as requested by task inputs

    if len(bucket_labels) != len(losses):
        raise ValueError("bucket_labels and losses must have the same length")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    if stats is None:
        stats = {
            "running_avg_loss": {},
            "loss_change": {},
            "batch_avg_loss": {},
        }

    # Compute batch average loss per bucket for buckets appearing in this batch.
    sums = defaultdict(float)
    counts = defaultdict(int)
    for b, l in zip(bucket_labels, losses): 
        sums[b] += float(l)     # sum up the losses for each bucket
        counts[b] += 1          # count how many examples are in each bucket
    
    # at the end: 
        # sums = {0: 1.0, 1: 0.7, 2: 2.2}
        # counts = {0: 2, 1: 1, 2: 2}
    # then compute the avg loss per bucket 

    batch_avg_loss = {b: sums[b] / counts[b] for b in counts}
    running_avg_loss = stats.get("running_avg_loss", {})
    loss_change = stats.get("loss_change", {})

    # Update only buckets that appeared in this batch.
    for b, batch_avg in batch_avg_loss.items():
        prev = running_avg_loss.get(b, batch_avg)
        new = alpha * batch_avg + (1 - alpha) * prev
        running_avg_loss[b] = new
        loss_change[b] = new - prev

    stats["running_avg_loss"] = running_avg_loss
    stats["loss_change"] = loss_change
    stats["batch_avg_loss"] = batch_avg_loss

    print("Avg loss per bucket:")
    for b in sorted(batch_avg_loss):
        print(f"  {b}: {batch_avg_loss[b]:.6f}")

    print("Change in loss per bucket:")
    for b in sorted(batch_avg_loss):
        print(f"  {b}: {loss_change[b]:.6f}")

    return stats


if __name__ == "__main__":
    demo_examples = [{}, {}, {}, {}, {}]
    demo_bucket_labels = [0, 0, 1, 2, 2]
    demo_losses = [1.2, 1.0, 0.9, 1.5, 1.1]

    stats = update_bucket_loss_stats(
        batch_examples=demo_examples,
        bucket_labels=demo_bucket_labels,
        losses=demo_losses,
        stats=None,
        alpha=0.3,
    )

    print("Updated stats:")
    print(stats)
