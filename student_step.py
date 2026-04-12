import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_gpu_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("No GPU backend available. Expected CUDA or MPS.")


def _ensure_gpu_device(device):
    if str(device) not in {"cuda", "mps"}:
        raise ValueError("This code is configured for GPU only. Use 'cuda' or 'mps'.")


def build_solution_only_batch(batch_examples, tokenizer, device, max_length=1024):
    """
    Build tensors for Causal LM training where loss is computed only on solution tokens.
    Expects each example to have: {"problem": ..., "solution": ...}
    """

    # If tokenizer has no padding token, reuse EOS token as padding
    # (common trick for causal language models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # means we reached the end of the sequence, so no loss is computed on padding tokens

    # Get integer IDs for padding and end-of-sequence tokens
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # These lists will store tokenized sequences before batching
    input_ids_list = []
    labels_list = []

    # Process each example in the batch individually
    for ex in batch_examples:
        problem = ex["problem"]      # input text (question)
        solution = ex["solution"]    # target text (answer / reasoning)

        # Format input as a prompt (this is what model sees)
        prompt_text = f"Problem:\n{problem}\n\nSolution:\n"


        # /// *** ///
        # Tokenize prompt and solution separately
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]

        # Concatenate prompt + solution → full input sequence
        full_ids = prompt_ids + solution_ids

        # Labels:
        # - mask prompt tokens with -100 (ignored in loss)
        # - keep solution tokens for loss computation
        full_labels = [-100] * len(prompt_ids) + solution_ids

        # Optionally append EOS token at the end
        if eos_id is not None:
            full_ids = full_ids + [eos_id]
            full_labels = full_labels + [eos_id]

        # Truncate sequences if too long
        full_ids = full_ids[:max_length]
        full_labels = full_labels[:max_length]

        # Convert to PyTorch tensors
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(full_labels, dtype=torch.long))

    # Determine max sequence length in this batch (for padding)
    max_len = max(len(x) for x in input_ids_list)
    batch_size = len(input_ids_list)

    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    # Fill tensors with actual data
    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        n = len(ids)

        # Fill input_ids with token IDs
        input_ids[i, :n] = ids

        # Attention mask: 1 = real token, 0 = padding
        attention_mask[i, :n] = 1

        # Labels: contains solution tokens, -100 elsewhere
        labels[i, :n] = labs

    # Move tensors to device (CPU or GPU)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def student_train_step(model, tokenizer, optimizer, batch_examples, device=None, max_length=1024):
    """
    One minimal student update step:
    forward -> cross-entropy loss -> backward -> parameter update

    Returns:
        updated_params: model.state_dict()
        batch_loss: float
    """

    if device is None:
        device = resolve_gpu_device()
    _ensure_gpu_device(device)

    # Put model in training mode (enables dropout, etc.)
    model.train()

    # Reset gradients before computing new ones
    optimizer.zero_grad()

    # Build input tensors (input_ids, attention_mask, labels)
    input_ids, attention_mask, labels = build_solution_only_batch(
        batch_examples=batch_examples,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )

    # Forward pass through model
    # When labels are provided, Hugging Face models automatically compute loss
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Extract computed loss (cross-entropy over non-masked tokens)
    loss = outputs.loss

    # Backpropagation: compute gradients
    loss.backward()

    # Update model parameters using optimizer
    optimizer.step()

    # Convert loss to Python float for logging
    batch_loss = float(loss.item())

    print(f"Batch loss: {batch_loss:.6f}")

    # Return updated model parameters and loss
    return model.state_dict(), batch_loss


def validation_batch_loss(
    model,
    tokenizer,
    D_val,
    m=8,
    device=None,
    max_length=1024,
    seed=None,
    validation_batch=None,
):
    """
    Sample a validation batch uniformly from D_val (or reuse validation_batch),
    then compute average validation loss for that batch.

    Returns:
        validation_batch: list of examples
        avg_val_loss: float
    """
    if device is None:
        device = resolve_gpu_device()
    _ensure_gpu_device(device)

    if validation_batch is None:
        if m <= 0:
            raise ValueError("m must be > 0")
        if len(D_val) == 0:
            raise ValueError("D_val is empty")

        rng = random.Random(seed)
        sample_size = min(m, len(D_val))
        sampled_indices = rng.sample(range(len(D_val)), sample_size)
        validation_batch = [D_val[i] for i in sampled_indices]

    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, labels = build_solution_only_batch(
            batch_examples=validation_batch,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        avg_val_loss = float(outputs.loss.item())

    print(f"Validation loss: {avg_val_loss:.6f}")
    return validation_batch, avg_val_loss


def compute_reward(val_loss_before, val_loss_after):
    """
    Minimal reward:
        reward = val_loss_before - val_loss_after
    """
    reward = float(val_loss_before) - float(val_loss_after)
    print(f"Reward: {reward:.6f}")
    return reward


if __name__ == "__main__":
    # Minimal usage example (downloads model/tokenizer from Hugging Face).
    model_name = "Qwen/Qwen3.5-0.8B"

    device = resolve_gpu_device()

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Define optimizer (AdamW is standard for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Example batch (toy data)
    demo_batch = [
        {
            "problem": "Compute 2 + 2.",
            "solution": "We add 2 and 2 to get 4.",
        },
        {
            "problem": "Compute 5 * 3.",
            "solution": "Multiplying gives 15.",
        },
    ]

    # Run one training step
    student_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        batch_examples=demo_batch,
        device=device,
        max_length=512,
    )

    # Demo validation set (toy data)
    D_val_demo = [
        {
            "problem": "Compute 10 - 3.",
            "solution": "Subtracting 3 from 10 gives 7.",
        },
        {
            "problem": "Compute 6 / 2.",
            "solution": "Dividing 6 by 2 gives 3.",
        },
        {
            "problem": "Compute 9 + 8.",
            "solution": "Adding gives 17.",
        },
        {
            "problem": "Compute 7 * 4.",
            "solution": "Multiplying gives 28.",
        },
        {
            "problem": "Compute 12 - 5.",
            "solution": "Subtracting gives 7.",
        },
        {
            "problem": "Compute 3 * 3.",
            "solution": "Multiplying gives 9.",
        },
        {
            "problem": "Compute 14 / 7.",
            "solution": "Dividing gives 2.",
        },
        {
            "problem": "Compute 11 + 6.",
            "solution": "Adding gives 17.",
        },
    ]

    # Run validation loss on a sampled batch
    val_batch, val_loss = validation_batch_loss(
        model=model,
        tokenizer=tokenizer,
        D_val=D_val_demo,
        m=8,
        device=device,
        max_length=512,
        seed=42,
    )
    print(f"Returned validation loss: {val_loss:.6f}")

    # Reuse the SAME batch to show deterministic reuse behavior
    _, val_loss_same_batch = validation_batch_loss(
        model=model,
        tokenizer=tokenizer,
        D_val=D_val_demo,
        device=device,
        max_length=512,
        seed=40,
        validation_batch=val_batch,
    )
    print(f"Returned validation loss (same batch): {val_loss_same_batch:.6f}")

    # Task 5: reward from validation improvement
    reward = compute_reward(
        val_loss_before=val_loss,
        val_loss_after=val_loss_same_batch,
    )
    print(f"Returned reward: {reward:.6f}")
