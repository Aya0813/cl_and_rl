import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device="auto"):
    """
    Resolve runtime device from:
      - auto: prefer cuda, then mps, then cpu
      - cuda / mps / cpu: enforce requested device
    """
    d = str(device).lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return "cuda"
    if d == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps' but MPS is not available.")
        return "mps"
    if d == "cpu":
        return "cpu"
    raise ValueError("device must be one of: auto, cpu, mps, cuda")


def resolve_gpu_device():
    # Backward-compatible alias used in other modules.
    return resolve_device("auto")


def _ensure_device(device):
    if str(device).lower() not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: cpu, mps, cuda")


def build_solution_only_batch(batch_examples, tokenizer, device, max_length=1024): # why this function name mentions solution only? - answer: because we're only computing loss on the solution tokens, not the problem tokens. The problem tokens are part of the input prompt, but we don't want to compute loss on them since they are not the target output. The model is trained to generate the solution given the problem, so we only care about the loss on the solution tokens. The function takes a batch of examples (each with a "problem" and "solution"), tokenizes them, and builds input tensors where the loss will only be computed on the solution part of the sequence.
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
    orig_lengths = []  # Track original (unclipped) sequence lengths
    clipped_counts = 0  # Track number of clipped examples

    # Process each example in the batch individually

    for ex in batch_examples:
        problem = ex["problem"]      # input text (question)
        solution = ex["solution"]    # target text (answer / reasoning)
        prompt_text = f"Problem:\n{problem}\n\nSolution:\n"  # Build prompt
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]  # Tokenize prompt
        solution_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]  # Tokenize solution
        full_ids = prompt_ids + solution_ids  # Concatenate prompt and solution
        full_labels = [-100] * len(prompt_ids) + solution_ids  # Mask prompt in labels (why? -> we don't want to compute loss on the prompt tokens, only on the solution tokens)
        # Why -100? -> In PyTorch's CrossEntropyLoss, the label -100 is used as a special value to indicate that the corresponding token should be ignored when computing the loss. By setting the labels for the prompt tokens to -100, we ensure that they do not contribute to the loss calculation, and only the solution tokens (which have their actual token IDs in the labels) are used for training the model. This allows us to train the model to generate the solution based on the prompt without penalizing it for the prompt tokens.
        orig_len = len(full_ids)  # Original sequence length
        orig_lengths.append(orig_len)  # Track original length
        # Append EOS token if available
        if eos_id is not None:
            full_ids = full_ids + [eos_id]
            full_labels = full_labels + [eos_id]
        # Clip to max_length
        if len(full_ids) > max_length:
            clipped_counts += 1  # Count as clipped
        full_ids = full_ids[:max_length] # keep only the first max_length tokens and discard the rest.
        full_labels = full_labels[:max_length]

        # WHY do we need to truncate seperatly full_ids and full_labels? answer: We need to truncate full_ids and full_labels separately because they may have different values. The full_ids list contains the token IDs for the input sequence (prompt + solution), while the full_labels list contains the corresponding labels for loss computation (with -100 for prompt tokens and actual token IDs for solution tokens). When we truncate to max_length, we need to ensure that both lists are truncated in the same way so that the input tokens and their corresponding labels remain aligned. If we were to truncate them together without considering their different values, we might end up with misaligned input and label sequences, which would lead to incorrect loss computation during training.
        # DIVE DEEP LATER 

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))  # Store input ids
        labels_list.append(torch.tensor(full_labels, dtype=torch.long))  # Store labels
        # Visual example of what the input_ids and labels would look like for a simple problem and solution ? answer: Let's say we have a simple problem and solution:
        # Problem: "Compute 2 + 2."
        # Solution: "We add 2 and 2 to get 4."
        # The prompt would be:
        # "Problem:\nCompute 2 + 2.\n\nSolution:\n"
        # After tokenization, let's say we get the following token IDs (these are just hypothetical and depend on the tokenizer):
        # prompt_ids = [101, 102, 103, 104]  # token IDs for "Problem:\nCompute 2 + 2.\n\nSolution:\n"
        # solution_ids = [201, 202, 203, 204, 205, 206]  # token IDs for "We add 2 and 2 to get 4."
        # full_ids = prompt_ids + solution_ids = [101, 102, 103, 104, 201, 202, 203, 204, 205, 206]
        # full_labels = [-100, -100, -100, -100, 201, 202, 203, 204, 205, 206]
        # In this example, the first four tokens correspond to the prompt and are masked with -100 in the labels, while the solution tokens have their actual token IDs in the labels for loss computation.

    # Determine max sequence length in this batch (for padding) - what is the goal of padding here ? answer: The goal of padding is to ensure that all sequences in the batch have the same length, which is necessary for efficient batch processing in PyTorch. Since different examples may have different lengths after tokenization, we pad the shorter sequences with a special padding token (with ID pad_id) so that they all match the length of the longest sequence in the batch. This allows us to create tensors of shape (batch_size, max_seq_length) that can be fed into the model without issues.

    max_len = max(len(x) for x in input_ids_list)
    batch_size = len(input_ids_list)

    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long) 
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long) 

    # Fill tensors with actual data, explain further this for loop: This for loop iterates over each example in the batch (using enumerate to get both the index i and the corresponding token IDs and labels). For each example, it fills the input_ids tensor with the token IDs, sets the attention_mask to 1 for real tokens (and leaves it as 0 for padding), and fills the labels tensor with the label IDs. The variable n is the length of the current example's token sequence, so we only fill up to n tokens for each example, leaving the rest as padding. This way, we create properly padded tensors that can be fed into the model for training.
    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)): # What s the difference between ids and labs here ? and what does labs stand for? answer: In this context, "ids" refers to the token IDs for the input sequence (which includes both the prompt and the solution), while "labs" refers to the label IDs that indicate which tokens should contribute to the loss (with -100 for prompt tokens and actual token IDs for solution tokens). The variable name "labs" is just a shorthand for "labels". So, for each example in the batch, "ids" contains the token IDs that will be fed into the model, and "labs" contains the corresponding labels that will be used to compute the loss during training.
        n = len(ids)

        # Fill input_ids with token IDs
        input_ids[i, :n] = ids

        # Attention mask: 1 = real token, 0 = padding
        attention_mask[i, :n] = 1

        # Labels: contains solution tokens, -100 elsewhere
        labels[i, :n] = labs
    
    # give a visual example of the for loop above: Let's say we have a batch of 2 examples with the following token IDs and labels after processing:
    # Example 1:
    # ids = [101, 102, 103, 104, 201, 202, 203, 204, 205, 206] (length n=10)
    # labs = [-100, -100, -100, -100, 201, 202, 203, 204, 205, 206]
    # Example 2:
    # ids = [101, 102, 103, 201, 202] (length n=5)
    # labs = [-100, -100, -100, 201, 202]
    # The max_len for this batch would be 10 (from example 1). The input_ids tensor would be initialized as a 2x10 tensor filled with the pad_id (let's say pad_id=0), and the labels tensor would be initialized as a 2x10 tensor filled with -100. After the for loop, the tensors would look like this:
    # input_ids = [[101, 102, 103, 104, 201, 202, 203, 204, 205, 206],
    #              [101, 102, 103, 201, 202, 0, 0, 0, 0, 0]]
    # labels = [[-100, -100, -100, -100, 201, 202, 203, 204, 205, 206],
    #           [-100, -100, -100, 201, 202, -100, -100, -100, -100, -100]]

    # why in theses examples labs always start with -100 ? answer: The "labs" (labels) start with -100 because the first part of the input sequence corresponds to the prompt, and we do not want to compute loss on the prompt tokens. The model is trained to generate the solution given the prompt, so we only care about the loss on the solution tokens. By setting the labels for the prompt tokens to -100, we ensure that they are ignored during loss computation, and only the solution tokens contribute to the training signal. This is a common technique when training language models on tasks where only a portion of the input should be used for computing loss.


    # Compute batch clipped ratio (fraction of examples clipped)
    batch_clipped_ratio = clipped_counts / batch_size if batch_size > 0 else 0.0
    # Compute per-example clipped ratio (how much was clipped for each example)
    clipped_ratios = []
    for i, orig_len in enumerate(orig_lengths):
        clipped_len = min(orig_len, max_length)
        ratio = 0.0 if orig_len == 0 else max(0, orig_len - clipped_len) / orig_len
        clipped_ratios.append(ratio)
    # Return tensors and clipping stats
    return input_ids.to(device), attention_mask.to(device), labels.to(device), batch_clipped_ratio, clipped_ratios


def student_train_step(model, tokenizer, optimizer, batch_examples, device=None, max_length=1024):
    """
    One minimal student update step:
    forward -> cross-entropy loss -> backward -> parameter update

    Returns:
        updated_params: model.state_dict()
        batch_loss: float
    """

    if device is None:
        device = resolve_device("auto")
    _ensure_device(device)

    # Put model in training mode (enables dropout, etc.)
    model.train()

    # Reset gradients before computing new ones 
    # Why do we need this step? answer: In PyTorch, gradients are accumulated by default. This means that if you call loss.backward() multiple times without resetting the gradients, they will be summed together. To prevent this and ensure that each training step only uses the gradients from the current batch, we call optimizer.zero_grad() at the beginning of each training step to reset all gradients to zero before computing new ones. This is a standard practice in PyTorch training loops.
    optimizer.zero_grad()


    # Build input tensors and clipping stats
    input_ids, attention_mask, labels, batch_clipped_ratio, clipped_ratios = build_solution_only_batch(
        batch_examples=batch_examples,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )


    # Forward pass through model
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

    print(f"Batch clipped ratio: {batch_clipped_ratio:.4f}")  # Print batch clipped ratio

    # Return updated model parameters, loss, and clipping stats
    return model.state_dict(), batch_loss, batch_clipped_ratio, clipped_ratios


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
        device = resolve_device("auto")
    _ensure_device(device)

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
        input_ids, attention_mask, labels, *_ = build_solution_only_batch( # unpack only the first 3 values
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

    device = resolve_device("auto")

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
