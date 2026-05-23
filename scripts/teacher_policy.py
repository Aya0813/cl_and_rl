import torch
import torch.nn as nn


def init_bucket_probs(K):
    if K <= 0:
        raise ValueError("K must be > 0")
    return [1.0 / K] * K # Without the * K, you'd have 1 probability instead of K probabilities (one per bucket)
    # e.g. if you have 3 buckets -> K = 3 -> [1.0 / 3] * 3 = [0.333, 0.333, 0.333]

def sample_a_t(bucket_probs):
    total = sum(bucket_probs)
    if total <= 0:
        raise ValueError("bucket_probs must have positive sum")
    return [p / total for p in bucket_probs]


class TeacherMLP(nn.Module):
    """
    Very simple 2-layer MLP teacher policy.
    Input: observation o_t of size (1 + K + K)
    Output: logits over K buckets
    """
    def __init__(self, obs_dim, hidden_dim, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, observation):
        return self.net(observation)


def init_teacher_model(K, device, hidden_dim=32):
    """
    Initialize a simple 2-layer MLP teacher.
    
    ARCHITECTURE VISUALIZATION: -> check tablette for detailed visual 
        - Input layer: obs_dim = 1 + 2K (step feature + K bucket stats + K bucket changes)
        - Hidden layer: hidden_dim (e.g., 32)
        - Output layer: K logits (one per bucket)
    
    HYPERPARAMETER IMPACT:
    - K (num_buckets): Directly determines input/output size. More buckets = larger observation + output.
    - hidden_dim: Controls model capacity. Larger hidden_dim = more expressive policy, but risk of overfitting.
    - obs_dim (computed): 1 + K + K = 1 + 2K (step feature + K bucket statistics + K bucket changes)
    
    CHOOSING hidden_dim:
    - Small (8-16): Fast, low capacity, may underfit (teacher can't learn complex curriculum)
    - Medium (32-64): Balance between speed and expressiveness (default 32 is good for PoC)
    - Large (128-256): High capacity, slower training, risk of overfitting to small number of episodes
    - Rule of thumb: Start with hidden_dim ≈ K to 2*K (e.g., K=3 → hidden_dim=32-64)
    """
    # Validate that K (number of buckets) is positive
    if K <= 0:
        raise ValueError("K must be > 0")
    
    # Compute observation dimension: step_feature(1) + running_avg_loss(K) + loss_change(K) = 1 + 2K
    obs_dim = 1 + K + K
    
    # Instantiate the TeacherMLP with computed obs_dim, specified hidden_dim, and output size K
    model = TeacherMLP(obs_dim=obs_dim, hidden_dim=hidden_dim, K=K).to(device)
    
    # Return the model (already moved to the specified device: cpu/cuda/mps)
    return model


def build_teacher_observation(step_t, stats, K, T, device): # [ TO DIVE INTO IT DEEPLY LATER ]
    """
    o_t = (current step, running avg loss per bucket, running change per bucket)
    """
    running_avg = stats.get("running_avg_loss", {}) if stats else {}
    loss_change = stats.get("loss_change", {}) if stats else {}

    step_feature = torch.tensor([step_t / max(1, T - 1)], dtype=torch.float32, device=device) # why max(1, T - 1) - answer: We use max(1, T - 1) to avoid division by zero in the case where T (total number of steps) is 1. If T is 1, then we would be dividing by zero when normalizing step_t, which would cause an error. By using max(1, T - 1), we ensure that we are dividing by at least 1, which allows the code to run without errors even in edge cases where T is very small. This way, the step feature will be correctly normalized to a value between 0 and 1 regardless of the total number of steps.
    # why do we need to normalize step_t? answer: Normalizing step_t to a value between 0 and 1 helps the teacher model learn more effectively by providing a consistent scale for the time feature. This allows the model to better understand the progression of time during the episode and how it relates to the student's learning progress. If we used raw step_t values, the model might struggle to learn meaningful patterns, especially if T is large, as the step feature would have a wide range of values. Normalization helps stabilize training and allows the teacher to focus on learning the relationship between time and the bucket statistics rather than dealing with varying scales of input.
    
    avg_vec = torch.tensor(
        [float(running_avg.get(k, 0.0)) for k in range(K)], # this line is creating a vector (avg_vec) that contains the running average loss for each bucket. It iterates over the range of K (number of buckets) and retrieves the running average loss for each bucket k from the stats dictionary. If there is no entry for a particular bucket k, it defaults to 0.0. This vector provides the teacher model with information about how well the student is performing on each bucket, which can inform the teacher's decisions on how to allocate training examples in future steps.
        dtype=torch.float32,
        device=device, 
    )
    change_vec = torch.tensor(
        [float(loss_change.get(k, 0.0)) for k in range(K)], # The loss change per bucket is computed and stored in the stats dictionary during the training loop. After each training step, we calculate the change in loss for each bucket by comparing the current running average loss to the previous running average loss. This change is then stored in the stats dictionary under "loss_change" for each bucket k. When we build the teacher observation, we retrieve these loss change values from the stats dictionary to create the change_vec, which provides the teacher model with information about how the student's performance is evolving over time for each bucket. This allows the teacher to make informed decisions based on both the current performance (running average loss) and the trend of that performance (loss change).
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([step_feature, avg_vec, change_vec], dim=0) # torch.cat is a PyTorch function that concatenates tensors along a specified dimension. In this case, we are concatenating the step_feature, avg_vec, and change_vec tensors along dimension 0 (the default) to create a single observation vector that combines all the relevant information for the teacher model. The resulting tensor will have a shape of (1 + K + K,) which corresponds to the step feature (1) plus the running average loss for each bucket (K) plus the loss change for each bucket (K). This combined observation vector is what the teacher model will use as input to make its decisions about how to allocate training examples across the buckets.

    # give me a visual example to understand what build_teacher_observation is doing more concretely:
    # Let's say we have K=3 buckets and we are at step_t=5 out of T=10 total steps. The stats dictionary contains the following information:
    # stats = {
    #     "running_avg_loss": {0: 0.5, 1: 0.3, 2: 0.1},
    #     "loss_change": {0: -0.1, 1: -0.05, 2: 0.02}
    # }
    # The build_teacher_observation function will create the following tensors:
    # step_feature = torch.tensor([5 / 9], dtype=torch.float32)
    # avg_vec = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float32)
    # change_vec = torch.tensor([-0.1, -0.05, 0.02], dtype=torch.float32)
    # Then it will concatenate these tensors to create the final observation:
    # observation = torch.cat([step_feature, avg_vec, change_vec], dim=0)
    # The resulting observation tensor will look like this:
    # observation = torch.tensor([0.5556, 0.5, 0.3, 0.1, -0.1, -0.05, 0.02], dtype=torch.float32)
    # This observation vector contains the normalized step feature (0.5556), the running average loss for each bucket (0.5, 0.3, 0.1), and the loss change for each bucket (-0.1, -0.05, 0.02). 
    # The teacher model will use this information to make decisions about how to allocate training examples across the buckets in the next step.


def teacher_policy_probs(teacher_model, observation): 
    """
    [ TO DIVE INTO IT DEEPLY LATER ]
    mu_phi(. | o_t) -> distribution over buckets.
    The notation mu_phi(. | o_t) is commonly used in RL to denote a policy that is parameterized by phi (the parameters of the teacher model) and conditioned on the observation o_t. The "mu" typically represents the policy function that maps observations to actions (or in this case, distributions over buckets). The "phi" indicates that this policy is parameterized by the weights of the teacher model, which are learned during training. By using this notation, we can clearly indicate that the policy's behavior is determined by the current parameters of the teacher model and the input observation, which is essential for understanding how the teacher learns to select buckets based on the student's progress.
    Kept simple: logits are phi plus a lightweight observation signal. 
    phi represents the parameters of the teacher model (the weights of the MLP). The logits are computed by passing the observation through the teacher model, which applies its learned parameters (phi) to produce the raw output. The observation signal includes the current step feature, running average loss per bucket, and loss change per bucket, which are all processed by the teacher model to inform its decision on how to distribute probabilities over the buckets. By keeping the architecture simple and focused on these key features, we allow the teacher to learn an effective curriculum strategy without overcomplicating the model or requiring additional inputs.
    The teacher model takes the current observation (which includes the step feature, running average loss per bucket, and loss change per bucket) and processes it through a simple 2-layer MLP to produce raw logits for each bucket. These logits represent the unnormalized preferences of the teacher for selecting each bucket based on the current state of the student's learning (as captured by the observation). By applying a softmax function to these logits, we convert them into a valid probability distribution over the buckets, which allows us to sample actions (bucket selections) according to the teacher's policy. This approach keeps the teacher's decision-making process lightweight and focused on the most relevant information about the student's progress, without requiring complex architectures or additional inputs.
    """
    logits = teacher_model(observation)
    return torch.softmax(logits, dim=0) # We use softmax to convert the raw logits output by the teacher model into a valid probability distribution over the buckets. The softmax function takes a vector of real-valued logits and transforms it into a vector of probabilities that sum to 1. This is essential because we want to sample actions (bucket selections) according to these probabilities, and they need to be non-negative and sum to 1 for that sampling process to be meaningful. By applying softmax, we ensure that the teacher's policy outputs a proper distribution over the buckets based on the current observation.
    # The teacher_policy_probs function is a crucial component of the REINFORCE algorithm as it defines the policy that the teacher uses to select actions (bucket selections) based on the current observation. In REINFORCE, we need to compute the log probabilities of the actions taken by the teacher in order to calculate the policy gradient and update the teacher's parameters. The probabilities returned by this function are used to sample actions and compute log probabilities, which are then used in the REINFORCE update step to optimize the teacher's policy based on the rewards received from the environment (the student's performance). Therefore, this function directly influences how the teacher learns to select buckets that will maximize the student's learning progress over time.

def sample_teacher_action(teacher_model, observation, concentration_scale=20.0): # [ TO CONFIRM ] 
    """
    Sample a_t (distribution over buckets) from Dirichlet policy.  [ TO DIVE INTO IT DEEPLY LATER ]
    We use a Dirichlet distribution to sample the action a_t because we want to model a probability distribution over the buckets that can capture uncertainty and allow for exploration. The Dirichlet distribution is a natural choice for modeling distributions over categorical variables (like our buckets) because it is defined over the simplex (the space of valid probability distributions). By sampling from a Dirichlet distribution parameterized by the teacher's output probabilities, we can generate a stochastic action that reflects the teacher's current beliefs about which buckets are more beneficial for the student's learning. This allows the teacher to explore different bucket selections while still being guided by its learned policy, which can lead to better curriculum strategies over time.

    Returns:
      - a_t_list: python list for data sampler
      - log_prob_t: torch scalar for REINFORCE
      - probs: current mean bucket probabilities
    """

    probs = teacher_policy_probs(teacher_model, observation)
    concentration = probs * concentration_scale + 1e-4 # This line is scaling the probabilities output by the teacher model to create the concentration parameters for the Dirichlet distribution. The concentration parameters determine how peaked or flat the distribution is. By multiplying the probabilities by a concentration_scale (e.g., 20.0), we can control the level of exploration in the sampling process. A higher concentration_scale will lead to a more peaked distribution, meaning that the teacher will be more likely to select buckets with higher probabilities, while a lower concentration_scale will lead to a flatter distribution, allowing for more exploration of different buckets. The addition of 1e-4 is a small constant added to ensure numerical stability and prevent issues with zero probabilities when creating the Dirichlet distribution.
    dist = torch.distributions.Dirichlet(concentration)
    a_t = dist.rsample() # We use rsample() instead of sample() because rsample() allows for reparameterization, which is necessary for backpropagation through the sampling process. In the REINFORCE algorithm, we need to compute gradients with respect to the teacher's parameters based on the actions taken and the rewards received. By using rsample(), we can backpropagate through the sampled action a_t and compute the policy gradient effectively. If we were to use sample(), we would not be able to compute gradients with respect to the teacher's parameters, which would prevent us from updating the teacher's policy based on the observed rewards.
    log_prob_t = dist.log_prob(a_t)
    print (f"probs: {probs.detach().cpu().tolist()}")
    print (f"concentration: {concentration.detach().cpu().tolist()}")
    print (f"dist: {dist}")
    print (f"Sampled a_t: {a_t.detach().cpu().tolist()}")
    print (f"log_prob_t: {log_prob_t.detach().cpu().tolist()}")
    
    return a_t.detach().cpu().tolist(), log_prob_t, probs.detach().cpu().tolist() 
    # We are returning three things from the sample_teacher_action function:
        # 1. a_t_list: This is the sampled action (a_t) from the Dirichlet distribution, which is detached from the computation graph (so it won't affect gradients), moved to the CPU (to ensure it's in a format that can be easily used for sampling), and converted to a Python list. This is the action that will be used by the data sampler to select buckets for training examples.
        # 2. log_prob_t: This is the log probability of the sampled action a_t under the teacher's policy. It is returned as a torch scalar, which will be used in the REINFORCE update to compute the policy gradient and update the teacher's parameters based on the rewards received from the environment.
        # 3. probs: This is the current mean bucket probabilities output by the teacher model, which is also detached from the computation graph and moved to the CPU and converted to a list. This can be useful for logging or analysis purposes to see what the teacher's current beliefs are about the bucket probabilities at the time of sampling the action.
        # note = it will work with .cpu() even when running on HPC. The reason we use .cpu() is to ensure that the sampled action and probabilities are in a format that can be easily used for further processing, such as sampling from the data sampler or logging. When we detach the tensors from the computation graph and move them to the CPU, we can convert them to Python lists, which are more convenient for certain operations that may not require GPU tensors. This does not affect the training process on the GPU, as the teacher model and its parameters will still be on the GPU for efficient computation. The .cpu() call simply ensures that the outputs we need for sampling and logging are in a compatible format without interfering with the GPU computations of the teacher model.
        # The computation graph is a data structure used in automatic differentiation frameworks like PyTorch to keep track of the operations performed on tensors and how they are connected. When we perform operations on tensors, PyTorch builds a computation graph that records the sequence of operations and the dependencies between tensors. This graph allows PyTorch to compute gradients efficiently during backpropagation by traversing the graph in reverse order. When we call .detach() on a tensor, we are telling PyTorch to stop tracking operations on that tensor and to treat it as a constant for the purposes of gradient computation. This means that any operations performed on the detached tensor will not be recorded in the computation graph, and gradients will not be computed for it during backpropagation. This is useful when we want to use certain values (like sampled actions or probabilities) without affecting the gradient flow for the teacher model's parameters.

def compute_stability_penalty_kl(a_t, p, eps=1e-8):
    """
    KL(a_t || p)
    """
    if len(a_t) != len(p):
        raise ValueError("a_t and p must have same length")
    kl = 0.0
    for ai, pi in zip(a_t, p):
        ai = max(float(ai), eps)
        pi = max(float(pi), eps)
        kl += ai * torch.log(torch.tensor(ai / pi)).item()
    return float(kl)


def compute_compute_cost(training_batch):
    """
    Simple compute proxy c(training_batch):
    average character length, normalized.
    """
    if len(training_batch) == 0:
        return 0.0
    total_chars = 0
    for ex in training_batch:
        total_chars += len(str(ex.get("problem", "")))
        total_chars += len(str(ex.get("solution", "")))
    return (total_chars / len(training_batch)) / 1000.0


def discounted_returns(rewards, gamma):
    """
    q_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
    """
    q = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        q.append(running)
    q.reverse()
    return q


def reinforce_teacher_update(
    optimizer,
    log_probs,
    rewards,
    gamma,
    use_advantage_baseline=True,
    baseline_type="mean_return",
    normalize_advantage=True,
):
    """
    One REINFORCE update after an episode:
      grad sum_t log_prob_t * q_t
    """
    if len(log_probs) == 0:
        return 0.0

    q_t = discounted_returns(rewards, gamma)
    returns = torch.tensor(q_t, dtype=torch.float32, device=log_probs[0].device)

    # Variance reduction baseline (PoC: mean return baseline).
    if use_advantage_baseline:
        if baseline_type != "mean_return":
            raise ValueError("Only baseline_type='mean_return' is supported in this PoC.")
        baseline = returns.mean()
        advantages = returns - baseline
    else:
        advantages = returns

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_objective = 0.0
    for lp, adv in zip(log_probs, advantages):
        policy_objective = policy_objective + lp * adv

    loss = -policy_objective
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


