# Deep RL Training Diagnostic Metrics: Research Compendium

## 1. Observation Space Quality Metrics

### 1.1 Effective Dimension / Effective Rank (eRank)

**Paper:** Roy & Bhatt, "The Effective Rank: A Measure of Effective Dimensionality" (EUSIPCO 2007)
**Applied to RL:** Kumar et al., "Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning" (ICLR 2021)

**Formula:**
```
erank(A) = exp( H(p_1, p_2, ..., p_Q) )

where:
  H(p_1, ..., p_Q) = -sum_k( p_k * log(p_k) )     # Shannon entropy
  p_k = sigma_k / ||sigma||_1                        # normalized singular values
  sigma_k = k-th singular value of A
  ||sigma||_1 = sum of all singular values
```

**Properties:**
- 1 <= erank(A) <= rank(A) <= n
- When features are orthogonal: erank ~ d (full utilization)
- When features collapse to subspace: erank << d (aliasing)

**Implementation (PyTorch):**
```python
def effective_rank(feature_matrix):
    """feature_matrix: (batch, feature_dim)"""
    # SVD
    _, S, _ = torch.linalg.svd(feature_matrix, full_matrices=False)
    # Normalize singular values to form a distribution
    p = S / S.sum()
    # Shannon entropy
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy)
```

**Connection to RL:** Kumar et al. showed that in value-based RL with bootstrapping, the effective rank of the penultimate-layer feature matrix drops steeply during training, directly correlating with performance collapse. The feature matrix Phi satisfies Q(s,a) = w^T * Phi(s,a).

---

### 1.2 Stable Rank

**Formula:**
```
stable_rank(W) = ||W||_F^2 / ||W||_2^2

where:
  ||W||_F = Frobenius norm = sqrt(sum of squared entries) = sqrt(sum of sigma_i^2)
  ||W||_2 = spectral norm = largest singular value sigma_1
```

**Interpretation:** Stable rank is small when a matrix has few dominant eigenvectors. For weight matrices in deep RL, low stable rank indicates the network has collapsed its representation.

**Implementation (PyTorch):**
```python
def stable_rank(weight_matrix):
    S = torch.linalg.svdvals(weight_matrix)
    frobenius_sq = (S ** 2).sum()
    spectral_sq = S[0] ** 2
    return frobenius_sq / spectral_sq
```

---

### 1.3 Information Bottleneck for RL Observations

**Paper:** Yavari et al., "Learning Representations in Reinforcement Learning: An Information Bottleneck Approach" (2019, arXiv:1911.05695)

**Core idea:** Compress observation X into representation Z such that:
```
min  I(X; Z) - beta * I(Z; Y)

where:
  I(X; Z) = mutual information between observation and representation (compression)
  I(Z; Y) = mutual information between representation and task-relevant target Y
  beta = trade-off parameter
```

Two complementary regularizers:
- **State representation bottleneck**: enforces compact, task-relevant embeddings
- **Policy bottleneck**: constrains I(S; A) — actions should be as uninformative about state as possible while maximizing return

Applied to A2C and PPO, this significantly improves sample efficiency.

---

### 1.4 Mutual Information Between Observations and Actions: MI(S; A)

**Paper:** "Mutual Information Tracks Policy Coherence in Reinforcement Learning" (arXiv:2509.10423)

**Formula:**
```
MI(S; A) = H(A) - H(A|S)

where:
  H(A) = action entropy (marginal)
  H(A|S) = conditional entropy of actions given states
```

**Diagnostic interpretation:**
- MI(S;A) increasing despite growing state entropy = policy is learning state-dependent behavior
- MI(S;A) stagnating or decreasing = policy ignoring observations
- Empirically, MI(S;A) increased from 29.8% to 37.3% of state entropy during successful training
- Provides early warning of misalignment before performance visibly degrades

---

### 1.5 Gradient-Based Saliency for Observation Importance

**Paper:** "Are Gradient-based Saliency Maps Useful in Deep Reinforcement Learning?" (arXiv:2012.01281)

**Methods:**
1. **Vanilla Gradient:** saliency_i = |dQ/dx_i| or |d(log pi)/dx_i|
2. **Smoothed Gradients:** Average over N perturbations: saliency_i = (1/N) * sum |dQ/d(x+noise)_i|
3. **Integrated Gradients:** Accumulate along path from baseline: saliency_i = (x_i - x_baseline_i) * integral_0^1 (dF/dx_i at x_baseline + t*(x - x_baseline)) dt

**Implementation (PyTorch):**
```python
def vanilla_gradient_saliency(model, obs):
    obs.requires_grad_(True)
    output = model(obs)
    output.sum().backward()
    return obs.grad.abs().mean(dim=0)  # per-feature importance
```

**Caveat:** Gradients measure sensitivity, not necessarily importance. Integrated gradients are more reliable but slower.

---

### 1.6 Feature Utilization via Conditional Mutual Information

**Paper:** Nguyen et al., "Feature Selection for Reinforcement Learning: Evaluating Implicit State-Reward Dependency via Conditional Mutual Information" (ECML 2010)

Key metric: CMI(feature_i; reward | other_features) — if near zero, feature_i is redundant.

---

## 2. Value Function Quality Metrics

### 2.1 Explained Variance of Returns

**Used in:** Stable Baselines3, CleanRL, SKRL

**Formula:**
```
explained_variance = 1 - Var(y_true - y_pred) / Var(y_true)

where:
  y_true = actual returns (discounted sum of rewards)
  y_pred = value function predictions V(s)
```

**Interpretation:**
- ev = 1.0: perfect prediction
- ev = 0.0: might as well have predicted zero
- ev < 0.0: worse than predicting zero (broken value function)
- Healthy PPO training: ev should increase toward 0.5-0.9 over training

**Implementation:**
```python
def explained_variance(y_pred, y_true):
    var_y = y_true.var()
    if var_y == 0:
        return float('nan')
    return 1.0 - (y_true - y_pred).var() / var_y
```

---

### 2.2 TD Error Distribution Analysis

**Formula:**
```
delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)
```

**Diagnostic metrics to compute:**
- Mean |delta|: average prediction error magnitude
- Std(delta): prediction uncertainty
- Skewness(delta): positive skew = systematic under-estimation; negative = over-estimation
- Kurtosis(delta): heavy tails indicate occasional large mispredictions
- Fraction of delta > 0 vs delta < 0: bias detection

**Implementation:**
```python
def td_error_diagnostics(td_errors):
    return {
        'td_mean_abs': td_errors.abs().mean(),
        'td_std': td_errors.std(),
        'td_skew': ((td_errors - td_errors.mean())**3).mean() / (td_errors.std()**3 + 1e-8),
        'td_positive_frac': (td_errors > 0).float().mean(),
    }
```

---

### 2.3 Value Function Collapse / Capacity Loss

**Paper:** Lyle et al., "Understanding and Preventing Capacity Loss in Reinforcement Learning" (ICLR 2022, OpenReview: ZkC8wKoLbQ7)

**Definition:** Capacity loss = networks trained on a sequence of target values lose their ability to quickly fit new functions over time.

**Detection metrics:**
- Explained variance suddenly dropping
- Value loss plateauing while reward is still changing
- Feature rank of critic's penultimate layer decreasing
- Dead neuron count in critic increasing

---

### 2.4 PPO-Specific Value Diagnostics

**From CleanRL "37 Implementation Details":**

| Metric | Formula | Healthy Range |
|--------|---------|--------------|
| value_loss | MSE(V(s), returns) | decreasing trend |
| explained_variance | 1 - Var(returns - V)/Var(returns) | > 0, trending to 0.5-0.9 |
| approx_kl (k3) | ((ratio-1) - log(ratio)).mean() | < 0.02 |
| clip_fraction | (|ratio - 1| > epsilon).float().mean() | 10-30% early, < 10% late |
| entropy_loss | -sum(pi * log(pi)) | slowly decreasing |

---

## 3. Policy Quality Metrics

### 3.1 Action Entropy

**Formula:**
```
H(pi) = -sum_a pi(a|s) * log(pi(a|s))

# For continuous: differential entropy
# For discrete: Shannon entropy
```

**Per-region analysis:**
- Compute entropy separately for "near-obstacle" vs "open-space" observations
- Low entropy near obstacles = policy is decisive (good or stuck)
- High entropy near obstacles = policy is confused
- Entropy collapsing to 0 everywhere = premature convergence

---

### 3.2 Policy Entropy Collapse Detection

**Paper:** "Policy Entropy Collapse in RL" (emergentmind.com topic survey)

**Detection:**
```
# Monitor entropy over training
if entropy_t < 0.1 * entropy_0:  # dropped to <10% of initial
    warn("entropy collapse detected")

# Or monitor entropy rate of change
d_entropy/d_step < threshold  # entropy dropping too fast
```

---

### 3.3 Gradient Signal-to-Noise Ratio (SNR)

**Paper:** Roberts & Tedrake, "Signal-to-Noise Ratio Analysis of Policy Gradient Algorithms" (NeurIPS 2008)
**Recent:** "Non-Uniform Noise-to-Signal Ratio in the REINFORCE Policy-Gradient Estimator" (arXiv:2602.01460, Feb 2026)

**Formula:**
```
SNR = ||E[g]||^2 / Var(g)

where:
  g = policy gradient estimate
  E[g] = true gradient (estimated by averaging over many samples)
  Var(g) = variance of gradient estimates

# Practical approximation: over a mini-batch
SNR_approx = ||mean(g_batch)||^2 / var(g_batch)
```

**Key findings:**
- SNR is a good predictor of long-term learning performance
- Near optimum, NSR (noise-to-signal ratio = 1/SNR) increases, causing unreliable gradients
- Cost-to-go baseline is optimal for SNR in episodic settings

**Implementation:**
```python
def gradient_snr(model, loss_fn, batch, n_samples=10):
    grads = []
    for _ in range(n_samples):
        loss = loss_fn(model, batch)
        loss.backward()
        flat_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(flat_grad.clone())
        model.zero_grad()
    grads = torch.stack(grads)
    signal = grads.mean(0).norm() ** 2
    noise = grads.var(0).sum()
    return (signal / (noise + 1e-8)).item()
```

---

### 3.4 Approximate KL Divergence

**From PPO implementations:**
```python
# k1 estimator (biased):
approx_kl_k1 = (-logratio).mean()

# k3 estimator (unbiased, less variance):
approx_kl_k3 = ((ratio - 1) - logratio).mean()

# where: logratio = log_prob_new - log_prob_old
#         ratio = exp(logratio)
```
**Healthy range:** < 0.02. If > 0.02, policy is changing too quickly.

---

### 3.5 Clip Fraction

```python
clip_fraction = (torch.abs(ratio - 1.0) > clip_coef).float().mean()
```
**Interpretation:** Starts ~20%, should decrease below 10% as training stabilizes.

---

## 4. Training Health Metrics (Beyond Grad Norms)

### 4.1 Dormant Neuron Score & ReDo

**Paper:** Sokar et al., "The Dormant Neuron Phenomenon in Deep Reinforcement Learning" (ICML 2023, arXiv:2302.12902)

**Dormancy score formula:**
```
s_i^l = E_{x in D}[|h_i^l(x)|] / ( (1/H^l) * sum_k E_{x in D}[|h_k^l(x)|] )

where:
  h_i^l(x) = activation of neuron i in layer l for input x
  H^l = number of neurons in layer l
```

**Definition:** Neuron i is tau-dormant if s_i^l <= tau

**Recommended thresholds:**
- tau = 0.025 (default)
- tau = 0.1 (benchmarking)
- tau = 0.0 (strict: completely dead)

**Implementation (PyTorch):**
```python
def dormant_neuron_ratio(activations, tau=0.1):
    """activations: (batch, neurons) — output of a hidden layer"""
    mean_abs = activations.abs().mean(dim=0)       # per-neuron mean |activation|
    layer_mean = mean_abs.mean()                    # average across layer
    scores = mean_abs / (layer_mean + 1e-8)         # normalized score
    dormant = (scores <= tau).float().mean()
    return dormant.item()
```

**ReDo Algorithm:**
Every F steps (F=1000 for DQN, F=200K for SAC):
1. Compute dormancy scores for all neurons
2. For each tau-dormant neuron:
   - Reinitialize incoming weights (from original distribution)
   - Set outgoing weights to 0 (preserve network output)

**PyTorch implementation:** https://github.com/timoklein/redo

---

### 4.2 Dead Neuron Ratio (Dead ReLU)

**Paper:** Dohare et al., "Loss of Plasticity in Deep Continual Learning" (Nature 2024)

**Definition:** A ReLU neuron is "dead" if it outputs 0 for ALL inputs in a batch:
```
dead_ratio = count(neurons where max(activation) == 0 across batch) / total_neurons
```

**Key findings:**
- With step size 0.01: up to 25% of units die after 800 tasks
- Long-term: up to 90% of units can die
- Only continual backpropagation maintains ~0% dead units

**Implementation:**
```python
def dead_neuron_ratio(activations):
    """activations: (batch, neurons) after ReLU"""
    max_per_neuron = activations.max(dim=0).values
    dead = (max_per_neuron == 0).float().mean()
    return dead.item()
```

---

### 4.3 Weight Magnitude Growth

**Paper:** Dohare et al. (Nature 2024)

**Formula:**
```
avg_weight_magnitude = sum(|w_i|) / total_number_of_weights
```

**Interpretation:** Increasing weight magnitude correlates with plasticity loss. Methods that keep weights small (L2 reg, Shrink+Perturb, continual backprop) maintain plasticity.

**Implementation:**
```python
def avg_weight_magnitude(model):
    total_abs = sum(p.abs().sum() for p in model.parameters())
    total_count = sum(p.numel() for p in model.parameters())
    return (total_abs / total_count).item()
```

---

### 4.4 Feature Correlation / Redundancy

**Paper:** Lee et al., "On the Importance of Feature Decorrelation for Unsupervised Representation Learning in RL" (ICML 2023, arXiv:2306.05637)
**Also:** "Deep Reinforcement Learning with Decorrelation" (arXiv:1903.07765)

**Metric:** Pearson correlation matrix of feature activations
```
C_ij = Cov(f_i, f_j) / (std(f_i) * std(f_j))

# Aggregate metric:
mean_off_diagonal_correlation = (sum |C_ij| for i != j) / (d * (d-1))
```

**Finding:** Features in baseline DQN are highly correlated (redundant). Decorrelation regularization improves performance.

**Implementation:**
```python
def feature_correlation(features):
    """features: (batch, dim)"""
    # Standardize
    f = features - features.mean(dim=0)
    f = f / (f.std(dim=0) + 1e-8)
    # Correlation matrix
    corr = (f.T @ f) / (f.shape[0] - 1)
    # Mean off-diagonal absolute correlation
    d = corr.shape[0]
    mask = ~torch.eye(d, dtype=bool, device=corr.device)
    return corr.abs()[mask].mean().item()
```

---

### 4.5 Plasticity Loss: Comprehensive Metrics Summary

**Paper:** "Plasticity Loss in Deep Reinforcement Learning: A Survey" (arXiv:2411.04832, Nov 2024)
**Paper:** "Disentangling the Causes of Plasticity Loss in Neural Networks" (arXiv:2402.18762)

**Complete metric battery:**

| Metric | Formula | What it detects |
|--------|---------|-----------------|
| Dead units (%) | fraction of neurons with max(activation) = 0 | Gradient death |
| Dormant neurons (%) | fraction with normalized score <= tau | Near-death neurons |
| Weight magnitude | mean(|w|) across all params | Weight explosion |
| Stable rank (weights) | ||W||_F^2 / ||W||_2^2 per layer | Weight rank collapse |
| Effective rank (features) | exp(H(normalized_singular_values)) | Feature collapse |
| Feature correlation | mean |off-diagonal| of corr matrix | Redundancy |
| eNTK rank | rank of gradient dot-product matrix | Optimization difficulty |

**"Zombie units"** (from Lyle et al., arXiv:2402.18762): ReLU units with only positive inputs that behave like identity functions — a new pathology not previously studied.

---

### 4.6 Mitigation Strategies Ranked by Effectiveness (On-Policy PPO)

From "A Study of Plasticity Loss in On-Policy Deep RL" (NeurIPS 2024):

1. **Soft Shrink+Perturb**: x_new = alpha * x_current + beta * x_init (best)
2. **LayerNorm**: Normalizes activations; works well with regularization
3. **ReDo**: Recycle dormant neurons periodically
4. **L2 regularization**: toward zero (not initialization)
5. **Continual Backpropagation**: reinitialize fraction of least-used units

Key insight: constraining weight deviation from initialization distribution > architectural modifications for on-policy PPO.

---

## 5. Summary: Recommended Diagnostic Dashboard

For a PPO-based navigation RL agent, the recommended metrics to log per training epoch:

### Tier 1 (Essential, cheap to compute):
- `explained_variance` — value function quality
- `approx_kl` (k3 estimator) — policy stability, target < 0.02
- `clip_fraction` — trust region violation rate
- `entropy` — exploration level
- `dead_neuron_ratio` — per layer, fraction of always-zero neurons
- `avg_weight_magnitude` — plasticity health

### Tier 2 (Important, moderate cost):
- `dormant_neuron_ratio(tau=0.1)` — per layer
- `feature_effective_rank` — penultimate layer feature matrix erank
- `feature_correlation` — mean off-diagonal |correlation|
- `td_error_stats` — mean, std, skew of TD errors
- `stable_rank` — per weight matrix

### Tier 3 (Deep diagnostics, compute-heavy):
- `gradient_snr` — policy gradient signal-to-noise
- `gradient_saliency` — per observation dimension importance
- `MI(S;A)` — mutual information between states and actions
- `observation_effective_dimension` — erank of observation batch

---

## Key Paper References

1. **Dormant neurons:** Sokar et al., "The Dormant Neuron Phenomenon in Deep RL" (ICML 2023) — https://arxiv.org/abs/2302.12902
2. **Plasticity loss (Nature):** Dohare et al., "Loss of Plasticity in Deep Continual Learning" (Nature 2024) — https://www.nature.com/articles/s41586-024-07711-7
3. **Plasticity survey:** "Plasticity Loss in Deep RL: A Survey" (Nov 2024) — https://arxiv.org/abs/2411.04832
4. **On-policy plasticity:** "A Study of Plasticity Loss in On-Policy Deep RL" (NeurIPS 2024) — https://arxiv.org/abs/2405.19153
5. **Implicit under-param:** Kumar et al., "Implicit Under-Parameterization Inhibits Data-Efficient Deep RL" (ICLR 2021) — https://openreview.net/forum?id=O9bnihsFfXU
6. **Effective rank:** Roy & Bhatt, "The Effective Rank: A Measure of Effective Dimensionality" (EUSIPCO 2007)
7. **Feature decorrelation:** Lee et al., "On the Importance of Feature Decorrelation for Unsupervised Representation Learning in RL" (ICML 2023) — https://arxiv.org/abs/2306.05637
8. **DR3 regularization:** Kumar et al., "DR3: Value-Based Deep RL Requires Explicit Regularization" (ICLR 2022) — https://arxiv.org/abs/2112.04716
9. **Gradient SNR:** Roberts & Tedrake, "Signal-to-Noise Ratio Analysis of Policy Gradient Algorithms" (NeurIPS 2008) — https://proceedings.neurips.cc/paper/2008/hash/8df707a948fac1b4a0f97aa554886ec8-Abstract.html
10. **Non-uniform NSR:** "Non-Uniform Noise-to-Signal Ratio in REINFORCE" (Feb 2026) — https://arxiv.org/abs/2602.01460
11. **Info bottleneck RL:** Yavari et al., "Learning Representations in RL: An Information Bottleneck Approach" (2019) — https://arxiv.org/abs/1911.05695
12. **MI tracks coherence:** "Mutual Information Tracks Policy Coherence in RL" (2025) — https://arxiv.org/pdf/2509.10423
13. **Saliency in RL:** "Are Gradient-based Saliency Maps Useful in Deep RL?" — https://arxiv.org/abs/2012.01281
14. **Capacity loss:** Lyle et al., "Understanding and Preventing Capacity Loss in RL" (ICLR 2022) — https://openreview.net/forum?id=ZkC8wKoLbQ7
15. **Disentangling plasticity:** Lyle et al., "Disentangling the Causes of Plasticity Loss in Neural Networks" (2024) — https://arxiv.org/abs/2402.18762
16. **Representation properties:** Wang et al., "Investigating the Properties of Neural Network Representations in RL" (AI Journal 2024) — https://arxiv.org/abs/2203.15955
17. **PPO implementation details:** Huang et al., "The 37 Implementation Details of PPO" (ICLR Blog 2022) — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
18. **ReDo code:** PyTorch implementation — https://github.com/timoklein/redo
19. **Plasticity code:** Official Nature paper code — https://github.com/shibhansh/loss-of-plasticity
20. **Churn & plasticity:** "Mitigating Plasticity Loss by Reducing Churn" (ICML 2025) — https://openreview.net/forum?id=EkoFXfSauv
