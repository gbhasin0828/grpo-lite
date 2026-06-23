# GRPO Take-Home Assignment — Solution Writeup

**Author:** Gaurav Bhasin  
**Model:** Qwen/Qwen2-0.5B-Instruct  
**Hardware:** 1x H100 NVL (Vast.ai), CUDA 13.0, Python 3.12  

========================================================================================================================

## Task 1A: Environment Setup

### What we changed and why

The original `requirements.txt` was a raw `pip freeze` dump from an Ubuntu system — 150+ packages including OS-level tools like `cloud-init`, `ufw`, and `python-apt` that have nothing to do with this project.

We replaced it with a clean `pyproject.toml` using `uv` — the modern Python package manager. Only the 7 packages the code actually imports are listed as dependencies:

| Package | Version | Why |
|---|---|---|
| `torch` | 2.12.0 | Deep learning framework |
| `transformers` | 4.48.2 | Load and run Qwen models |
| `datasets` | 3.2.0 | Download GSM8K from HuggingFace |
| `accelerate` | 1.3.0 | Required by transformers for GPU loading |
| `tqdm` | ≥4.67 | Progress bars |
| `numpy` | ≥1.24 | Numerical arrays |
| `matplotlib` | ≥3.5 | Training plots |

### PyTorch CUDA build

The default PyPI version of `torch` is CPU-only. We configured a custom index in `pyproject.toml` pointing to PyTorch's CUDA wheel server:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

This ensures `torch` is installed with CUDA 13.0 support automatically when anyone runs `uv sync`.

### Flash Attention 2

No prebuilt wheel exists for this environment (Python 3.12 + CUDA 13.0 + PyTorch 2.12). Building from source takes 45-60 minutes and failed in this environment.

**Decision:** Modified `llms.py` to use `attn_implementation="eager"` instead. This is slightly slower than Flash Attention 2 but functionally identical for training correctness. Once a prebuilt wheel is available for this environment, switching back is a one-line change.

### How to set up

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

========================================================================================================================

---

## Task 1B: Bug Fixes in `evaluator.py`

### What does `evaluator.py` do?

During GRPO training, the model generates 16 different responses to the same math question. The evaluator scores each response across 5 dimensions and returns a `[16 × 5]` reward tensor. These scores are the entire training signal — the model learns to do more of what gets high scores and less of what gets low scores.

The 5 reward functions are:

| # | Function | What it measures |
|---|---|---|
| 1 | Correctness | Did the model get the right answer? |
| 2 | Integer Format | Is the answer a clean integer? |
| 3 | Strict Format | Does the response follow exact newline formatting? |
| 4 | Soft Format | Are the tags present and in the right order? |
| 5 | XML Count | Partial credit for having individual tags present |

### Reward Design Philosophy

We adopted one consistent rule across all reward functions:

| Situation | Score |
|---|---|
| Did the right thing | positive |
| Did the wrong thing | negative |
| Did not attempt at all | `0.0` |

This creates a clear 3-way training signal. The model can always distinguish between "I got it right", "I got it wrong", and "I didn't try".

---

### Bug 1: Wrong answers received no penalty

**Where:** `_correctness_reward`

**Problem:** Wrong answers got `0.0` — same as no attempt. The model couldn't distinguish between getting it wrong and not trying at all. Training signal was very weak.

**Fix:** Wrong answers now get `-1.0`. This creates a clear `3.0` point gap between correct (`2.0`) and wrong (`-1.0`), with no attempt sitting at `0.0` in between.

```python
# Before (broken)
rewards.append(2.0 if pred_i == gt_i else 0.0)

# After (fixed)
rewards.append(2.0 if pred_i == gt_i else -1.0)
```

---

### Bug 2: Word numbers not recognized as correct answers

**Where:** `_parse_int`

**Problem:** If the model wrote `"thirteen"` instead of `"13"`, the original code couldn't parse it and gave `-1.0` correctness reward — even though the answer was semantically correct.

**Fix:** Added `word2number` library as a fallback in `_parse_int`. Now `"thirteen"` correctly parses to `13` and gets `2.0` correctness reward. The integer format penalty (`-0.5`) still applies — teaching the model to use digits not words.

```python
# Try word to number e.g. "thirteen" -> 13
try:
    return int(w2n.word_to_num(s))
except:
    return None
```

---

### Bug 3: No answer tag — no intelligent fallback

**Where:** `_correctness_reward` and `_extract_raw_answer`

**Problem:** When the model responded without `<answer>` tags, the original code returned an empty string and gave `0.0` — even if the correct answer was clearly stated in the response text.

**Fix:** Added a 3-strategy fallback in `_extract_raw_answer` to intelligently extract the answer from raw text:

| Strategy | Example Input | Extracted |
|---|---|---|
| 1. Number after "answer/result" | `"the answer is 50"` | `"50"` |
| 2. Word number after "answer/result" | `"the answer is fifty"` | `"50"` |
| 3. Last resort — last number in text | `"there are 13 left"` | `"13"` |

Correctness scoring now applies the same logic regardless of whether tags are present — correct gets `2.0`, wrong gets `-1.0`, truly no answer gets `0.0`.

---

### Bug 4: Wrong format received no penalty

**Where:** `_int_format_reward` and `_strict_format_reward`

**Problem:** Both functions returned `0.0` for wrong format — same issue as Bug 1. Empty responses were treated identically to wrong format responses, violating our reward philosophy.

**Fix:**

`_int_format_reward` now has 3 outcomes:
- Answer is an integer → `0.5`
- Answer found but not an integer → `-0.5`  
- No answer found at all → `0.0` (neutral — didn't attempt)

`_strict_format_reward`:
- Correct strict format → `0.5`
- Wrong format → `-0.5`

---

### Bug 5: Soft format only handled 2 cases

**Where:** `_soft_format_reward`

**Problem:** The original code had only two outcomes — tags in right order (`0.5`) or everything else (`0.0`). A model that put `<answer>` before `<reasoning>`, produced garbage text, or produced nothing at all — all got the same `0.0`. No distinction between trying-but-wrong and not-trying-at-all.

**Fix:** Added `REVERSE_RE` regex and now handle 4 distinct cases:

| Situation | Score | Reason |
|---|---|---|
| `<reasoning>` then `<answer>` | `0.5` | Correct format |
| `<answer>` then `<reasoning>` | `0.25` | Wrong order but tried |
| Empty response | `0.0` | Didn't attempt → neutral |
| No tags, non-empty response | `-0.5` | Ignored format instructions |

---

### Impact of Bug Fixes

| Response Type | OLD Total | NEW Total | Change |
|---|---|---|---|
| Perfect response | 4.00 | 4.00 | No change |
| Wrong answer, correct format | 2.00 | 1.00 | Correctness penalty |
| "thirteen" as correct answer | 1.00 | 2.00 | word2number fix |
| "twelve" as wrong answer | 1.00 | -1.00 | Correctness + format penalty |
| Answer before reasoning | 2.85 | 2.60 | Partial soft format credit |
| Empty response | 0.00 | -2.50 | Penalties across all functions |


========================================================================================================================

---

## Task 2A: Implementing `compute_loss`

### What is `compute_loss` trying to do?

During GRPO training, for every math question we generate 16 completions. The evaluator scores each one. Now we need to tell the model:

**"Do more of what got high scores, less of what got low scores — but don't change too drastically."**

`compute_loss` translates that into a single scalar number that PyTorch uses to update the model weights via backpropagation and gradient descent.

---

### Worked Example

To make the implementation concrete, we walk through all 7 steps using 4 completions for the question:

**"Janet has 16 eggs. She eats 3 for breakfast. How many are left?"** (Answer: 13)

C1: "<reasoning>\n16 - 3 = 13\n</reasoning>\n<answer>\n13\n</answer>"         → 15 tokens, correct

C2: "<reasoning>\n16 minus 3 equals 13\n</reasoning>\n<answer>\n13\n</answer>" → 15 tokens, correct

C3: "The answer is 13"                                                          → 4 tokens, no tags

C4: "<reasoning>\n16-3=13\n</reasoning>\n<answer>\n13\n</answer>\nHope this helps!" → 20 tokens, correct but junk after

---

### Step 1: Per-Token Log Probabilities from Current Model

The current model already generated these completions. We go back and ask: **"How confident were you about each token you generated?"**

At each position the model assigns probabilities to all 151,936 tokens in its vocabulary. We keep only the log probability of the token actually chosen.

For C1, the raw probabilities of each chosen token are:
C1 probs: [0.819, 0.905, 0.741, 0.819, 0.905, 0.819, 0.670, 0.905, 0.819, 0.905, 0.741, 0.905, 0.819, 0.905, 0.741, pad, pad, pad, pad, pad]

Taking the log:
per_token_logps C1: [-0.2, -0.1, -0.3, -0.2, -0.1, -0.2, -0.4, -0.1, -0.2, -0.1, -0.3, -0.1, -0.2, -0.1, -0.3, 0, 0, 0, 0, 0]

All 4 completions are padded to length 20 (longest completion). Shape: **[4 × 20]**

---

### Step 2: Per-Token Log Probabilities from Reference Model

Feed the exact same tokens to the frozen reference model. It never generates anything — it just evaluates how likely it would have rated each token.
ref_per_token_logps C1: [-0.33, -0.37, -0.44, -0.33, -0.22, -0.22, -0.48, -0.29, -0.27, -0.24, -0.39, -0.53, -0.38, -0.23, -0.75, 0, 0, 0, 0, 0]

The reference model is less confident (more negative values) — it hasn't been trained yet. Shape: **[4 × 20]**

---

### Step 3: Per-Token KL Divergence

KL divergence measures how much the current model has drifted from the reference model, token by token.

log_ratio = ref_per_token_logps - per_token_logps

kl        = exp(log_ratio) - log_ratio - 1

log_ratio = -0.33 - (-0.2) = -0.13

kl        = exp(-0.13) - (-0.13) - 1 = 0.878 + 0.13 - 1 = 0.008

KL is always positive. Zero when models are identical. Grows as models drift apart. Shape: **[4 × 20]**

---

### Step 4: Compute Advantages

The evaluator scores each completion across 5 reward functions returning **[4 × 5]**:

Correct  IntFmt  Strict  Soft   XML    Total
C1:       [ 2.0,    0.5,    0.5,   0.5,   0.5]  = 4.0

C2:       [ 2.0,    0.5,   -0.5,   0.5,   0.5]  = 3.0

C3:       [ 2.0,   -0.5,   -0.5,  -0.5,   0.1]  = 0.6

C4:       [ 2.0,    0.5,   -0.5,   0.5,   0.3]  = 2.8

Sum across reward functions → `rewards = [4.0, 3.0, 0.6, 2.8]`

Compute mean and std, then normalize:
mean = 2.6,  std = 1.27
advantages = (rewards - mean) / std

C1:  1.10  ← much better than average

C2:  0.31  ← slightly better than average

C3: -1.57  ← much worse than average

C4:  0.16  ← slightly better than average

Shape: **[4 × 1]**. In real training with 16 completions: **[16 × 1]**.

---

### Step 5: Per-Token Policy Objective

ratio            = exp(per_token_logps - ref_per_token_logps)

policy_objective = ratio × advantages

The ratio measures how much the current model has changed from reference for each token. Multiplying by advantages scales the update — good completions get strongly reinforced, bad completions get strongly discouraged.

For C1 token 1:
ratio            = exp(-0.2 - (-0.33)) = exp(0.13) = 1.14
policy_objective = 1.14 × 1.10 = 1.25

For C3 token 1 (bad completion, advantage = -1.57):
ratio            = 1.22
policy_objective = 1.22 × (-1.57) = -1.92

Shape: **[4 × 20]**

---

### Step 6: Combine with KL Penalty

per_token_loss = -(policy_objective) + beta × kl

Where `beta = 0.04`. The negative sign is because PyTorch minimizes loss but we want to maximize the policy objective.

For C1 token 1 (good completion):
per_token_loss = -(1.25) + 0.04 × 0.008 = -1.2497  ← negative → reinforced ✅

For C3 token 1 (bad completion):
per_token_loss = -(-1.92) + 0.04 × 0.019 = 1.9208  ← positive → discouraged ✅

The KL penalty prevents the model from drifting too far from the reference even when chasing high rewards. Shape: **[4 × 20]**

---

### Step 7: Average Down to One Scalar

```python
# Apply mask to zero out padding
masked = per_token_loss * completion_mask

# Sum over tokens, normalize by completion length
per_completion_loss = masked.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)

# Average over all completions
loss = per_completion_loss.mean()
```

For our example:
C1: -19.71 / 15 = -1.314

C2: -5.10  / 15 = -0.340

C3:  7.68  /  4 =  1.920

C4: -3.66  / 20 = -0.183
loss = (-1.314 + (-0.340) + 1.920 + (-0.183)) / 4 = 0.021

PyTorch then calls `loss.backward()` and `optimizer.step()` to update the model weights.

---

### Shape Summary

per_token_logps        [4 × 20]  →  [16 × max_len] in real training

ref_per_token_logps    [4 × 20]

log_ratio              [4 × 20]

kl                     [4 × 20]

ratio                  [4 × 20]

advantages             [4 × 1]

policy_objective       [4 × 20]

per_token_loss         [4 × 20]

masked_loss            [4 × 20]

sum per completion     [4 × 1]

normalized by length   [4 × 1]

final loss             scalar

