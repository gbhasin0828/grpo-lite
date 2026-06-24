## Implementation Notes

### Task 1A: Environment Setup

**Dependency Management:**
Converted `requirements.txt` to `pyproject.toml` using `uv`. The original `requirements.txt` was a raw `pip freeze` dump containing ~150 system-level packages. Trimmed to the 7 packages the code actually imports: `torch`, `transformers`, `datasets`, `accelerate`, `tqdm`, `numpy`, `matplotlib`.

**PyTorch Installation:**
Configured a custom PyTorch index (`https://download.pytorch.org/whl/cu130`) to ensure the CUDA-enabled build is installed rather than the CPU-only PyPI version.

**Flash Attention 2:**
No prebuilt wheel exists for the environment on the provisioned H100 (Python 3.12, CUDA 13.0, PyTorch 2.12). Building from source takes 45-60 minutes and failed in this environment. Modified `llms.py` to use `attn_implementation="eager"` as a pragmatic alternative. This is slightly slower but functionally equivalent for correctness.

**Verified on:** 1x H100 NVL (Vast.ai), CUDA 13.0, Python 3.12

==============================================================================================================

### Task 1B: Bug Fixes in evaluator.py

#### What does evaluator.py do?

During GRPO training, the model generates 16 different responses to the 
same math question. The evaluator scores each response across 5 dimensions 
and returns a [16 x 5] reward tensor. These scores drive the training signal 
— the model learns to do more of what gets high scores and less of what gets 
low scores.

The 5 reward functions are:
1. **Correctness** — did the model get the right answer?
2. **Integer Format** — is the answer expressed as an integer?
3. **Strict Format** — does the response follow exact newline formatting?
4. **Soft Format** — are the tags present and in the right order?
5. **XML Count** — partial credit for having individual tags present

---

#### Our Reward Design Philosophy

We adopted one simple, consistent rule across all reward functions:

| Situation | Score |
|---|---|
| Did the right thing | positive |
| Did the wrong thing | negative |
| Did not attempt / tag absent | 0.0 |

This creates a clear training signal — the model knows exactly what 
is good, what is bad, and what is neutral.

---

#### Bug 1: Wrong answers received no penalty

**Where:** `_correctness_reward`

**Problem:** A wrong answer was getting `0.0`. This means the model 
could not distinguish between "I got it wrong" and "I did nothing". 
The training signal was very weak.

**Fix:** Wrong answers now get `-1.0`. Combined with `2.0` for correct 
answers, the model now has a clear `3.0` point gap between right and wrong.
If the model does not return an answer then `0.0` this creates a clear segregation between correct answer vs No answer vs Wrong Answer


---

#### Bug 2: Word numbers were not recognized as correct answers

**Where:** `_parse_int` and `_correctness_reward`

**Problem:** If the model wrote `"thirteen"` instead of `"13"`, 
the original code couldn't parse it and gave `0.0` correctness reward — 
even though the answer was semantically correct.

**Fix:** Added `word2number` library to `_parse_int` as a fallback. 
Now `"thirteen"` correctly parses to `13` and gets `2.0` correctness reward. 
The integer format penalty (`-0.5`) still applies — teaching the model 
to use digits not words.

---

#### Bug 3: Correctness when no answer tag present

**Where:** `_correctness_reward`

**Problem:** When the model responded without `<answer>` tags, the 
original code compared an empty string against the ground truth and 
gave `0.0`. This was neither a penalty nor a reward.

**Fix:** We made this explicit — if no <answer> tag is found, 
the correctness score should depend on the actual answer (Same logic as in _correctness_reward - If answer is correct - +ve, if no answer present - zero, if wrong answer - -ve). 
The format reward functions handle the penalty for missing tags separately.

RIGHT NOW THIS IS WRONG ---- NEEDS FIXING

---

#### Bug 4: Wrong format received no penalty

**Where:** `_int_format_reward` and `_strict_format_reward`

**Problem:** Both functions returned `0.0` for wrong format. 
Same issue as Bug 1 — `0.0` teaches nothing. Additionally, empty 
responses were being treated the same as wrong format responses, 
which violates our reward philosophy (no attempt should be neutral, 
not penalized).

**Fix:**
- `_int_format_reward`: 
  - Answer is an integer → `0.5`
  - Answer found but not an integer → `-0.5`
  - No answer found at all → `0.0` (neutral — didn't attempt)
- `_strict_format_reward`: wrong format now gets `-0.5`

---

#### Bug 5: Soft format only handled 2 cases

**Where:** `_soft_format_reward`

**Problem:** The original code only had two outcomes:
- Tags in right order → `0.5`
- Everything else → `0.0`

This meant a model that put `<answer>` before `<reasoning>`, 
a model that produced garbage text, and a model that produced 
an empty response all got the same score of `0.0`. No distinction 
between trying-but-wrong and not-trying-at-all.

**Fix:** We added a new `REVERSE_RE` regex and now handle 4 cases:

| Situation | Score | Reason |
|---|---|---|
| `<reasoning>` then `<answer>` | `0.5` | Correct format |
| `<answer>` then `<reasoning>` | `0.25` | Wrong order but tried |
| Empty response | `0.0` | Didn't attempt → neutral |
| No tags, non-empty | `-0.5` | Ignored format instructions |



---

#### Before vs After (same 16 completions)

| Description | OLD Total | NEW Total | Why |
|---|---|---|---|
| Perfect response | 4.00 | 4.00 | No change |
| Wrong answer, correct format | 2.00 | 1.00 | Correctness penalty added |
| "thirteen" correct answer | 1.00 | 2.00 | word2number fix |
| "twelve" wrong answer | 1.00 | -1.00 | Both correctness + format penalty |
| Answer then reasoning | 2.85 | 2.60 | Partial soft format credit |
| Empty response | 0.00 | -2.50 | Penalties across all functions |


==============================================================================================================

### Task 2: (Core) Implement the Loss Function & Train a Model

#### What is compute_loss trying to do?

During GRPO training, for every math question we generate multiple completions (16 in real training, 4 in the below example just for my understanding). The evaluator scores each completion. Now we need to tell the model:
"Do more of what got high scores, less of what got low scores — but don't change too drastically."
compute_loss translates that into a single number that PyTorch uses to update the model weights via backpropagation and gradient descent.

#### The 4 completions used for this example are 
C1: "<reasoning>\n16 - 3 = 13\n</reasoning>\n<answer>\n13\n</answer>"
C2: "<reasoning>\n16 minus 3 equals 13\n</reasoning>\n<answer>\n13\n</answer>"
C3: "The answer is 13" (reasoning & answer tags missing)
C4: "<reasoning>\n16-3=13\n</reasoning>\n<answer>\n13\n</answer>\nHope this helps!"

#### Step 1: Get Per-Token Log Probabilities from Current Model
The current model already generated the above 4 completions. Now we go back and look at the probabilities of each (selected) token it generated at each position.
The token by token breakdown of C1 looks like 
C1: ["<reasoning>", "\n", "16", "-", "3", "=", "13", "\n", "</reasoning>", "\n", "<answer>", "\n", "13", "\n", "</answer>"]  → 15 tokens
Let's say in C1 it generated the following as probabilities for each selected token in that place
C1: [0.819, 0.905, 0.741, 0.819, 0.905, 0.819, 0.670, 0.905, 0.819, 0.905, 0.741, 0.905, 0.819, 0.905, 0.741, mask1, mask2, mask3, mask4, mask5] 
so for c1 the probabilities are as follows 

We then take the log of each selected tokens to compute our 1st metrics (Per-Token Log Probabilities) and it will look like 
Log C1: [-0.2, -0.1, -0.3, -0.2, -0.1, -0.2, -0.4, -0.1, -0.2, -0.1, -0.3, -0.1, -0.2, -0.1, -0.3, 0, 0, 0, 0, 0]

#### Step 2: Get Per-Token Log Probabilities from Reference Model
Just feed the output from cirrent model to reference and get probabilities for each position 
C1: ["<reasoning>", "\n", "16", "-", "3", "=", "13", "\n", "</reasoning>", "\n", "<answer>", "\n", "13", "\n", "</answer>"]  → 15 tokens
Corresponsing probabilities say are 
C1: [0.72, 0.6905, 0.641, 0.72, 0.805, 0.80, 0.620, 0.75, 0.76, 0.79, 0.674, 0.5905, 0.6819, 0.7905, 0.4741, mask1, mask2, mask3, mask4, mask5]
Now again take logs of above to get (ref_per_token_logps)
Log C1 (reference model) = [log(0.72), log(0.6905), log(0.641), ...]
Log C1 (reference model) = [-0.33, -0.37, -0.44, -0.33, -0.22, -0.22, -0.48, -0.29, -0.27, -0.24, -0.39, -0.53, -0.38, -0.23, -0.75, 0, 0, 0, 0, 0]

#### Step 3: Compute Per-Token KL Divergence
KL Divergence measures how much the current model has drifted from the reference model — token by token.
We first compute the log_ratio for each token:
log_ratio = ref_per_token_logps - per_token_logps

For C1:
[-0.13, -0.27, -0.14, -0.13, -0.12, -0.02, -0.08, -0.19, -0.07, -0.14, -0.09, -0.43, -0.18, -0.13, -0.45, 0, 0, 0, 0, 0]

All values are negative — meaning the current model is MORE confident than the reference model about every token. This makes sense — the current model has already trained for some steps.

We then compute KL for each token using:
kl = exp(log_ratio) - log_ratio - 1

For token 1 of C1 as an example:
log_ratio = -0.13

kl = exp(-0.13) - (-0.13) - 1
   = 0.878  + 0.13  - 1
   = 0.008

Final KL for C1:
kl C1: [0.008, 0.036, 0.010, 0.008, 0.007, 0.0002, 0.003, 0.018, 0.002, 0.010, 0.004, 0.092, 0.016, 0.008, 0.102, 0, 0, 0, 0, 0]

Shape of KL matrix for all 4 completions: [4 × 20]
V.IMPORTANT - In our actual exercise the shape will be [16 x 20] as for each prompt the model returns 16 responses.

#### Step 4: Compute Advantages
Before we compute the policy objective we need to know which completions were better or worse than average. This is called the advantage.
The evaluator scores each completion across 5 reward functions and returns a [4 × 5] tensor:
Correct  IntFmt  Strict  Soft   XML
C1:       [2.0,    0.5,    0.5,    0.5,   0.5]   → total = 4.0
C2:       [2.0,    0.5,   -0.5,    0.5,   0.5]   → total = 3.0
C3:       [2.0,   -0.5,   -0.5,   -0.5,   0.1]   → total = 0.6
C4:       [2.0,    0.5,   -0.5,    0.5,   0.3]   → total = 2.8

rewards = [4.0, 3.0, 0.6, 2.8]

Compute mean and standard deviation:
mean = (4.0 + 3.0 + 0.6 + 2.8) / 4 = 2.6
std  = 1.27

Compute advantage for each completion:
advantage = (reward - mean) / std

C1: (4.0 - 2.6) / 1.27 =  1.10  ← much better than average
C2: (3.0 - 2.6) / 1.27 =  0.31  ← slightly better than average
C3: (0.6 - 2.6) / 1.27 = -1.57  ← much worse than average
C4: (2.8 - 2.6) / 1.27 =  0.16  ← slightly better than average

Final advantages vector shape: [4 × 1]
advantages = [1.10, 0.31, -1.57, 0.16]

In real training with 16 completions this would be [16 × 1].

#### Step 5: Compute Per-Token Policy Objective

Now we combine the ratio (how much the current model has changed from reference) with the advantages (how good each completion was).
First compute the ratio for each token:
ratio = exp(per_token_logps - ref_per_token_logps)
      = exp(-log_ratio)


For C1 token 1:
ratio = exp(-(-0.13)) = exp(0.13) = 1.14


For C1 token 12 (largest drift):
ratio = exp(-(-0.43)) = exp(0.43) = 1.54


Full ratio for C1:
ratio C1: [1.14, 1.31, 1.15, 1.14, 1.13, 1.02, 1.08, 1.21, 1.07, 1.15, 1.09, 1.54, 1.20, 1.14, 1.57, 0, 0, 0, 0, 0]


Now multiply ratio by advantage for each completion.
Advantage for C1 = 1.10 — applied to every token position:
policy_objective C1 = ratio C1 × 1.10
= [1.25, 1.44, 1.26, 1.25, 1.24, 1.12, 1.19, 1.33, 1.18, 1.26, 1.20, 1.69, 1.32, 1.25, 1.73, 0, 0, 0, 0, 0]


Advantage for C3 = -1.57 — negative because bad completion:
policy_objective C3 = ratio C3 × (-1.57)
= [-1.92, -1.92, -1.92, -1.92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


Final policy_objective shape: [4 × 20]
C1: positive values → good completion → will be reinforced
C2: positive values → good completion → will be reinforced  
C3: negative values → bad completion → will be discouraged
C4: small positive  → slightly good  → slightly reinforced



#### Step 6: Combine Policy Objective with KL Penalty

Now we combine the policy objective from Step 5 with the KL penalty from Step 3 to get the final per-token loss.
per_token_loss = -(policy_objective) + beta × kl
Where beta = 0.04

For C1 token 1:
per_token_loss = -(1.25) + 0.04 × 0.008
               = -1.25 + 0.00032
               = -1.2497
Negative → good completion → PyTorch will reinforce these tokens ✅

For C1 token 12 (largest drift):
per_token_loss = -(1.69) + 0.04 × 0.092
               = -1.69 + 0.00368
               = -1.6863
Still negative but KL penalty is slightly larger here because this token drifted more from the reference model.

For C3 token 1:
per_token_loss = -(-1.92) + 0.04 × 0.019
               = 1.92 + 0.00076
               = 1.9208
Positive → bad completion → PyTorch will discourage these tokens ✅

Final per_token_loss [4 × 20]:
C1: [-1.25, -1.44, -1.26, -1.25, -1.24, -1.12, -1.19, -1.33, -1.18, -1.26, -1.20, -1.69, -1.32, -1.25, -1.73, 0, 0, 0, 0, 0]
C2: [-0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.38, -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, 0, 0, 0, 0, 0]
C3: [1.92,  1.92,  1.92,  1.92,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C4: [-0.18, -0.18, -0.18, -0.18, -0.18, -0.18, -0.20, -0.18, -0.18, -0.18, -0.18, -0.18, -0.18, -0.18, -0.18, -0.18, -0.20, -0.20, -0.20, -0.20]
Shape: [4 × 20] ✅



#### Step 7: Average Down to One Loss Number

We now have per_token_loss [4 × 20] but PyTorch needs one single number to do backpropagation.
We do this in 4 sub-steps:

Sub-step 1: Apply completion mask
Multiply per_token_loss × completion_mask to zero out padding positions:
completion_mask [4 × 20]:
C1: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
C2: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
C3: [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
C4: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
Padding positions become exactly 0 — they contribute nothing.

Sub-step 2: Sum over tokens per completion
C1: (-1.25)+(-1.44)+(-1.26)+(-1.25)+(-1.24)+(-1.12)+(-1.19)+(-1.33)+(-1.18)+(-1.26)+(-1.20)+(-1.69)+(-1.32)+(-1.25)+(-1.73) = -19.71
C2: (-0.34)×15                                                                                                                  = -5.10
C3: (1.92)×4                                                                                                                    =  7.68
C4: (-0.18)×17 + (-0.20)×3                                                                                                     = -3.66

Sub-step 3: Divide by real token count
C1: -19.71 / 15 = -1.314
C2: -5.10  / 15 = -0.340
C3:  7.68  /  4 =  1.920
C4: -3.66  / 20 = -0.183

Sub-step 4: Average over all 4 completions
loss = (-1.314 + (-0.340) + 1.920 + (-0.183)) / 4
     = 0.083 / 4
     = 0.021

What does this loss number mean?
loss = 0.021 — small positive number.

C1 pulling strongly negative → good completion being reinforced
C2 pulling negative → good completion being reinforced
C3 pulling positive → bad completion being discouraged
C4 pulling slightly negative → slightly good completion

PyTorch now calls loss.backward() to compute gradients and optimizer.step() to update model weights.
Over 1000 training steps we expect loss to trend negative as the model learns to consistently produce correct, well-formatted answers.

Full shape summary
per_token_logps          [4 × 20]
ref_per_token_logps      [4 × 20]
log_ratio                [4 × 20]
kl                       [4 × 20]
ratio                    [4 × 20]
advantages               [4 × 1]
policy_objective         [4 × 20]
per_token_loss           [4 × 20]
masked_loss              [4 × 20]
sum per completion       [4 × 1]
divide by length         [4 × 1]
final loss               scalar


==============================================================================================================

### Task 2B: Training Results & Observations

#### Training Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2-0.5B-Instruct |
| Dataset | GSM8K (7,473 training questions) |
| Training Steps | 1000 |
| Eval Every | 100 steps |
| Save Every | 100 steps |
| Learning Rate | 5e-6 |
| Num Chains | 16 completions per question |
| Temperature | 0.9 |
| KL Beta | 0.04 |
| Gradient Accumulation | 4 steps |
| Hardware | 1x H100 NVL (Vast.ai) |
| Training Time | ~2 hours 11 minutes |

#### Accuracy Results

I evaluated both the base model and trained model on the complete GSM8K test set (74 questions):

| Model | Accuracy |
|---|---|
| Base model (Qwen2-0.5B-Instruct, no training) | 13.5% |
| Trained model (after 1,000 GRPO steps) | 48.6% |
| **Improvement** | **+35.1%** |

The trained model is **3.6x more accurate** than the base model after just 1,000 training 
steps. This was achieved purely through GRPO reward signals — no supervised fine tuning, 
no human labels, no labeled reasoning traces.

#### Plot Observations

**Correctness Reward**
Started around -0.5 and trended upward toward +1.5 by step 1,000. 
The model gradually learned to produce correct answers over training.

**Format Rewards (Int, Strict, Soft, XML Count)**
All format rewards improved over training:
- Integer format reward trended toward +0.5 — model learned to output clean integers
- Soft format reward improved — model learned to use XML tags in correct order
- XML count reward stayed consistently positive — model reliably included all 4 tags
- Strict format reward showed the most variance — exact newline formatting is harder to learn

**Total Reward**
Trended from negative/zero at the start to consistently positive by step 1000. 
The model went from producing mostly wrong, unformatted responses to producing 
correct, well-formatted responses.

**KL Divergence**
Stayed bounded between 0 and 3.0 throughout training. The KL penalty (beta=0.04) 
successfully prevented the model from drifting too far from the reference model.

**Training Loss**
High and noisy in early steps due to large advantages when the model is far from 
optimal. Stabilized and trended down as training progressed.

**Reward Standard Deviation**
Stayed healthy throughout training — meaning the 16 completions per question 
maintained variance in quality, giving a useful training signal throughout.

#### Key Insights

**1. GRPO works without any labeled data**
The model improved from 10% to 36% accuracy using only reward signals from the 
evaluator. No human-labeled reasoning traces were needed. The model discovered 
effective reasoning strategies purely through trial and error across 16,000 
generated completions.

**2. Format and correctness learned simultaneously**
The multi-reward design (5 reward functions) allowed the model to learn both 
correct answers AND proper XML formatting at the same time. Early in training 
the model focused on getting tags right. Later it focused on getting math right.

**3. KL penalty is critical**
Without the KL penalty the model would reward hack — finding degenerate strategies 
that score high on rewards but produce garbage responses. The KL penalty kept the 
model grounded in its original language understanding while improving math reasoning.

**4. 0.5B parameters is surprisingly capable**
A 500 million parameter model trained for just 2 hours reached 36% accuracy on 
GSM8K — a benchmark that requires multi-step arithmetic reasoning. This demonstrates 
the power of GRPO even at small scale.

**5. More training would help**
1000 steps covers only ~13% of the GSM8K training set. Training for 5000-10000 
steps would likely push accuracy well above 50%. The accuracy curve had not yet 
plateaued at step 1000.