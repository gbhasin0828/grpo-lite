## Implementation Notes

### Task 1A: Environment Setup

**Dependency Management:**
Converted `requirements.txt` to `pyproject.toml` using `uv`. The original `requirements.txt` was a raw `pip freeze` dump containing ~150 system-level packages. Trimmed to the 7 packages the code actually imports: `torch`, `transformers`, `datasets`, `accelerate`, `tqdm`, `numpy`, `matplotlib`.

**PyTorch Installation:**
Configured a custom PyTorch index (`https://download.pytorch.org/whl/cu130`) to ensure the CUDA-enabled build is installed rather than the CPU-only PyPI version.

**Flash Attention 2:**
No prebuilt wheel exists for the environment on the provisioned H100 (Python 3.12, CUDA 13.0, PyTorch 2.12). Building from source takes 45-60 minutes and failed in this environment. Modified `llms.py` to use `attn_implementation="eager"` as a pragmatic alternative. This is slightly slower but functionally equivalent for correctness.

**Verified on:** 1x H100 NVL (Vast.ai), CUDA 13.0, Python 3.12

================
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


