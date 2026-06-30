import re
import os
import json
import time
import torch
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Rubric — 3 LoRAs, 21 criteria total
# ---------------------------------------------------------------------------
RUBRIC = {
    "narration": {
        "weight": 0.40,
        "criteria": [
            "prompt_utilization",
            "narrative_promise",
            "goal_formation",
            "decision_consequence",
            "character_transformation",
            "resolution_satisfaction",
            "narrative_compression",
            "thematic_integration",
            "structural_symmetry",
        ],
    },
    "story": {
        "weight": 0.40,
        "criteria": [
            "information_density",
            "information_novelty",
            "information_timing",
            "information_efficiency",
            "information_connectivity",
            "curiosity_trajectory",
            "prediction_updates",
        ],
    },
    "language": {
        "weight": 0.20,
        "criteria": [
            "precision",
            "sensory_concreteness",
            "rhythm_and_flow",
            "voice_consistency",
            "lexical_efficiency",
        ],
    },
}

SECTIONS = list(RUBRIC.keys())   # ["narration", "story", "language"]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are a rigorous literary judge scoring short story continuations.

You will receive an opening line and N story continuations labeled STORY_1 … STORY_N.
Score EVERY criterion below for EVERY story on a scale of 1–5:
  1 = Very poor  |  2 = Below average  |  3 = Average  |  4 = Good  |  5 = Exceptional

STRICT RULES:
1. Score every criterion for every story — no omissions.
2. Spread scores. Best and worst story must differ by at least 1 point on most criteria.
3. No two stories may receive the same total score.
4. A 5 is rare — reserve it for genuinely exceptional writing.
5. Return ONLY valid JSON. No markdown, no explanation, no preamble.

CRITERIA:

NARRATION (40%)
  prompt_utilization       : Does the story meaningfully build on the opening line?
  narrative_promise        : Does it establish compelling unanswered questions?
  goal_formation           : Is the protagonist's objective clear?
  decision_consequence     : Do character choices produce meaningful consequences?
  character_transformation : Does something fundamentally change by the end?
  resolution_satisfaction  : Does the ending fulfill the opening's promises?
  narrative_compression    : Does every sentence meaningfully contribute?
  thematic_integration     : Does a larger idea emerge naturally through events?
  structural_symmetry      : Do the opening and ending reinforce one another?

STORY (40%)
  information_density      : How much meaningful new information per sentence?
  information_novelty      : Does each sentence add genuinely new understanding?
  information_timing       : Are reveals introduced at maximum narrative impact?
  information_efficiency   : What proportion of content advances plot/character/theme?
  information_connectivity : Do details connect forward rather than remain isolated?
  curiosity_trajectory     : Does curiosity steadily build throughout?
  prediction_updates       : Does the story intelligently revise reader expectations?

LANGUAGE (20%)
  precision                : Is language specific rather than vague?
  sensory_concreteness     : Are scenes grounded in concrete sensory details?
  rhythm_and_flow          : Does sentence structure create effective pacing?
  voice_consistency        : Is the narrative voice stable throughout?
  lexical_efficiency       : Does each word contribute meaningful value?

REQUIRED JSON FORMAT — return exactly this, nothing else:
{
  "STORY_1": {
    "narration": {"prompt_utilization":<1-5>,"narrative_promise":<1-5>,"goal_formation":<1-5>,"decision_consequence":<1-5>,"character_transformation":<1-5>,"resolution_satisfaction":<1-5>,"narrative_compression":<1-5>,"thematic_integration":<1-5>,"structural_symmetry":<1-5>},
    "story":    {"information_density":<1-5>,"information_novelty":<1-5>,"information_timing":<1-5>,"information_efficiency":<1-5>,"information_connectivity":<1-5>,"curiosity_trajectory":<1-5>,"prediction_updates":<1-5>},
    "language": {"precision":<1-5>,"sensory_concreteness":<1-5>,"rhythm_and_flow":<1-5>,"voice_consistency":<1-5>,"lexical_efficiency":<1-5>}
  },
  "STORY_2": { ... },
  "STORY_N": { ... }
}
"""

# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------
def _section_score(section_data: Dict, criteria: List[str]) -> float:
    """Mean of criteria scores (1–5), normalized to [-1, +1]. Mean of 3 → 0.0"""
    scores = [float(section_data[c]) for c in criteria if c in section_data]
    if not scores:
        return 0.0
    return (sum(scores) / len(scores) - 3.0) / 2.0


def _extract_opening_line(prompt: List[Dict[str, str]]) -> str:
    for msg in prompt:
        if msg.get("role") == "user":
            content = msg["content"]
            m = re.search(r'["\u201c\u2018](.+?)["\u201d\u2019]', content)
            return m.group(1).strip() if m else content.strip()
    return ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class RewardEvaluator(ABC):

    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_evaluator(name: str) -> RewardEvaluator:
    if name.lower() == "creative_writing":
        return CreativeWritingEvaluator()
    raise NotImplementedError(f"No evaluator for '{name}'")


# ---------------------------------------------------------------------------
# CreativeWritingEvaluator
# ---------------------------------------------------------------------------
class CreativeWritingEvaluator(RewardEvaluator):
    """
    LLM-as-a-judge evaluator for creative writing with 3-LoRA reward structure.

    Output:
        rewards_per_func : Tensor (num_completions, 3)
            col 0 = narration score  [-1, +1]
            col 1 = story score      [-1, +1]
            col 2 = language score   [-1, +1]

        metrics : dict
            rewards/narration_mean
            rewards/story_mean
            rewards/language_mean
            rewards/score_std        ← watch for collapse
            reward                   ← weighted total, expected by main.py
            accuracy                 ← always 0.0, expected by main.py
    """

    def __init__(
        self,
        model: str         = "claude-sonnet-4-6",
        max_retries: int   = 3,
        retry_delay: float = 2.0,
    ):
        self.num_reward_functions = 3
        self.model       = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client      = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    # ------------------------------------------------------------------
    def _call_judge(self, opening_line: str, stories: List[str]) -> Optional[Dict]:
        """One batched API call → raw JSON dict. Returns None on failure."""
        stories_block = "\n\n".join(
            f"--- STORY_{i+1} ---\n{s.strip()}" for i, s in enumerate(stories)
        )
        user_msg = (
            f'OPENING LINE:\n"{opening_line}"\n\n'
            f"Score the following {len(stories)} continuations.\n\n"
            f"{stories_block}"
        )

        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw)

            except Exception as e:
                print(f"[Judge] attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        print("[Judge] all retries exhausted — returning neutral scores.")
        return None

    # ------------------------------------------------------------------
    def _parse_scores(self, judge_response: Optional[Dict], n: int) -> List[List[float]]:
        """
        Parse judge JSON → list of n rows, each row = [narration, story, language].
        Falls back to [0.0, 0.0, 0.0] per story on any parse error.
        """
        if judge_response is None:
            return [[0.0, 0.0, 0.0]] * n

        rows = []
        for i in range(n):
            key = f"STORY_{i+1}"
            try:
                sd = judge_response[key]
                row = [
                    _section_score(sd["narration"], RUBRIC["narration"]["criteria"]),
                    _section_score(sd["story"],     RUBRIC["story"]["criteria"]),
                    _section_score(sd["language"],  RUBRIC["language"]["criteria"]),
                ]
            except Exception as e:
                print(f"[Judge] parse error for {key}: {e} — using 0.0")
                row = [0.0, 0.0, 0.0]
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        opening_line = _extract_opening_line(prompts[0])
        stories      = [c[0]["content"] for c in completions]
        n            = len(completions)

        judge_response   = self._call_judge(opening_line, stories)
        rows             = self._parse_scores(judge_response, n)

        # shape: (n, 3)
        rewards_per_func = torch.tensor(rows, dtype=torch.float32, device=device)

        # weighted total per completion:  0.40·narration + 0.40·story + 0.20·language
        weights      = torch.tensor([0.40, 0.40, 0.20], device=device)
        total_reward = (rewards_per_func * weights).sum(dim=1)   # shape: (n,)

        col_means = rewards_per_func.mean(dim=0)   # [narration_mean, story_mean, lang_mean]

        metrics = {
            "rewards/narration_mean": col_means[0].item(),
            "rewards/story_mean":     col_means[1].item(),
            "rewards/language_mean":  col_means[2].item(),
            "rewards/score_std":      rewards_per_func.std().item(),
            "reward":                 total_reward.mean().item(),
            "accuracy":               0.0,
        }

        return rewards_per_func, metrics

    # ------------------------------------------------------------------
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        return {
            "narration": reward_scores[0].item(),
            "story":     reward_scores[1].item(),
            "language":  reward_scores[2].item(),
        }