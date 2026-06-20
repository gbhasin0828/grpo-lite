import re
import torch
from typing import List, Any, Optional
from word2number import w2n

# --- Regexes ---
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
SOFT_RE = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
REVERSE_RE = re.compile(r"<answer>.*?</answer>\s*<reasoning>.*?</reasoning>", re.DOTALL)
STRICT_RE = re.compile(
    r"^\s*<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$",
    re.DOTALL,
)
INT_RE = re.compile(r"[+-]?\d+$")

def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class GSM8kEvaluatorNEW:
    def __init__(self):
        self.num_reward_functions = 5

    def _extract_xml_answer(self, text: str) -> str:
        text = _normalize_newlines(text)
        m = ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""

    def _parse_int(self, s: str) -> Optional[int]:
        s = s.strip()
        # Try direct integer parse first
        if INT_RE.fullmatch(s):
            try:
                return int(s)
            except ValueError:
                return None
        # Try word to number e.g. "thirteen" -> 13
        try:
            return int(w2n.word_to_num(s))
        except:
            return None

    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        rewards: List[float] = []
        for pred, gt in zip(extracted, answer):
            gt_str = str(gt).strip()
            pred_str = pred.strip()
            pred_i = self._parse_int(pred_str)
            gt_i = self._parse_int(gt_str)
            if pred_i is not None and gt_i is not None:
                rewards.append(2.0 if pred_i == gt_i else -1.0)  # Fix 1
            else:
                rewards.append(2.0 if pred_str == gt_str else -1.0)  # Fix 1
        return rewards

    def _int_format_reward(self, completions) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if INT_RE.fullmatch(a.strip()) else -0.5 for a in extracted]  # Fix 2

    def _strict_format_reward(self, completions) -> List[float]:
        responses = [_normalize_newlines(c[0]["content"]) for c in completions]
        return [0.5 if STRICT_RE.fullmatch(r) else 0.0 for r in responses]

    def _soft_format_reward(self, completions) -> List[float]:  # Fix 3
        responses = [c[0]["content"] for c in completions]
        rewards = []
        for r in responses:
            r = _normalize_newlines(r)
            if SOFT_RE.search(r):           # reasoning then answer
                rewards.append(0.5)
            elif REVERSE_RE.search(r):      # answer then reasoning
                rewards.append(0.25)
            elif r.strip() == "":           # empty
                rewards.append(-0.5)
            else:                           # no tags or one tag
                rewards.append(0.0)
        return rewards

    def _xml_count_reward(self, completions) -> List[float]:
        def score(text: str) -> float:
            text = _normalize_newlines(text)
            r_open = "<reasoning>" in text
            r_close = "</reasoning>" in text
            a_open = "<answer>" in text
            a_close = "</answer>" in text
            s = 0.0
            s += 0.10 if r_open else 0.0
            s += 0.10 if r_close else 0.0
            s += 0.10 if a_open else 0.0
            s += 0.10 if a_close else 0.0
            if r_open and r_close and a_open and a_close:
                if text.find("<reasoning>") < text.find("</reasoning>") < text.find("<answer>") < text.find("</answer>"):
                    s += 0.10
            if "<reasoning>" in text:
                pre = text.split("<reasoning>", 1)[0]
                s -= min(0.25, 0.001 * len(pre))
            if "</answer>" in text:
                post = text.split("</answer>", 1)[-1]
                s -= min(0.25, 0.001 * len(post))
            return s
        responses = [c[0]["content"] for c in completions]
        return [score(r) for r in responses]

    def compute_rewards(self, prompts, completions, answer, device):
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        return rewards_per_func


if __name__ == "__main__":

    question = "Janet has 16 eggs. She eats 3 for breakfast. How many are left?"
    correct_answer = "13"

    completions = [
        [{"role": "assistant", "content": "<reasoning>\n16 - 3 = 13\n</reasoning>\n<answer>\n13\n</answer>"}],
        [{"role": "assistant", "content": "<reasoning>\n16 minus 3 equals 13\n</reasoning>\n<answer>\n13\n</answer>"}],
        [{"role": "assistant", "content": "<reasoning>\n13\n</reasoning>\n<answer>\n13\n</answer>"}],
        [{"role": "assistant", "content": "<reasoning>16-3=13</reasoning><answer>13</answer>"}],
        [{"role": "assistant", "content": "Here is my answer: <reasoning>16-3=13</reasoning><answer>13</answer>"}],
        [{"role": "assistant", "content": "<reasoning>\n16 - 3 = 12\n</reasoning>\n<answer>\n12\n</answer>"}],
        [{"role": "assistant", "content": "<reasoning>\n16 - 3 = 14\n</reasoning>\n<answer>\n14\n</answer>"}],
        [{"role": "assistant", "content": "<reasoning>I think</reasoning><answer>thirteen</answer>"}],
        [{"role": "assistant", "content": "<reasoning>I think</reasoning><answer>twelve</answer>"}],
        [{"role": "assistant", "content": "<answer>13</answer><reasoning>16-3=13</reasoning>"}],
        [{"role": "assistant", "content": "<answer>12</answer><reasoning>16-3=12</reasoning>"}],
        [{"role": "assistant", "content": "The answer is 13"}],
        [{"role": "assistant", "content": "I think it is 12"}],
        [{"role": "assistant", "content": "<reasoning>16-3=13</reasoning>"}],
        [{"role": "assistant", "content": ""}],
        [{"role": "assistant", "content": "<reasoning>\n16-3=13\n</reasoning>\n<answer>\n13\n</answer>\nHope this helps!"}],
    ]

    prompts = [[{"role": "user", "content": question}]] * 16
    answer = [correct_answer] * 16

    new_evaluator = GSM8kEvaluatorNEW()
    rewards = new_evaluator.compute_rewards(prompts, completions, answer, "cpu")

    print("=" * 100)
    print("NEW EVALUATOR RESULTS")
    print("=" * 100)
    print(f"Tensor shape: {rewards.shape}")
    print()
    print(f"{'#':<3} {'Description':<25} {'Correct':>8} {'IntFmt':>8} {'Strict':>8} {'Soft':>8} {'XML':>8} {'TOTAL':>8}")
    print("-" * 100)

    labels = [
        "1. Perfect",
        "2. Perfect variant",
        "3. Short reasoning",
        "4. No newlines",
        "5. Junk before tags",
        "6. Wrong ans 12",
        "7. Wrong ans 14",
        "8. Word ans thirteen",
        "9. Word ans twelve",
        "10. Ans then Rsn OK",
        "11. Ans then Rsn bad",
        "12. No tags correct",
        "13. No tags wrong",
        "14. Only reasoning",
        "15. Empty",
        "16. Junk after ans",
    ]

    for i, label in enumerate(labels):
        row = rewards[i]
        total = row.sum().item()
        print(f"{i+1:<3} {label:<25} {row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.2f} {row[4]:>8.2f} {total:>8.2f}")