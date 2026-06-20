import re
from typing import List

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
INT_RE = re.compile(r"[+-]?\d+$")

class GSM8KEvaluator:

    def _extract_xml_answer(self, text):
        m = ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""

    def _parse_int(self, s):
        s = s.strip()
        if INT_RE.fullmatch(s):
            try:
                return int(s)
            except ValueError:
                return None
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
                rewards.append(2.0 if pred_i == gt_i else 0.0)
            else:
                rewards.append(2.0 if pred_str == gt_str else 0.0)

        return rewards

    def _int_format_reward(self, completions) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if INT_RE.fullmatch(a.strip()) else 0.0 for a in extracted]


if __name__ == "__main__":

    question = "Janet has 16 eggs. She eats 3 for breakfast. How many are left?"
    correct_answer = "13"

    completions = [
        [{"role": "assistant", "content": "<reasoning>16 - 3 = 13</reasoning><answer>13</answer>"}],
        [{"role": "assistant", "content": "<reasoning>16 - 3 = 12</reasoning><answer>12</answer>"}],
        [{"role": "assistant", "content": "<reasoning>I am not sure</reasoning><answer>thirteen</answer>"}],
        [{"role": "assistant", "content": "<reasoning>I am not sure</reasoning><answer>twelve</answer>"}],
    ]

    answer = [correct_answer, correct_answer, correct_answer, correct_answer]

    evaluator = GSM8KEvaluator()
    correctness = evaluator._correctness_reward(None, completions, answer)
    int_format  = evaluator._int_format_reward(completions)

    print(f"{'Completion':<15} {'Extracted':<12} {'Correctness':<15} {'Int Format':<12} {'Total'}")
    print("-" * 65)
    for i, comp in enumerate(completions):
        extracted = evaluator._extract_xml_answer(comp[0]["content"])
        total = correctness[i] + int_format[i]
        print(f"Completion {i+1:<4} {extracted:<12} {correctness[i]:<15} {int_format[i]:<12} {total}")