import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
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


class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")


class GSM8kEvaluator(RewardEvaluator):

    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """

    def __init__(self):
        self.num_reward_functions = 5

    def _extract_xml_answer(self, text: str) -> str:
        text = _normalize_newlines(text)
        m = ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""

 

    def _extract_raw_answer(self, text: str) -> str:
        """
        Fallback method to extract a number from raw text when no <answer> tags are found.
        
        We try three strategies in order:

        Strategy 1: Find the next digit number immediately after the word 'answer' or 'result'.
        Examples handled:
            'the answer I computed is 50'  → '50'
            'answer: 50'                   → '50'
            'answer = -50'                 → '-50'
            'the result is 50'             → '50'
        We use [^0-9-]* to skip any non-digit characters between 'answer' and the number,
        so we always get the NEXT number after the word 'answer', not the first or last
        number in the entire text.

        Strategy 2: Find a word number immediately after the word 'answer' or 'result'.
        Examples handled:
            'the answer I computed is Fifty'  → '50'
            'the answer is twenty five'       → '25'
        We use word2number (w2n) to convert word numbers to integers.
        If w2n cannot parse the word, we move to Strategy 3.

        Strategy 3: Last resort — return the last digit number in the entire text.
        Examples handled:
            'there are 100 males and 50 females so 50'  → '50'
        This is unreliable but better than returning nothing.
        The format reward functions will penalize responses that don't use <answer> tags,
        so this fallback should rarely be needed once training progresses.
        """
        # Strategy 1: Find next digit number after "answer" or "result"
        m = re.search(r'(?:answer|result)[^0-9-]*(-?\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1)
        
        # Strategy 2: Find next word number after "answer" or "result"
        m = re.search(r'(?:answer|result)[^a-zA-Z]*([a-zA-Z][a-zA-Z\s]+?)(?:\.|,|\d|$)', text, re.IGNORECASE)
        if m:
            try:
                return str(w2n.word_to_num(m.group(1).strip()))
            except:
                pass
        
        # Strategy 3: Last resort — last number in text
        numbers = re.findall(r'-?\d+', text)
        return numbers[-1] if numbers else ""





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
        rewards: List[float] = []
        for response, gt in zip(responses, answer):
            gt_str = str(gt).strip()
            gt_i = self._parse_int(gt_str)

            # First try to extract from <answer> tags
            pred_str = self._extract_xml_answer(response).strip()

            # No answer tag found → try extracting from raw text
            if pred_str == "":
                pred_str = self._extract_raw_answer(response).strip()

            # Still nothing found → truly no attempt → neutral
            if pred_str == "":
                rewards.append(0.0)
                continue

            pred_i = self._parse_int(pred_str)

            if pred_i is not None and gt_i is not None:
                rewards.append(2.0 if pred_i == gt_i else -1.0)
            else:
                rewards.append(2.0 if pred_str == gt_str else -1.0)

        return rewards

    def _int_format_reward(self, completions) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        rewards: List[float] = []
        for response in responses:
            # First try tags
            extracted = self._extract_xml_answer(response).strip()

            # Fallback to raw text if no tags found
            if extracted == "":
                extracted = self._extract_raw_answer(response).strip()

            # No answer found at all → didn't attempt → neutral
            if extracted == "":
                rewards.append(0.0)
            else:
                rewards.append(0.5 if INT_RE.fullmatch(extracted) else -0.5)

        return rewards

    def _strict_format_reward(self, completions) -> List[float]:
        responses = [_normalize_newlines(c[0]["content"]) for c in completions]
        return [0.5 if STRICT_RE.fullmatch(r) else -0.5 for r in responses]

    def _soft_format_reward(self, completions) -> List[float]:
        responses = [c[0]["content"] for c in completions]
        rewards = []
        for r in responses:
            r = _normalize_newlines(r)
            if SOFT_RE.search(r):
                rewards.append(0.5)
            elif REVERSE_RE.search(r):
                rewards.append(0.25)
            elif r.strip() == "":
                rewards.append(0.0)   # didn't attempt → neutral
            else:
                rewards.append(-0.5)  # tried but wrong format → penalty
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

    def compute_rewards(
            self,
            prompts: List[List[Dict[str, str]]],
            completions: List[List[Dict[str, str]]],
            answer: Any,
            device: str
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute all rewards for the given completions."""

            num_completions = len(completions)
            rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

            # Compute all reward functions
            all_scores = [
                self._correctness_reward(prompts, completions, answer),
                self._int_format_reward(completions),
                self._strict_format_reward(completions),
                self._soft_format_reward(completions),
                self._xml_count_reward(completions)
            ]
            
            # Fill rewards tensor
            for i, scores in enumerate(all_scores):
                rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
            
            # Compute metrics
            reward_per_func = rewards_per_func.mean(0)
            
            # Calculate accuracy (perfect correctness score)
            correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
            num_perfect = (correctness_scores == 2.0).sum().item()
            accuracy = num_perfect / num_completions
            
            metrics = {
                "rewards/correctness_reward_func": reward_per_func[0].item(),
                "rewards/int_reward_func": reward_per_func[1].item(), 
                "rewards/strict_format_reward_func": reward_per_func[2].item(),
                "rewards/soft_format_reward_func": reward_per_func[3].item(),
                "rewards/xmlcount_reward_func": reward_per_func[4].item(),
                "reward": rewards_per_func.sum(dim=1).mean().item(),
                "accuracy": accuracy
            }
            
            return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }