import torch
from evaluator import RewardEvaluator

# Maps --section argument to reward tensor column index
SECTION_TO_COL = {
    "narration": 0,
    "story":     1,
    "language":  2,
}


def score_completions(
    completions_text: list,
    messages: list,
    section: str,
    eval_class: RewardEvaluator,
    device: str,
    args,
) -> tuple:
    """
    Score 16 story completions and compute advantages for one LoRA.

    Args:
        completions_text : list of 16 decoded story strings from generation.py
        messages         : original chat messages (prompt) for this step
        section          : which LoRA is training — "narration", "story", "language"
        eval_class       : CreativeWritingEvaluator instance
        device           : "cuda" or "cpu"
        args             : training args (num_chains)

    Returns:
        rewards          : (num_chains,) — one scalar reward per completion
        advantages       : (num_chains,) — normalized rewards for GRPO
        rewards_per_func : (num_chains, 3) — all 3 section scores, for logging
        metrics          : dict of aggregated metrics
        log_data         : dict of per-story scores for writing to disk
    """

    # ------------------------------------------------------------------
    # 1. Format inputs for evaluator
    #    evaluator.compute_rewards expects:
    #      prompts     : list of chat message lists
    #      completions : list of [{"role": "assistant", "content": story}]
    #      answer      : None for creative writing (no ground truth)
    # ------------------------------------------------------------------
    mock_prompts     = [messages] * len(completions_text)
    mock_completions = [[{"role": "assistant", "content": c}]
                        for c in completions_text]

    # ------------------------------------------------------------------
    # 2. Call evaluator → rewards_per_func shape (16, 3)
    #    col 0 = narration score
    #    col 1 = story score
    #    col 2 = language score
    # ------------------------------------------------------------------
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=None,
        device=device,
    )

    # ------------------------------------------------------------------
    # 3. Select ONE column — this is the core LoRA logic
    #    narration LoRA only sees narration rewards
    #    story LoRA only sees story rewards
    #    language LoRA only sees language rewards
    # ------------------------------------------------------------------
    col     = SECTION_TO_COL[section]
    rewards = rewards_per_func[:, col]   # shape: (16,)

    # ------------------------------------------------------------------
    # 4. Compute advantages — identical math to original main.py
    #    Normalize within the group of 16 completions:
    #      advantage = (reward - mean) / (std + epsilon)
    # ------------------------------------------------------------------
    mean_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_rewards  = rewards.view(-1, args.num_chains).std(dim=1)

    mean_rewards = mean_rewards.repeat_interleave(args.num_chains, dim=0)
    std_rewards  = std_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)

    metrics["reward_std"] = std_rewards.mean().item()

    # ------------------------------------------------------------------
    # 5. Build log data — one entry per story, all 3 section scores
    # ------------------------------------------------------------------
    log_data = {
        "section":     section,
        "prompt":      messages[-1]["content"],   # user message = the opening line
        "generations": [],
    }

    for i, (story, reward_scores) in enumerate(
        zip(completions_text, rewards_per_func)
    ):
        log_data["generations"].append({
            "story":      story,
            "scores":     eval_class.get_reward_breakdown(reward_scores),
            "reward_used": rewards[i].item(),    # the column this LoRA trained on
            "advantage":   advantages[i].item(),
        })

    return rewards, advantages, rewards_per_func, metrics, log_data