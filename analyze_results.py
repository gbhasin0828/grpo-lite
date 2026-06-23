"""
Analyze model_comparison.txt and compute accuracy metrics
for both base and trained model across all 5 reward functions.
"""
import re
import sys
import evaluator
from collections import defaultdict

# ── Load evaluator ─────────────────────────────────────────────────────────────
eval_class = evaluator.get_evaluator('gsm8k')

# ── Parse model_comparison.txt ─────────────────────────────────────────────────
def parse_comparison_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into question blocks
    blocks = content.split('=' * 80)
    blocks = [b.strip() for b in blocks if b.strip()]

    results = []
    i = 0
    while i < len(blocks):
        block = blocks[i]

        # Parse question block
        if block.startswith('Question'):
            lines = block.split('\n')
            question_line = lines[0]
            gt_line = lines[1]

            question = question_line.split(':', 1)[1].strip()
            ground_truth = gt_line.split(':', 1)[1].strip()

            # Next block has base and trained responses
            if i + 1 < len(blocks):
                response_block = blocks[i + 1]

                # Split on TRAINED MODEL
                parts = response_block.split('TRAINED MODEL')
                base_part = parts[0]
                trained_part = parts[1] if len(parts) > 1 else ''

                # Extract base response
                base_response = re.sub(r'BASE MODEL \(.*?\):', '', base_part).strip()

                # Extract trained response
                trained_response = re.sub(r'\(.*?\):', '', trained_part, count=1).strip()

                results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'base_response': base_response,
                    'trained_response': trained_response
                })
                i += 2
            else:
                i += 1
        else:
            i += 1

    return results


def score_response(response, ground_truth):
    """Score a single response across all 5 reward functions."""
    completions = [[{"role": "assistant", "content": response}]]
    answers = [ground_truth]
    prompts = [[{"role": "user", "content": ""}]]
    rewards, _ = eval_class.compute_rewards(prompts, completions, answers, "cpu")
    return rewards[0].tolist()  # [correctness, int_fmt, strict, soft, xml]


def is_pass(score, func_idx):
    """Determine if a score counts as passing for a given reward function."""
    if func_idx == 4:  # XML count — partial credit
        return score > 0.0
    return score > 0.0


# ── Main analysis ──────────────────────────────────────────────────────────────
def main():
    filepath = 'model_comparison.txt'
    results = parse_comparison_file(filepath)
    print(f"Parsed {len(results)} questions\n")

    reward_names = ['Correctness', 'Int Format', 'Strict Format', 'Soft Format', 'XML Count']

    # Track scores for each model
    base_scores = []
    trained_scores = []

    # Bucket analysis for correctness
    buckets = {'A': [], 'B': [], 'C': [], 'D': []}

    for i, r in enumerate(results):
        base_s = score_response(r['base_response'], r['ground_truth'])
        trained_s = score_response(r['trained_response'], r['ground_truth'])

        base_scores.append(base_s)
        trained_scores.append(trained_s)

        # Correctness bucket
        base_correct = base_s[0] > 0
        trained_correct = trained_s[0] > 0

        if base_correct and trained_correct:
            buckets['A'].append(i)
        elif not base_correct and trained_correct:
            buckets['B'].append(i)
        elif base_correct and not trained_correct:
            buckets['C'].append(i)
        else:
            buckets['D'].append(i)

    # ── Print accuracy table ───────────────────────────────────────────────────
    print("=" * 60)
    print("ACCURACY BY REWARD FUNCTION")
    print("=" * 60)
    print(f"{'Reward Function':<20} {'Base':>10} {'Trained':>10} {'Delta':>10}")
    print("-" * 60)

    for j, name in enumerate(reward_names):
        base_acc = sum(1 for s in base_scores if is_pass(s[j], j)) / len(results) * 100
        trained_acc = sum(1 for s in trained_scores if is_pass(s[j], j)) / len(results) * 100
        delta = trained_acc - base_acc
        sign = '+' if delta >= 0 else ''
        print(f"{name:<20} {base_acc:>9.1f}% {trained_acc:>9.1f}% {sign}{delta:>8.1f}%")

    # ── Print bucket analysis ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CORRECTNESS BUCKET ANALYSIS")
    print("=" * 60)
    print(f"Bucket A (Both correct):         {len(buckets['A']):>3} questions")
    print(f"Bucket B (Only trained correct): {len(buckets['B']):>3} questions ← GRPO added value")
    print(f"Bucket C (Only base correct):    {len(buckets['C']):>3} questions ← GRPO hurt")
    print(f"Bucket D (Both wrong):           {len(buckets['D']):>3} questions")

    # ── Print per-question breakdown ───────────────────────────────────────────
    print()
    print("=" * 60)
    print("PER QUESTION BREAKDOWN")
    print("=" * 60)
    print(f"{'Q#':<4} {'Question':<45} {'Base':>6} {'Trained':>8}")
    print(f"{'':4} {'':45} {'C I S T X':>6} {'C I S T X':>8}")
    print("-" * 70)

    for i, r in enumerate(results):
        base_s = base_scores[i]
        trained_s = trained_scores[i]

        def fmt(scores):
            return ' '.join(['✓' if is_pass(s, j) else '✗' for j, s in enumerate(scores)])

        q_short = r['question'][:42] + '...' if len(r['question']) > 42 else r['question']
        print(f"{i+1:<4} {q_short:<45} {fmt(base_s):>6} {fmt(trained_s):>8}")

    # ── Print example from each bucket ────────────────────────────────────────
    print()
    print("=" * 60)
    print("EXAMPLE FROM EACH BUCKET")
    print("=" * 60)

    bucket_labels = {
        'A': 'Bucket A — Both Correct',
        'B': 'Bucket B — Only Trained Correct (GRPO added value)',
        'C': 'Bucket C — Only Base Correct (GRPO hurt)',
        'D': 'Bucket D — Both Wrong'
    }

    for bucket, label in bucket_labels.items():
        if buckets[bucket]:
            idx = buckets[bucket][0]
            r = results[idx]
            print(f"\n{label}")
            print(f"Question: {r['question']}")
            print(f"Ground Truth: {r['ground_truth']}")
            print(f"\nBASE MODEL:\n{r['base_response'][:300]}...")
            print(f"\nTRAINED MODEL:\n{r['trained_response'][:300]}...")
            print()


if __name__ == '__main__':
    main()