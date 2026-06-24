"""
Generate gold standard SFT dataset using Claude API.

For each GSM8K training problem, Claude generates a high quality
reasoning trace in the exact format required by our training pipeline:

<reasoning>
...step by step working...
</reasoning>
<answer>
INTEGER
</answer>

Output: sft_dataset.json — list of {question, answer, solution} dicts

Notes:
- Uses first 1,000 examples from GSM8K training split
- Test evaluation uses official GSM8K test split (1,319 questions)
- No overlap between training and test sets
"""

import json
import time
import anthropic
from datasets import load_dataset
from tqdm import tqdm
import re

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_EXAMPLES = 1000
OUTPUT_FILE = "sft_dataset.json"
MODEL = "claude-sonnet-4-6"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a math tutoring assistant.
You will be given a math problem and its correct answer.
Your job is to write a clear, concise step-by-step solution.

You MUST respond in EXACTLY this format and nothing else:
<reasoning>
...your step by step working...
</reasoning>
<answer>
INTEGER
</answer>

Rules:
- The very first characters of your response must be "<reasoning>\\n"
- Show each calculation step clearly on its own line
- Place ONLY the final integer answer inside <answer> tags
- No text before <reasoning> or after </answer>
- The answer must match the provided correct answer exactly
"""

def extract_hash_answer(text: str):
    """Extract answer from GSM8K format: '#### 42'"""
    if '####' in text:
        return text.split('####')[-1].strip()
    return None

def generate_solution(client, question, answer):
    """Call Claude API to generate a gold standard solution."""
    user_message = f"""Problem: {question}

Correct answer: {answer}

Write a step-by-step solution in the required format."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_message}
        ],
        system=SYSTEM_PROMPT
    )
    
    return response.content[0].text


def verify_solution(solution, answer):
    """Verify solution contains correct answer in tags."""
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', solution, re.DOTALL)
    if not m:
        return False
    extracted = m.group(1).strip()
    return extracted == str(answer).strip()


def main():
    # Load GSM8K from local JSON file (avoids Python 3.14 datasets compatibility issue)
    print("Loading GSM8K from local file...")
    with open('gsm8k_train_1000.json', 'r') as f:
        examples = json.load(f)
    
    print(f"Loaded {len(examples)} training examples")
    print(f"Test evaluation will use official GSM8K test split (no overlap)")
    
    # Initialize Anthropic client
    client = anthropic.Anthropic()
    
    # Generate solutions
    dataset = []
    failed = 0
    
    print(f"\nGenerating {len(examples)} gold standard solutions using {MODEL}...")
    
    for i, item in enumerate(tqdm(examples)):
        question = item['question']
        answer = item['answer']
        
        try:
            solution = generate_solution(client, question, answer)
            
            # Verify solution contains correct answer
            if verify_solution(solution, answer):
                dataset.append({
                    "question": question,
                    "answer": answer,
                    "solution": solution
                })
            else:
                # Retry once
                time.sleep(0.5)
                solution = generate_solution(client, question, answer)
                if verify_solution(solution, answer):
                    dataset.append({
                        "question": question,
                        "answer": answer,
                        "solution": solution
                    })
                else:
                    failed += 1
                    print(f"\nFailed to verify Q{i+1}: {question[:50]}...")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            failed += 1
            print(f"\nError on Q{i+1}: {e}")
            time.sleep(1)
    
    # Save dataset
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDone!")
    print(f"Successfully generated: {len(dataset)} examples")
    print(f"Failed/skipped:        {failed} examples")
    print(f"Saved to:              {OUTPUT_FILE}")
    
    # Show a sample
    if dataset:
        print(f"\nSample entry:")
        print(f"Question: {dataset[0]['question'][:100]}...")
        print(f"Answer:   {dataset[0]['answer']}")
        print(f"Solution:\n{dataset[0]['solution']}")


if __name__ == '__main__':
    main()