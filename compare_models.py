"""
Compare base model vs trained model on 100 GSM8K test questions.
Outputs a text file with side by side responses and accuracy summary.
"""
import torch
import rl_datasets
import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load GSM8K test set ────────────────────────────────────────────────────────
train_loader, test_loader = rl_datasets.get_dataloaders('gsm8k')

# Collect 100 test questions
test_questions = []
test_loader.reset()
for question, answer in test_loader:
    test_questions.append((question, answer))
    if len(test_questions) >= 100:
        break

print(f"Loaded {len(test_questions)} test questions")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = train_loader.pre_prompt

# ── Evaluator ─────────────────────────────────────────────────────────────────
eval_class = evaluator.get_evaluator('gsm8k')

# ── Helper functions ───────────────────────────────────────────────────────────
def get_response(model, tokenizer, question):
    prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response


def is_correct(response, answer):
    """Use evaluator to check if response is correct."""
    completions = [[{"role": "assistant", "content": response}]]
    answers = [answer]
    prompts = [[{"role": "user", "content": ""}]]
    rewards, _ = eval_class.compute_rewards(prompts, completions, answers, "cpu")
    return rewards[0][0].item() == 2.0


# ── Load models ────────────────────────────────────────────────────────────────
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None
).to(device)
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
print("Base model loaded.")

print("Loading trained model...")
trained_model = AutoModelForCausalLM.from_pretrained(
    "/workspace/rl-takehome-lz/output/Qwen-0.5B/checkpoints/step_1000",
    torch_dtype=torch.bfloat16,
    device_map=None
).to(device)
trained_tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/rl-takehome-lz/output/Qwen-0.5B/checkpoints/step_1000"
)
print("Trained model loaded.")

# ── Run comparison ─────────────────────────────────────────────────────────────
base_correct = 0
trained_correct = 0

with open("/workspace/rl-takehome-lz/model_comparison.txt", "w") as f:
    for i, (question, answer) in enumerate(test_questions):
        print(f"Question {i+1}/100...")

        base_response = get_response(base_model, base_tokenizer, question)
        trained_response = get_response(trained_model, trained_tokenizer, question)

        base_right = is_correct(base_response, answer)
        trained_right = is_correct(trained_response, answer)

        if base_right:
            base_correct += 1
        if trained_right:
            trained_correct += 1

        f.write("=" * 80 + "\n")
        f.write(f"Question {i+1}: {question}\n")
        f.write(f"Ground Truth: {answer}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"BASE MODEL ({'✓ CORRECT' if base_right else '✗ WRONG'}):\n")
        f.write(base_response + "\n\n")

        f.write(f"TRAINED MODEL ({'✓ CORRECT' if trained_right else '✗ WRONG'}):\n")
        f.write(trained_response + "\n\n")

    # Summary
    f.write("=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total questions:       100\n")
    f.write(f"Base model accuracy:   {base_correct}%\n")
    f.write(f"Trained model accuracy:{trained_correct}%\n")
    f.write(f"Improvement:           +{trained_correct - base_correct}%\n")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Base model accuracy:    {base_correct}%")
print(f"Trained model accuracy: {trained_correct}%")
print(f"Improvement:            +{trained_correct - base_correct}%")
print("Written to model_comparison.txt")