"""
Train one LoRA adapter per section.
Usage:
    python lora_main.py --section narration
    python lora_main.py --section story
    python lora_main.py --section language
"""
import os, sys, json, torch, argparse
from tqdm import tqdm
from peft import get_peft_model, LoraConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import llms, utils
from evaluator import get_evaluator
from rl_datasets import get_creative_dataloader
from main_helpers.generation import generate_completions
from main_helpers.scoring import score_completions
from main_helpers.loss import compute_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--section",       required=True, choices=["narration","story","language"])
    p.add_argument("--model_name",    default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--prompts_path",  default="prompts.json")
    p.add_argument("--output_dir",    default="lora_adapters")
    p.add_argument("--lora_rank",     type=int,   default=16)
    p.add_argument("--lora_alpha",    type=int,   default=32)
    p.add_argument("--num_train_iters",type=int,  default=500)
    p.add_argument("--num_chains",    type=int,   default=16)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--kl_weight_beta",type=float, default=0.04)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_percent",type=float, default=0.18)
    p.add_argument("--temperature",   type=float, default=0.9)
    p.add_argument("--max_prompt_length",    type=int, default=256)
    p.add_argument("--max_completion_length",type=int, default=512)
    p.add_argument("--save_steps",    type=int,   default=100)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dirs
    adapter_dir = os.path.join(args.output_dir, args.section)
    log_dir     = os.path.join(adapter_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Models
    model,      tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _         = llms.get_llm_tokenizer(args.model_name, device)

    # Attach LoRA to trainable model only
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Data + evaluator
    data_iter  = iter(get_creative_dataloader(args.prompts_path))
    eval_class = get_evaluator("creative_writing")

    # Optimizer on LoRA params only
    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.learning_rate, eps=1e-8)
    warmup     = int(args.warmup_percent * args.num_train_iters / args.gradient_accumulation_steps)
    scheduler  = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, s/warmup) if warmup > 0 else 1.0
    )

    # Training loop
    train_logs = {}
    optimizer.zero_grad()

    for step in tqdm(range(args.num_train_iters), desc=f"LoRA [{args.section}]"):

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(get_creative_dataloader(args.prompts_path))
            batch     = next(data_iter)

        # Generate → Score → Loss
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = \
            generate_completions(model, tokenizer, batch["messages"], device, args)

        rewards, advantages, rewards_per_func, metrics, log_data = \
            score_completions(completions_text, batch["messages"], args.section, eval_class, device, args)

        completion_mask = attention_mask[:, prompt_ids.size(1):]
        loss, loss_metrics = compute_loss(
            model, base_model, prompt_completion_ids, prompt_ids,
            completion_ids, attention_mask, completion_mask, advantages, args
        )

        # Backward
        (loss / args.gradient_accumulation_steps).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Log
        metrics.update(loss_metrics)
        metrics["loss"] = loss.item()
        train_logs[step] = metrics
        with open(os.path.join(log_dir, "train_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=2)

        # Save adapter
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            model.save_pretrained(os.path.join(adapter_dir, f"step_{step+1}"))

    # Final save
    model.save_pretrained(os.path.join(adapter_dir, "final"))
    tokenizer.save_pretrained(os.path.join(adapter_dir, "final"))
    print(f"Done. Adapter saved → {adapter_dir}/final")