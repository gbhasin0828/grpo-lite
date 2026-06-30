import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list,
    device: str,
    args,
) -> tuple:
    """
    Generate 16 story completions for one opening line.

    Args:
        model     : the LoRA-wrapped policy model
        tokenizer : Qwen tokenizer
        messages  : chat messages list already built by collate_fn
                    [{"role": "system", ...}, {"role": "user", ...}]
        device    : "cuda" or "cpu"
        args      : training args (num_chains, max_prompt_length,
                    max_completion_length, temperature)

    Returns:
        prompt_completion_ids : (num_chains, prompt_len + completion_len)
        prompt_ids            : (num_chains, prompt_len)
        completion_ids        : (num_chains, completion_len)
        attention_mask        : (num_chains, prompt_len + completion_len)
        completions_text      : list of 16 decoded story strings
        prompt_text           : the full formatted prompt string
    """

    # ------------------------------------------------------------------
    # 1. Tokenize prompt
    # ------------------------------------------------------------------
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    prompt_ids  = prompt_inputs["input_ids"]
    prompt_mask = prompt_inputs["attention_mask"]

    # ------------------------------------------------------------------
    # 2. Truncate + repeat for num_chains (16 parallel generations)
    # ------------------------------------------------------------------
    prompt_ids  = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]

    prompt_ids  = prompt_ids.repeat(args.num_chains, 1).to(device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(device)

    # ------------------------------------------------------------------
    # 3. Generate
    # ------------------------------------------------------------------
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_completion_ids = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config,
    )

    # ------------------------------------------------------------------
    # 4. Split prompt / completion
    # ------------------------------------------------------------------
    prompt_length  = prompt_ids.size(1)
    prompt_ids     = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # ------------------------------------------------------------------
    # 5. Build completion mask (1 up to and including EOS, 0 after)
    # ------------------------------------------------------------------
    is_eos   = completion_ids == tokenizer.eos_token_id
    eos_idx  = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
    )
    eos_idx[is_eos.any(dim=1)] = (
        is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    )
    seq_idx        = torch.arange(is_eos.size(1), device=device)
    seq_idx        = seq_idx.expand(is_eos.size(0), -1)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # ------------------------------------------------------------------
    # 6. Decode to text
    # ------------------------------------------------------------------
    completions_text = tokenizer.batch_decode(
        completion_ids, skip_special_tokens=True
    )

    return (
        prompt_completion_ids,
        prompt_ids,
        completion_ids,
        attention_mask,
        completions_text,
        prompt_text,
    )