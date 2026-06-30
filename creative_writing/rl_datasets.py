import json
import random
from torch.utils.data import Dataset, DataLoader


class CreativeWritingDataset(Dataset):
    """
    Loads opening lines from prompts.json.
    Each item is one prompt dict: {"id": 1, "genre": "...", "text": "..."}
    """

    def __init__(self, prompts_path: str, shuffle: bool = True, seed: int = 42):
        with open(prompts_path, "r") as f:
            data = json.load(f)

        self.prompts = data["prompts"]          # list of 75 dicts

        if shuffle:
            random.seed(seed)
            random.shuffle(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx % len(self.prompts)]   # wraps around safely


def collate_fn(batch):
    """
    Converts a batch of prompt dicts into the format train_lora.py expects.

    Input  (batch of 1):
        [{"id": 1, "genre": "literary_dramatic", "text": "She had rehearsed..."}]

    Output:
        {
            "prompt_text" : "She had rehearsed...",
            "genre"       : "literary_dramatic",
            "id"          : 1,
            "messages"    : [{"role": "user", "content": "Continue this story: \"She had rehearsed...\""}]
        }

    The "messages" key is what gets tokenized and fed to Qwen.
    The opening line is wrapped in quotes so _extract_opening_line()
    in evaluator.py can reliably pull it back out.
    """
    item = batch[0]   # one prompt per training step, same as GSM8K

    opening_line = item["text"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative writer. "
                "Given an opening line, continue the story in 3-5 sentences. "
                "Be specific, surprising, and avoid clichés."
            ),
        },
        {
            "role": "user",
            "content": f'Continue this story: "{opening_line}"',
        },
    ]

    return {
        "prompt_text": opening_line,
        "genre":       item["genre"],
        "id":          item["id"],
        "messages":    messages,
    }


def get_creative_dataloader(
    prompts_path: str,
    shuffle: bool = True,
    seed: int     = 42,
) -> DataLoader:
    """
    Returns a DataLoader that serves one prompt per training step.

    Usage in train_lora.py:
        dataloader = get_creative_dataloader("prompts.json")
        for batch in dataloader:
            opening_line = batch["prompt_text"]
            messages     = batch["messages"]
            ...
    """
    dataset = CreativeWritingDataset(prompts_path, shuffle=shuffle, seed=seed)

    return DataLoader(
        dataset,
        batch_size=1,           # one prompt per step, same as GSM8K
        shuffle=False,          # shuffling handled inside Dataset
        collate_fn=collate_fn,
    )