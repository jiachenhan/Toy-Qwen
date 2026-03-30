"""
Inspect TinyStories format.

If Arrow cache exists (from toy-prepare), loads from disk instantly.
Otherwise streams from HuggingFace without a full download.

Usage:
    uv run toy-peek
    uv run python -m data.peek
"""

import textwrap
from pathlib import Path

import tiktoken
from datasets import load_dataset

DATA_DIR     = Path(__file__).parent
RAW_DIR      = DATA_DIR / "raw"
DATASET_NAME = "roneneldan/TinyStories"
ENCODING     = "cl100k_base"
N_EXAMPLES   = 3


def main() -> None:
    cached = RAW_DIR.exists() and any(RAW_DIR.iterdir())

    if cached:
        print("Arrow cache found — loading from disk...\n")
        ds = load_dataset(DATASET_NAME, cache_dir=str(RAW_DIR), split="train", trust_remote_code=False)
    else:
        print("No cache — streaming from HuggingFace...\n")
        ds = load_dataset(DATASET_NAME, split="train", streaming=True, trust_remote_code=False)

    enc = tiktoken.get_encoding(ENCODING)

    for i, example in enumerate(ds):
        if i >= N_EXAMPLES:
            break

        text: str = example["text"]
        ids = enc.encode_ordinary(text)

        print(f"{'─' * 60}")
        print(f"Example {i + 1}")
        print(f"{'─' * 60}")
        print(f"Fields : {list(example.keys())}")
        print(f"Text   : {textwrap.shorten(text, width=120, placeholder=' ...')!r}")
        print(f"Length : {len(text)} chars")
        print()
        print(f"Tokens : {len(ids)}")
        print(f"First 16 IDs : {ids[:16]}")
        print(f"Decoded back : {enc.decode(ids[:16])!r}")
        print()

    print(f"{'─' * 60}")
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    print(f"<|endoftext|> token ID : {eot}  (appended after every story during prepare)")
    print(f"Vocab size             : {enc.n_vocab}")
    print(f"Max token ID           : {enc.n_vocab - 1}  →  requires uint32 (> uint16 max 65535)")


if __name__ == "__main__":
    main()
