"""
Download TinyStories from HuggingFace and tokenize into binary shards.

Output layout:
  data/cache/train.bin   - uint32 array of token IDs
  data/cache/val.bin     - uint32 array of token IDs
  data/cache/meta.json   - vocab_size, shard sizes, etc.

uint32 is required: cl100k_base vocab = 100277, which exceeds uint16 max (65535).
"""

import json
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR  = Path(__file__).parent
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR   = DATA_DIR / "raw"

DATASET_NAME = "roneneldan/TinyStories"
ENCODING     = "cl100k_base"
BATCH_SIZE   = 1000

# cl100k_base: <|endoftext|> = 100257
EOT_TOKEN = 100257


def _write_split(shard, enc: tiktoken.Encoding, out_path: Path) -> int:
    """
    Tokenize stories in `shard` and stream-write token IDs to `out_path`.

    Processes BATCH_SIZE stories at a time: tokenize → uint32 array → append to
    file. Memory footprint stays constant regardless of dataset size.
    Returns total token count.
    """
    n_total = 0
    with open(out_path, "wb") as f:
        for start in tqdm(range(0, len(shard), BATCH_SIZE), desc=out_path.name, unit="batch"):
            batch = shard[start : start + BATCH_SIZE]["text"]
            ids: list[int] = []
            for text in batch:
                ids.extend(enc.encode_ordinary(text))
                ids.append(EOT_TOKEN)
            chunk = np.array(ids, dtype=np.uint32)
            chunk.tofile(f)
            n_total += len(chunk)
    return n_total


def prepare() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = CACHE_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Cache already exists — {meta['train_tokens']:,} train tokens, "
              f"{meta['val_tokens']:,} val tokens. Delete data/cache/ to re-run.")
        return

    cached = RAW_DIR.exists() and any(RAW_DIR.iterdir())
    print(f"Loading TinyStories ({'from disk cache' if cached else 'downloading from HuggingFace'})...")
    t0 = time.time()
    ds = load_dataset(DATASET_NAME, cache_dir=str(RAW_DIR), trust_remote_code=False)
    print(f"Loaded in {time.time() - t0:.1f}s — "
          f"train: {len(ds['train']):,} stories, "
          f"val: {len(ds['validation']):,} stories")

    enc = tiktoken.get_encoding(ENCODING)
    print(f"Tokenizer: {ENCODING}, vocab size: {enc.n_vocab}\n")

    token_counts: dict[str, int] = {}

    for split, name in [("train", "train"), ("validation", "val")]:
        t0 = time.time()
        out_path = CACHE_DIR / f"{name}.bin"
        n = _write_split(ds[split], enc, out_path)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  {n:,} tokens → {out_path} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
        token_counts[name] = n

    meta = {
        "encoding": ENCODING,
        "vocab_size": enc.n_vocab,
        "eot_token": EOT_TOKEN,
        "train_tokens": token_counts["train"],
        "val_tokens": token_counts["val"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Total tokens: {sum(token_counts.values()):,}")


def main() -> None:
    prepare()


if __name__ == "__main__":
    main()
