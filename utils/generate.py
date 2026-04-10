import tiktoken
import torch
import torch.nn.functional as F

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def generate_sample(model, prompt: str, max_new_tokens: int, temperature: float,
                    context_len: int, device: str) -> str:
    """Greedy/temperature sampling from a text prompt."""
    model.eval()
    ids = _TOKENIZER.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]  # (1, T) where T ≤ context_len
            logits = model(idx_cond)          # (1, T, vocab_size)
            logits = logits[:, -1, :]         # (1, vocab_size) 只需要看最后一个位置的预测
            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True) # [1, 1]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    model.train()
    return _TOKENIZER.decode(idx[0].tolist())
