# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["transformers", "huggingface_hub", "matplotlib", "sentencepiece", "protobuf"]
# ///
"""Tokenizer fertility study: how many tokens each candidate spends on Sinhala.

Run:  uv run analysis/tokenizer_fertility.py
Out:  reports/fertility_report.md, reports/fertility.png
"""
import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ponytail: unsloth mirrors are ungated copies; same tokenizers as google/ and meta-llama/
CANDIDATES = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Gemma-3 (4B/12B)": "unsloth/gemma-3-12b-it",
    "Gemma-4 (E4B/12B)": "google/gemma-4-12B-it",
    "Llama-3.2-3B": "unsloth/Llama-3.2-3B-Instruct",
}
N, SEED, LONG = 1000, 42, 512
OUT = Path("reports")


def load_sample():
    path = hf_hub_download("Jainamshahhh/helalm-data", "helalm_sinhala_instruct.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return random.Random(SEED).sample(rows, N)


def measure(tok, rows):
    # Fertility on completions (clean Sinhala); length stats on prompt+completion (what we train on).
    n_tok = n_word = n_char = 0
    lengths = []
    for r in rows:
        c = r["completion"]
        n_tok += len(tok.encode(c, add_special_tokens=False))
        n_word += len(c.split())
        n_char += len(c)
        lengths.append(len(tok.encode(r["prompt"] + "\n" + r["completion"], add_special_tokens=False)))
    lengths.sort()
    return {
        "vocab": len(tok),
        "tok_per_word": n_tok / n_word,
        "tok_per_char": n_tok / n_char,
        "pct_over_512": 100 * sum(l > LONG for l in lengths) / len(lengths),
        "p50_len": statistics.median(lengths),
        "p95_len": lengths[int(0.95 * len(lengths))],
        "total_tokens": sum(lengths),
    }


def main():
    rows = load_sample()
    results = {name: measure(AutoTokenizer.from_pretrained(mid), rows) for name, mid in CANDIDATES.items()}
    best = min(r["total_tokens"] for r in results.values())

    OUT.mkdir(exist_ok=True)
    lines = [
        f"# Tokenizer fertility on Sinhala\n\n{N} random HelaLM rows (seed {SEED}). "
        "Fertility uses completions; lengths use prompt+completion. "
        "`cost_x` = total tokens relative to the cheapest tokenizer (≈ relative training cost).\n",
        "| Model | Vocab | Tokens/word | Tokens/char | % > 512 tok | p50 len | p95 len | cost_x |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["tok_per_word"]):
        lines.append(
            f"| {name} | {r['vocab']:,} | {r['tok_per_word']:.2f} | {r['tok_per_char']:.3f} | "
            f"{r['pct_over_512']:.1f} | {r['p50_len']:.0f} | {r['p95_len']} | {r['total_tokens'] / best:.2f} |"
        )
    (OUT / "fertility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    names = sorted(results, key=lambda n: results[n]["tok_per_word"])
    plt.figure(figsize=(8, 4))
    plt.bar(names, [results[n]["tok_per_word"] for n in names])
    plt.ylabel("Tokens per Sinhala word (lower = cheaper)")
    plt.title("Tokenizer fertility on Sinhala (HelaLM sample)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fertility.png", dpi=150)


if __name__ == "__main__":
    main()
