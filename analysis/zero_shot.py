"""Zero-shot Sinhala check: run each candidate on 50 prompts, score script purity.

Kaggle (GPU T4 x2, Internet on):
    !pip install -q -U transformers accelerate bitsandbytes
    !python analysis/zero_shot.py              # all models, skips ones already done
    !python analysis/zero_shot.py gemma-4-12b  # just one
Local self-check (no GPU):  python analysis/zero_shot.py --selftest

Out: eval/results/zero_shot/<model>.jsonl, reports/zero_shot.md, reports/zero_shot_samples.md
"""
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

MODELS = {
    "gemma-4-12b": "google/gemma-4-12B-it",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
}
PROMPTS = Path("eval/prompts_si_50.jsonl")
RESULTS = Path("eval/results/zero_shot")
MAX_NEW, BATCH, PASS = 512, 10, 0.85
N_SAMPLES = 15  # answers per model in the side-by-side reading file


def script_purity(text):
    """Share of letters/marks that are Sinhala (U+0D80–U+0DFF). Digits, punctuation, spaces, ZWJ ignored."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    return sum("඀" <= c <= "෿" for c in letters) / len(letters)


def repetition(text, n=4):
    """Share of repeated word n-grams; > 0.2 means the answer is looping."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def generate(model_id, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    # 4-bit like QLoRA training. fp32 compute: T4 has no bf16 and Gemma overflows in fp16.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32),
        # E4B's per-layer embedding table is ~2.8B params; fp32 (11GB) can't fit one T4.
        dtype=torch.bfloat16 if "E4B" in model_id else torch.float32,
        device_map="auto",
    )
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in prompts
    ]
    outs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i : i + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1] :]:
            n = int((row != tok.pad_token_id).sum())
            outs.append((tok.decode(row, skip_special_tokens=True).strip(), n))
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return outs


def run(name, rows):
    out_file = RESULTS / f"{name}.jsonl"
    if out_file.exists():
        print(f"skip {name} (done)")
        return
    print(f"== {name}", flush=True)
    t0 = time.time()
    outs = generate(MODELS[name], [r["prompt"] for r in rows])
    with open(out_file, "w", encoding="utf-8") as f:
        for r, (text, n) in zip(rows, outs):
            rec = {**r, "model": name, "output": text, "new_tokens": n, "purity": round(script_purity(text), 3)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  done in {time.time() - t0:.0f}s")
    # Kaggle disk is limited and 4 models are ~60GB of weights; drop each after use.
    shutil.rmtree(Path.home() / ".cache/huggingface/hub" / f"models--{MODELS[name].replace('/', '--')}", ignore_errors=True)


def report():
    res = {f.stem: [json.loads(l) for l in open(f, encoding="utf-8")] for f in sorted(RESULTS.glob("*.jsonl"))}
    lines = [
        "# Zero-shot Sinhala check\n",
        f"50 prompts (`{PROMPTS}`), greedy, max {MAX_NEW} new tokens, 4-bit. "
        f"Pass = script purity ≥ {PASS}. Truncated = hit the token cap (often a repetition loop).\n",
        "| Model | Pass rate | Mean purity | Looping (rep4 > 0.2) | Truncated | Mean new tokens |",
        "|---|---|---|---|---|---|",
    ]
    for name, rs in sorted(res.items(), key=lambda kv: -sum(r["purity"] >= PASS for r in kv[1])):
        n = len(rs)
        lines.append(
            f"| {name} | {100 * sum(r['purity'] >= PASS for r in rs) / n:.0f}% | "
            f"{sum(r['purity'] for r in rs) / n:.2f} | {100 * sum(repetition(r['output']) > 0.2 for r in rs) / n:.0f}% | {100 * sum(r['new_tokens'] >= MAX_NEW for r in rs) / n:.0f}% | "
            f"{sum(r['new_tokens'] for r in rs) / n:.0f} |"
        )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/zero_shot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Side-by-side answers for reading by hand; spread across categories.
    ids = sorted({r["id"] for rs in res.values() for r in rs})[:: max(1, 50 // N_SAMPLES)][:N_SAMPLES]
    by_id = {name: {r["id"]: r for r in rs} for name, rs in res.items()}
    s = ["# Zero-shot samples (read and note fluency/correctness)\n"]
    for i in ids:
        first = next(iter(by_id.values()))[i]
        s.append(f"## {i}. [{first['category']}] {first['prompt']}\n")
        for name in res:
            r = by_id[name][i]
            s.append(f"**{name}** (purity {r['purity']}, {r['new_tokens']} tok)\n\n{r['output']}\n")
        s.append("---\n")
    Path("reports/zero_shot_samples.md").write_text("\n".join(s), encoding="utf-8")


def selftest():
    assert script_purity("ආයුබෝවන්") == 1.0
    assert script_purity("hello") == 0.0
    assert script_purity("") == 0.0
    assert script_purity("ශ්‍රී ලංකාව 2024!") == 1.0  # ZWJ, digits, punctuation ignored
    assert 0.4 < script_purity("ලංකාව Lanka") < 0.6
    assert repetition("ද පාර " * 20) > 0.9
    assert repetition("එක දෙක තුන හතර පහ හය") == 0.0
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            run(name, rows)
    else:
        # One process per model: GPU memory isn't reliably freed between models in one process.
        for name in MODELS:
            subprocess.run([sys.executable, __file__, name], check=False)
    report()
