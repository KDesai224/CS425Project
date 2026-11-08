"""Shared helpers: paths, JSONL I/O, the prompt, model loading and generation.

Every script uses the same prompt builder, so the model sees an identical input
format during training, at evaluation time and in the final pipeline.
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

SYSTEM_PROMPT = (
    "You rewrite contract clauses in plain English for a reader with no legal training. "
    "Use short sentences and everyday words, at about a Grade 8 reading level. "
    "Keep every party, obligation, condition, exception, negation, date and quantity from the clause. "
    "Do not add information, advice or commentary. Reply with the rewrite only."
)

# LoRA is attached to the text decoder only; Gemma 4 also ships vision and audio
# encoders, which this task never uses.
LORA_TARGETS = r"^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_messages(source, shots=()):
    """Chat messages for one clause. `shots` are (source, rewrite) pairs for few-shot prompting."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_source, shot_rewrite in shots:
        messages.append({"role": "user", "content": shot_source})
        messages.append({"role": "assistant", "content": shot_rewrite})
    messages.append({"role": "user", "content": source})
    return messages


def render_prompt(tokenizer, messages):
    """Text of the prompt up to the point where the model starts writing."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def pick_shots(train_rows, k):
    """Fixed few-shot examples: k shorter-than-median training pairs, evenly spaced by length."""
    by_length = sorted(train_rows, key=lambda r: len(r["source"].split()))
    lower_half = by_length[: max(k, len(by_length) // 2)]
    step = len(lower_half) / (k + 1)
    return [(lower_half[int(step * (i + 1))]["source"], lower_half[int(step * (i + 1))]["reference"]) for i in range(k)]


def load_model(cfg, adapter=None):
    """Load the tokenizer and base model on the GPU, optionally with a trained LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    on_gpu = torch.cuda.is_available()
    kwargs = {"device_map": {"": 0} if on_gpu else "cpu", "dtype": torch.bfloat16 if on_gpu else torch.float32}
    if cfg.get("load_in_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    return model, tokenizer


def clean_output(text):
    """Strip Gemma's turn/channel markup from decoded text, leaving only the rewrite."""
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
    text = text.split("<turn|>")[0]
    return re.sub(r"<(eos|bos|pad)>", "", text).strip()


def generate(model, tokenizer, sources, cfg, shots=()):
    """Greedy-decode a rewrite for each source clause. Returns outputs in input order."""
    import torch

    gen_cfg = cfg["generate"]
    prompts = [render_prompt(tokenizer, build_messages(s, shots)) for s in sources]
    # Longest first, so an out-of-memory error shows up on the first batch.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs = [None] * len(prompts)
    tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()
    for start in range(0, len(order), gen_cfg["batch_size"]):
        batch = order[start : start + gen_cfg["batch_size"]]
        enc = tokenizer(
            [prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=False,
                use_cache=True,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for i, ids in zip(batch, new_tokens):
            outputs[i] = clean_output(tokenizer.decode(ids, skip_special_tokens=False))
    model.train(was_training)
    return outputs
