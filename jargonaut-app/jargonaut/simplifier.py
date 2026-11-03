"""Step 6: rewrite the clause in plain English with Gemma 4 E4B and the Run 6 LoRA adapter.

SYSTEM_PROMPT, build_messages, render_prompt, clean_output and the model loading
and decoding settings are copied unchanged from Grpproj/jargonaut/common.py, the
code the adapter was trained and validated with: the same system prompt, Gemma's
chat template with thinking off, 4-bit NF4 weights, greedy decoding and at most
512 new tokens. scripts/check_parity.py compares them with the original file.
"""
import re

from .config import ADAPTER_DIR, GEMMA_ID, GEMMA_LOAD_IN_4BIT, GEMMA_MAX_NEW_TOKENS

SYSTEM_PROMPT = (
    "You rewrite contract clauses in plain English for a reader with no legal training. "
    "Use short sentences and everyday words, at about a Grade 8 reading level. "
    "Keep every party, obligation, condition, exception, negation, date and quantity from the clause. "
    "Do not add information, advice or commentary. Reply with the rewrite only."
)


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


def clean_output(text):
    """Strip Gemma's turn/channel markup from decoded text, leaving only the rewrite."""
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
    text = text.split("<turn|>")[0]
    return re.sub(r"<(eos|bos|pad)>", "", text).strip()


class Simplifier:
    def __init__(self, model_id=GEMMA_ID, adapter_dir=ADAPTER_DIR, load_in_4bit=GEMMA_LOAD_IN_4BIT,
                 max_new_tokens=GEMMA_MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        on_gpu = torch.cuda.is_available()
        kwargs = {"device_map": {"": 0} if on_gpu else "cpu", "dtype": torch.bfloat16 if on_gpu else torch.float32}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.eval()

    def rewrite_many(self, sources, batch_size=4):
        """Greedy-decode a rewrite for each clause, in input order (common.generate, unchanged)."""
        torch, tokenizer, model = self.torch, self.tokenizer, self.model
        prompts = [render_prompt(tokenizer, build_messages(s)) for s in sources]
        # Longest first, so an out-of-memory error shows up on the first batch.
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        outputs = [None] * len(prompts)
        tokenizer.padding_side = "left"
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            enc = tokenizer(
                [prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            for i, ids in zip(batch, new_tokens):
                outputs[i] = clean_output(tokenizer.decode(ids, skip_special_tokens=False))
        return outputs

    def rewrite(self, source):
        return self.rewrite_many([source], batch_size=1)[0]
