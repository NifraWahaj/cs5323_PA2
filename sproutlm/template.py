"""The single prompt format shared by SFT, DPO and generation.

We use Qwen's ChatML markers (they already exist in the base vocabulary) but write
the string by hand instead of calling `apply_chat_template`, so the exact bytes the
model sees are visible here and cannot drift between library versions.

    <|im_start|>system
    {SYSTEM_LINE}<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|endoftext|>

The response is closed by the tokenizer's EOS (<|endoftext|> for Qwen2.5 base), which
packing.py appends. At generation time we stop on either EOS or <|im_end|>.
"""

from __future__ import annotations

SYSTEM_LINE = "You are Sprout, a small assistant. Answer directly, stay accurate, and decline unsafe requests briefly."

_HEAD = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def render_prompt(instruction: str, system: str = SYSTEM_LINE) -> str:
    """Instruction text -> the full prompt string the model is conditioned on."""
    return _HEAD.format(system=system, instruction=instruction.strip())


def clean_response(text: str) -> str:
    """Trim whitespace and anything after a stray stop marker inside a target/generation."""
    for marker in STOP_STRINGS:
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut]
    return text.strip()


def stop_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for marker in STOP_STRINGS:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)
