"""Minimal text utilities (no external tokenizers)."""


def tokenize_ascii(text, vocab_size=4096):
    """A crude but deterministic tokenizer: one token per ascii word / punct.

    The mock backend uses this so that token counts are real (length-scaled),
    not hand-waved. Non-ascii characters are kept as single tokens so the
    function round-trips any input without loss.
    """
    tokens = []
    buf = []
    for ch in text:
        o = ord(ch)
        if ch.isalnum() and o < 128:
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def word_counts(text):
    """Rough word count for benchmark normalization."""
    return max(1, len(text.split()))
