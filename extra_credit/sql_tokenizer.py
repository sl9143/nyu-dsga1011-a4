from transformers import T5TokenizerFast


def load_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    tokenizer.add_tokens(["<", "<=", "<>"])    # add_tokens() function added ["<", "<=", "<>"] as common token but not special token, which worked as our expectation.
    return tokenizer