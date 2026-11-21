'''Bash:
conda run -n dsga1011-hw4-part2 python \
  "/Users/shiyili/Desktop/dsga_1011/assignment/nyu-dsga1011-a4/extra_credit/export_t5-small_tokenizer_vocab.py"
'''


from pathlib import Path
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained(
    "google-t5/t5-small", local_files_only=True
)
vocab = tokenizer.get_vocab()

output = Path(__file__).resolve().with_name("t5-small-vocab.tsv")
output.write_text(
    "".join(
        f"{token_id}\t{token}\n"
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
    ),
    encoding="utf-8",
)
print(f"{len(vocab):,} tokens written to {output}")