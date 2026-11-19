from pathlib import Path

from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent / "data"


def read_split(split):
    questions = (DATA_DIR / f"{split}.nl").read_text(
        encoding="utf-8"
    ).splitlines()
    queries = (DATA_DIR / f"{split}.sql").read_text(
        encoding="utf-8"
    ).splitlines()

    if not questions or len(questions) != len(queries):
        raise ValueError(
            f"{split}: found {len(questions)} questions "
            f"and {len(queries)} SQL queries"
        )

    return questions, queries


def preprocess_example(question, query):
    # Apply the same text preprocessing used in your T5 training pipeline.
    return question, query


def token_stats(texts, tokenizer):
    token_ids = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        verbose=False,
    )["input_ids"]

    mean_length = sum(len(ids) for ids in token_ids) / len(token_ids)
    observed_vocab = {token_id for ids in token_ids for token_id in ids}
    return mean_length, len(observed_vocab)


def split_stats(questions, queries, tokenizer):
    question_length, question_vocab_size = token_stats(questions, tokenizer)
    query_length, query_vocab_size = token_stats(queries, tokenizer)

    return {
        "Number of examples": len(questions),
        "Mean sentence length": f"{question_length:.3f}",
        "Mean SQL query length": f"{query_length:.3f}",
        "Vocabulary size (natural language)": question_vocab_size,
        "Vocabulary size (SQL)": query_vocab_size,
    }


def print_table(title, results, include_count):
    print(f"\n{title}")
    print("| Statistic | Train | Dev |")
    print("|---|---:|---:|")

    for name in results["train"]:
        if name == "Number of examples" and not include_count:
            continue
        print(f"| {name} | {results['train'][name]} | {results['dev'][name]} |")


def main():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    before = {}
    after = {}

    for split in ("train", "dev"):
        questions, queries = read_split(split)
        before[split] = split_stats(questions, queries, tokenizer)

        processed_pairs = [
            preprocess_example(question, query)
            for question, query in zip(questions, queries)
        ]
        processed_questions, processed_queries = zip(*processed_pairs)
        after[split] = split_stats(
            list(processed_questions), list(processed_queries), tokenizer
        )

    print_table("Table 1 - Before preprocessing", before, include_count=True)
    print_table("Table 2 - After preprocessing", after, include_count=False)


if __name__ == "__main__":
    main()