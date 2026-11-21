'''Bash:
conda run -n dsga1011-hw4-part2 python \
  "/Users/shiyili/Desktop/dsga_1011/assignment/nyu-dsga1011-a4/extra_credit/check_t5_sql_tokenization.py"
'''



from transformers import T5TokenizerFast


def main():
    tokenizer = T5TokenizerFast.from_pretrained(
        "google-t5/t5-small",
        local_files_only=True,
    )

    sql = "flight_1.departure_time <= 1200"
    ids = tokenizer(sql, add_special_tokens=True)["input_ids"]

    print("Original SQL:", sql)
    print("Token IDs:", ids)
    print("Tokens:", tokenizer.convert_ids_to_tokens(ids))

    for skip in (False, True):
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=skip,
            clean_up_tokenization_spaces=False,
        )
        print(f"skip_special_tokens={skip}: {decoded!r}")


if __name__ == "__main__":
    main()
