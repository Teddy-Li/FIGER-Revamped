from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, use_fast=True, max_length=128,
                                          padding="max_length", truncation=True)
print(tokenizer([",", "a cat sits on a mat."], add_special_tokens=True, padding=True)["input_ids"])
print(tokenizer.decode([1996, 2316, 2036, 4207, 5779, 2007, 1996, 2714, 1010,]))
