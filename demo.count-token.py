import os
import tiktoken

tk = tiktoken.get_encoding("cl100k_base")

for dir_path, _, file_list in os.walk("gpt-4-paper"):
    for file_name in file_list:
        if not file_name.endswith(".tex"):
            continue
        full_path = os.path.join(dir_path, file_name)
        with open(full_path, "rt", encoding="UTF-8") as fp:
            txt = fp.read()
            print(len(tk.encode(txt, disallowed_special=())))

# 1018 + 31291 + 2156 = 34465
