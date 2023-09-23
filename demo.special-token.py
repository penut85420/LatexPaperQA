import tiktoken

tk = tiktoken.get_encoding("cl100k_base")

print(tk.encode("<|endofprompt|>"))
# ValueError: Encountered text corresponding to disallowed special token.

print(tk.encode("<|endofprompt|>", disallowed_special=()))
# 當作一般文字來編碼 - [27, 91, 408, 1073, 41681, 91, 29]

print(tk.encode("<|endofprompt|>", allowed_special="all"))
# 當作特殊 Token 來編碼 - [100276]
