import tiktoken

tk = tiktoken.get_encoding("cl100k_base")

try:
    print(tk.encode("<|endofprompt|>"))
except Exception as e:
    print(e)
# ValueError: Encountered text corresponding to disallowed special token.

print(tk.encode("<|endofprompt|>", disallowed_special=()))
# 當作一般文字來編碼 - [27, 91, 408, 1073, 41681, 91, 29]

print(tk.encode("<|endofprompt|>", allowed_special="all"))
# 當作特殊 Token 來編碼 - [100276]
