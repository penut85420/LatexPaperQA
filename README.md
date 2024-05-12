# Latex Paper QA

這是一份結合 OpenAI Embedding API 與 ChatGPT API 的論文問答機器人，為示範用的專案。

## 環境

- Ubuntu 22.04
- Pyhton 3.10
  - `tiktoken==0.6.0`
  - `openai==1.28.1`
  - `faiss-cpu==1.8.0`
- 注意：本專案會由 OpenAI 收取額外費用。

## 使用

- 將 OpenAI API Key 放在 `OPENAI_API_KEY` 環境變數裡面。

```sh
# Linux
export OPENAI_API_KEY="sk-..."
# Windows
$env:OPENAI_API_KEY="sk-..."
```

## 檔案

- `demo.count-token.py` 計算 GPT-4 論文總共包含多少 Tokens 在裡面。
- `demo.special-token.py` 示範 Tiktoken 應對 Special Token 時的不同行為。
- `step01.index.py` 對論文進行索引。
- `step02.query.py` 對論文進行詢問。
- `RESP.md` 紀錄範例問答的結果。

## 用法

1. 將想要處理的論文放在 `data` 資料夾裡面。
2. (Optional) 執行索引程式 `step01.index.py` 後，會在 `data` 資料夾底下產生 `embs.npy` 與 `chunks.json` 兩個檔案。
   - 我已經有預先產生好並放上來。
3. 修改 `step02.query.py` 裡面的問題並執行，等待模型生成輸出。

## 授權

本論文所使用之 [GPT-4](https://arxiv.org/abs/2303.08774) 和 [LongLoRA](https://arxiv.org/abs/2309.12307) 的 Latex Source Code 皆來自 [ArXiv](https://arxiv.org/) 網站提供，版權歸屬於原作者。本專案所提供之內容僅供教學範例用途，程式碼本身為 MIT License 授權。
