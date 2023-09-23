import json

import numpy as np
import openai

openai.api_key_path = "API.Key"

from faiss import IndexFlatL2


def main():
    query_text = "請介紹這篇論文"
    chunks, vectors = load_data("data/chunks.json", "data/embs.npy")
    query_emb = get_query_emb(query_text)
    prompt = build_prompt(query_emb, query_text, vectors, chunks)
    response = create_chat(prompt)
    stream_response(response)


def get_query_emb(query_text: str):
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query_text],
    )
    query_emb = resp["data"][0]["embedding"]
    return np.array([query_emb])


def load_data(chunk_path, emb_path):
    with open(chunk_path, "rt", encoding="UTF-8") as fp:
        chunks = json.load(fp)
    key_emb: np.ndarray = np.load(emb_path)
    vectors = IndexFlatL2(key_emb.shape[1])
    vectors.add(key_emb)

    return chunks, vectors


def build_prompt(q_emb, q_text, vectors: IndexFlatL2, value_chunks):
    dist, index = vectors.search(q_emb, k=5)
    print(f"Distance: {dist}")
    print(f"Indices: {index}")
    prompts = [value_chunks[i] for i in reversed(index[0])]
    prompts.append(f"問題：{q_text}")
    return "\n\n".join(prompts)


def create_chat(prompt):
    sys_prompt = "你現在是個專業的文件檢索問答機器人，請根據文件內容的資訊回答問題。"
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )


def stream_response(response):
    for resp in response:
        try:
            token = resp["choices"][0]["delta"]["content"]
            print(end=token, flush=True)
        except:
            pass
    print()


if __name__ == "__main__":
    main()

"""

"""
