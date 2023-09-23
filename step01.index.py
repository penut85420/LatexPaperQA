import json
import os
from tempfile import NamedTemporaryFile as NTF
import numpy as np
import openai
import tiktoken

openai.api_key_path = "API.Key"


DUMP = False


def main():
    data_dir = "data"
    chunks = get_chunks(data_dir, chunk_size=300)
    dump_segments(chunks)
    embs = create_embeddings(chunks)
    dump_data(chunks, embs, data_dir)


def get_chunks(data_dir, chunk_size):
    tk = tiktoken.get_encoding("cl100k_base")

    chunks = list()
    for full_path in iter_tex(data_dir):
        segments = get_segments(full_path)
        segments = [[calc_tokens(tk, seg), seg] for seg in segments]
        segments = process_segments(segments, chunk_size)
        chunks.extend(segments)

        dump_segments(segments)

    return chunks


def iter_tex(data_dir):
    for dir_path, _, file_list in os.walk(data_dir):
        for file_name in file_list:
            if not file_name.endswith(".tex"):
                continue
            full_path = os.path.join(dir_path, file_name)
            yield full_path


def get_segments(full_path):
    with open(full_path, "rt", encoding="UTF-8") as fp:
        text = fp.read().strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text.split("\n")


def calc_tokens(tk: tiktoken.Encoding, seg: str):
    # disallowed_special=() 會將所有文字都當成一般文字
    # 不會將任何 Token 當成 Special Token
    tokens = tk.encode(seg, disallowed_special=())
    return len(tokens)


def process_segments(segments: list[tuple[int, str]], chunk_size):
    print(f"Original Segments: {len(segments)}")
    i = 0
    while i + 1 < len(segments):
        # 取得當前 Chunk 與下個 Chunk 的長度與內容
        seg1_len, seg1_txt = segments[i]
        seg2_len, seg2_txt = segments[i + 1]

        # 若兩個 Chunk 長度相加小於 chunk_size 則合併
        if seg1_len + seg2_len < chunk_size:
            segments[i][0] = seg1_len + seg2_len
            segments[i][1] = seg1_txt + "\n" + seg2_txt
            segments.pop(i + 1)  # 移除已被合併的 Chunk

        # 若 Chunk Size 超過上限則開始處理下一個
        else:
            i += 1
    print(f"Processed Segments: {len(segments)}")
    return [seg[1].strip() for seg in segments]


def dump_segments(segments):
    if not DUMP:
        return
    with NTF("wt", dir=".", delete=False) as fp:
        print(fp.name)
        for i, seg in enumerate(segments):
            fp.write(f"=== Chunk {i} Begin ===\n")
            fp.write(f"{seg}\n")
            fp.write(f"=== Chunk {i} End ===\n\n")


def create_embeddings(chunks):
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=chunks,
    )
    embs = [item["embedding"] for item in resp["data"]]
    embs = np.array(embs)

    print(f"Embedding Shape: {embs.shape}")

    return embs


def dump_data(chunks, embs, data_dir):
    with open(f"{data_dir}/chunks.json", "wt", encoding="UTF-8") as fp:
        json.dump(chunks, fp, ensure_ascii=False)
    np.save(f"{data_dir}/embs.npy", embs)


if __name__ == "__main__":
    main()
