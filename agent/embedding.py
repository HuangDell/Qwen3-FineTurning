import json
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.json"
MODEL_DIR = "/path/to/your/qwen3-embedding-4B"  # 本地目录也可以
PERSIST_DIR = "./chroma_db"
COLLECTION = "kb_chunks"

BATCH_SIZE = 32  # 4B 可能需要 16/8，按显存调

def clean_meta(m: dict):
    out = {}
    for k, v in (m or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

# 不开 flash_attention_2 的版本（最稳）
model = SentenceTransformer(MODEL_DIR)

# 如果你已装好 flash-attn，并且环境支持，可改用：
# model = SentenceTransformer(
#     MODEL_DIR,
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

chunks = json.load(open(CHUNKS_PATH, "r", encoding="utf-8"))
ids = [c["chunk_id"] for c in chunks]
docs = [c["text"] for c in chunks]
metas = [clean_meta(c.get("metadata", {})) for c in chunks]

client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_or_create_collection(name=COLLECTION)

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding+Upsert"):
    batch_docs = docs[i:i+BATCH_SIZE]
    embs = model.encode(batch_docs, normalize_embeddings=True)  # (B, dim) numpy
    col.upsert(
        ids=ids[i:i+BATCH_SIZE],
        documents=batch_docs,
        metadatas=metas[i:i+BATCH_SIZE],
        embeddings=embs.tolist(),
    )

print("done, count =", col.count())
print("persist dir =", PERSIST_DIR)