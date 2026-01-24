import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_DIR = "model/qwen-embedding"  # 本地目录也可以
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("kb_chunks")

model = SentenceTransformer(MODEL_DIR)

q = "慢性病全称是什么？"
q_emb = model.encode([q], prompt_name="query", normalize_embeddings=True).tolist()

res = col.query(query_embeddings=q_emb, n_results=5)
for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    # print(meta.get("filename"), meta.get("name"))
    print(doc[:200], "\n---")