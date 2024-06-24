from sentence_transformers import SentenceTransformer

sentences = ["我愛夏天", "芒果"]

# 開源的生態系中，最受歡迎的 embedding 模型是 all-MiniLM-L6-v2，有 384 個維度
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences)


# print 出來 384 個向量
print("Dimesion: ", len(embeddings[0]))
print(embeddings)
