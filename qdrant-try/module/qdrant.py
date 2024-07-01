from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct


def connection(collection_name):
    client = QdrantClient("http://localhost:6333")

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            # 向量相似度演算法 (Cosine 餘弦相似度 | Euclid 歐氏距離 | Dot 點積)
            distance=models.Distance.COSINE,
            size=384),  # 注意：向量大小（向量維度）要指定為你使用的模型大小
        optimizers_config=models.OptimizersConfigDiff(
            memmap_threshold=20000),  # 有關最佳化器
        hnsw_config=models.HnswConfigDiff(
            on_disk=True, m=16, ef_construct=100)  # 有關索引詳情
    )
    return client


def upsert_vector(client, collection_name, embedding_array, text_array):
    # points：相當 mysql 一行資料
    points = [
        PointStruct(id=i, vector=embedding, payload=sentence)
        for i, (sentence, embedding) in enumerate(zip(text_array, embedding_array))
    ]
    client.upsert(collection_name=collection_name, points=points)
    print("upsert finish")


def search_from_qdrant(client, collection_name, vector, k=1):
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=k,  # 回傳幾則資料
        append_payload=True,
    )
    return search_result
