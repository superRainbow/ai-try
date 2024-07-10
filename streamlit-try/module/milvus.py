from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

import constant


def create_collection(collection_name):
    fields = [
        # 建立資料類型為INT64的主鍵，且主鍵ID自動遞增
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            descrition="Ids",
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            description="text texts",
            max_length=500,
        ),
        # Milvus 的 FLOAT_VECTOR 支援固定維度的向量，要確保每個向量都有相同的維度
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="Embedding vectors",
            dim=constant.DIM,
        ),
    ]

    # 集合所包含的欄位
    # Schema 定義資料類型和資料屬性的元訊息
    # 每個 collection 都有自己的模式，定義了集合的所有欄位、自動ID（主鍵）分配啟用和集合描述
    schema = CollectionSchema(fields=fields, description="texts collection")
    # 建構集合
    collection = Collection(name=collection_name, schema=schema)
    # 建構索引所需的參數

    # index_type 搜尋方式
    #   FLAT：最適合在小規模，百萬級資料集上尋求完全準確和精確的搜尋結果的場景
    #   IVF_FLAT：量化索引，最適合在精度和查詢速度之間尋求理想平衡的場景
    #   IVF_SQ8：量化索引，最適合在磁碟、CPU和GPU記憶體消耗非常有限的場景中顯著減少資源消耗
    #   IVF_PQ：量化索引，最適合在高查詢速度的情況下以犧牲精確度為代價的場景
    #   HNSW：基於圖形的索引，最適合對搜尋效率有高需求的場景 (目前 Milvus 支援的性能最快的索引，但記憶體的開銷較高)
    #   ANNOY：基於樹狀結構的索引，最適合尋求高召回率的場景

    # metric_type：距離的計算方式
    #   L2：歐氏距離
    #   IP：內積
    #   COSINE：餘弦相似度

    # params：索引參數
    #   nlist：叢集的數量
    #   M：量化索引的參數，M 越大記憶體消耗越高 (通常建議設定在 8-32 之間)
    #   efConstruction：建構時的參數，控制索引時間和索引精準度，越大建構索引越長，但查詢精度越高 (常見參數為 128)
    #   ef：查詢時的參數，控制搜尋精確度和搜尋性能，注意 ef 必須大於 K
    index_params = {
        "index_type": "FLAT",
        "metric_type": "COSINE",
        "params": {"efConstruction": 1500, "M": 1024, "ef": 1000},
    }

    # 在向量上建構索引
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def insert(collection, text_array, embedding_array):
    # zip 對應的元素打包成一個個元組
    entities = [
        {"id": i, "text": sentence, "embedding": embedding}
        for i, (sentence, embedding) in enumerate(zip(text_array, embedding_array))
    ]
    collection.insert(entities)


def search(collection, query_embedding, k):
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 1, "radius": 0.6, "range_filter": 20},
    }

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",  # Search embedding 的欄位
        param=search_params,
        limit=k,  # limit：要回傳的最相似結果的數量
        output_fields=["text"],  # 包含進來要輸出的欄位
        consistency_level="Bounded",
    )

    ret = []
    for hit in results[0]:
        row = []
        row.extend([hit.id, hit.score, hit.entity.get("text")])
        ret.append(row)
    return ret
