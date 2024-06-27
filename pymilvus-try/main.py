from pymilvus import connections, utility
from module import milvus, embeddings


def main(connection, collection):
    # 開源的生態系中，最受歡迎的 embedding 模型是 all-MiniLM-L6-v2，有 384 個維度
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    text_array = ["我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離",
                  "而我，在這座城市遺失了你，順便遺失了自己，以為荒唐到底會有捷徑。而我，在這座城市失去了你，輸給慾望高漲的自己，不是你，過分的感情"]

    embedding_array = embeddings.get_embedding(
        text_array, EMBEDDING_MODEL_NAME)
    # embedding_array = get_embedding(text_array, EMBEDDING_MODEL_NAME)
    print('embedding_array', embedding_array)
    milvus.insert(collection, text_array, embedding_array)

    query_text = "工程師寫城市"
    query_embedding = embeddings.get_embedding(
        query_text, EMBEDDING_MODEL_NAME)
    results = milvus.search(collection, query_embedding, k=1)
    print(f"尋找 {query_text}:", results)


if __name__ == '__main__':
    # 連接 pymilvus
    # connections.connect(
    #     alias="default",
    #     user='root',
    #     password='xxx',
    #     host='demodb.tbd.tw',
    #     port='443',
    #     secure=True,
    # )
    connection = connections.connect("default", host="localhost", port="19530")
    # 連接 collection
    # 如果已經建立過 collection 就註解掉這裡
    COLLECTION_NAME = "Lyrics_collection"

    # 檢查某個db有沒有在裡面
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    # collection 類似關聯式資料庫 table
    collection = milvus.create_collection(COLLECTION_NAME)
    collection.load()
    main(connection, collection)
