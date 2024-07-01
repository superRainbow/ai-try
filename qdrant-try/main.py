from module import qdrant, embeddings


def main():
    COLLECTION_NAME = "Lyrics_collection"
    qclient = qdrant.connection(COLLECTION_NAME)

    # 開源的生態系中，最受歡迎的 embedding 模型是 all-MiniLM-L6-v2，有 384 個維度
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    text_array = [
        {
            "id": 1,
            "lyric": "我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離"
        },
        {
            "id": 2,
            "lyric": "而我，在這座城市遺失了你，順便遺失了自己，以為荒唐到底會有捷徑。而我，在這座城市失去了你，輸給慾望高漲的自己，不是你，過分的感情"
        }
    ]

    embedding_array = embeddings.get_embedding(
        text_array, EMBEDDING_MODEL_NAME)

    qdrant.upsert_vector(qclient, COLLECTION_NAME, embedding_array, text_array)

    query_text = "工程師寫城市"
    query_embedding = embeddings.get_embedding(
        query_text, EMBEDDING_MODEL_NAME)
    results = qdrant.search_from_qdrant(
        qclient, COLLECTION_NAME, query_embedding, k=1)
    print(f"尋找 {query_text}:", results)


if __name__ == '__main__':
    main()
