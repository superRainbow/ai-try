import streamlit as st
from pymilvus import Collection, connections

import constant
from module import embeddings, milvus


def main(connection, collection):
    EMBEDDING_MODEL_NAME = constant.EMBEDDING_MODEL_NAME

    # 只是個初始化
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    # 重新畫所有訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # := (海象運算子 walrus) 是  Python 3.8 語法
    # 這個運算子允許你在同一行內對變數賦值並檢查變數的值, 檢查是否有值
    if prompt := st.chat_input("想問什麼問題勒～"):
        with st.chat_message("user"):
            # 畫出一個 user 頭像的聊天訊息，而且訊息內容是 prompt 裡的值，以 markdown 格式畫出
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.write(prompt)

        with st.chat_message("assistant"):
            print("prompt", prompt)
            query_embedding = embeddings.get_embedding(prompt, EMBEDDING_MODEL_NAME)
            print("query_embedding", query_embedding)
            results = milvus.search(collection, query_embedding, k=5)
            print("results", results)

            text = "可能的答案："
            for result in results:
                text += f"  \n  id：{result[0]} 分數：{result[1]} 文字：{result[2]}"
            st.write(text)
        st.session_state.messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    st.set_page_config(page_title="RAG 聊天", page_icon="🗯️")

    connection = connections.connect("default", host="localhost", port="19530")
    # 連接 collection
    # 如果已經建立過 collection 就註解掉這裡
    COLLECTION_NAME = constant.COLLECTION_NAME

    collection = Collection(COLLECTION_NAME)
    collection.load()
    main(connection, collection)
