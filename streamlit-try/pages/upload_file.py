import streamlit as st
from pymilvus import Collection, connections, utility
from PyPDF2 import PdfReader

import constant
from module import embeddings, langchain, milvus


def main(connection, collection):
    # 開源的生態系中，最受歡迎的 embedding 模型是 all-MiniLM-L6-v2，有 384 個維度
    EMBEDDING_MODEL_NAME = constant.EMBEDDING_MODEL_NAME
    text_array = []

    uploaded_file = st.file_uploader("Embedded 檔案匯入", type=["pdf"], help="檔案為 PDF file")

    if uploaded_file is not None:
        text = ""
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

        if text:  # check if path is not None
            text_array = langchain.split_text_using_RCTS(text)
            print("text", text_array)

    if len(text_array) > 0:
        embedding_array = embeddings.get_embedding(text_array, EMBEDDING_MODEL_NAME)
        milvus.insert(collection, text_array, embedding_array)


if __name__ == "__main__":
    st.set_page_config(page_title="上傳 RAG 檔案", page_icon="📈")

    connection = connections.connect("default", host="localhost", port="19530")
    # 連接 collection
    # 如果已經建立過 collection 就註解掉這裡
    COLLECTION_NAME = constant.COLLECTION_NAME

    # 檢查某個db有沒有在裡面
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
    else:
        collection = milvus.create_collection(COLLECTION_NAME)

    collection.load()
    main(connection, collection)
