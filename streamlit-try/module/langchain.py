from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text_using_RCTS(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    split_texts = text_splitter.split_text(pdf_text)

    paragraphs = []
    for text in split_texts:
        paragraphs.extend(text.split("\n"))
    return [text.strip(" ") for text in paragraphs]
