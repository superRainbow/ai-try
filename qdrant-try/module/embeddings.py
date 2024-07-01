from sentence_transformers import SentenceTransformer


def get_embedding(input, model_name):
    model = SentenceTransformer(model_name)
    if type(input) is list:
        return [model.encode(text["lyric"]) for text in input]
    else:
        return model.encode(input)
