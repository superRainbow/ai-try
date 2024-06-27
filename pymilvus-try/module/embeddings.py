from sentence_transformers import SentenceTransformer


def get_embedding(input, model_name):
    model = SentenceTransformer(model_name)
    if type(input) == list():
        return [model.encode(text) for text in input]
    else:
        return model.encode(input)
