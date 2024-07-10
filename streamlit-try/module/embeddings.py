from sentence_transformers import SentenceTransformer


def get_embedding(input, model_name):
    model = SentenceTransformer(model_name)
    if type(input) is list:
        # return [model.encode(input, normalize_embeddings=True) for text in input]
        return model.encode(input)
    else:
        # return [model.encode(input, normalize_embeddings=True)]
        return model.encode([input])
