from sentence_transformers import SentenceTransformer, models


class ExpressionEncoder:
    def __init__(self, device, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name,device = device)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        projection = models.Dense(in_features=embedding_dim, out_features=64)
        self.model.add_module('projection', projection)

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)
