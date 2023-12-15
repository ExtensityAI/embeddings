from symai import Symbol, Expression
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Expression):
    '''We support all models from https://www.sbert.net/docs/pretrained_models.html#model-overview'''

    def __init__(self, model='all-mpnet-base-v2'):
        super().__init__()
        self.model_name = model
        self.model      = SentenceTransformer(model)

    def forward(self, sym: Symbol, *args, **kwargs):
        return self.model.encode(str(sym))
