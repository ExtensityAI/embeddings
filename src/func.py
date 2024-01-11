from symai import Symbol, Expression
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Expression):
    '''We support all models from https://www.sbert.net/docs/pretrained_models.html#model-overview'''

    def __init__(self, model='all-mpnet-base-v2', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self.model      = SentenceTransformer(model)

    def forward(self, sym: Symbol, *args, **kwargs):
        value = sym.value if isinstance(sym, Symbol) else sym
        return self.model.encode(value)
