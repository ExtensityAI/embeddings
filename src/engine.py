import logging

from typing import Optional
from sentence_transformers import SentenceTransformer

from symai.backend.engine.base import Engine
from symai.backend.engine.settings import SYMAI_CONFIG


logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class EmbeddingEngine(Engine):
    def __init__(self, model: Optional[str] = None):
        super().__init__()
        logger = logging.getLogger('sentence_transformers')
        logger.setLevel(logging.WARNING)
        self.config             = SYMAI_CONFIG
        self.model_name         = self.config['EMBEDDING_ENGINE_MODEL'] if model is None else model
        self.model              = SentenceTransformer(self.model_name)
        self.pricing            = self.api_pricing()
        self.max_tokens         = self.api_max_tokens()

    def id(self) -> str:
        if  'mpnet' in self.config['EMBEDDING_ENGINE_MODEL']:
            return 'embedding'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'EMBEDDING_ENGINE_MODEL' in kwargs:
            self.model_name = kwargs['EMBEDDING_ENGINE_MODEL']
            self.model      = SentenceTransformer(self.model_name)

    def forward(self, argument):
        prepared_input = argument.prop.prepared_input
        args            = argument.args
        kwargs          = argument.kwargs

        input_          = prepared_input if isinstance(prepared_input, list) else [prepared_input]
        except_remedy   = kwargs['except_remedy'] if 'except_remedy' in kwargs else None

        try:
            res = self.model.encode(input_)
        except Exception as e:
            if except_remedy is None:
                raise e
            callback = self.model.encode
            res = except_remedy(e, input_, callback, self, *args, **kwargs)

        rsp = [res]

        metadata = {}
        return [rsp], metadata

    def prepare(self, argument):
        assert not argument.prop.processed_input, "EmbeddingEngine does not support processed_input."
        argument.prop.prepared_input = argument.prop.entries
