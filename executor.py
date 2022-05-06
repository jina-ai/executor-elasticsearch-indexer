from jina import Executor, DocumentArray, requests


class ElasticSearchIndexer(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
