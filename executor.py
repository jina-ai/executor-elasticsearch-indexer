from jina import Executor, requests
from typing import Optional, Dict, Any, List, Union, Mapping
from docarray import DocumentArray
from jina.logging.logger import JinaLogger


class ElasticSearchIndexer(Executor):
    def __init__(
        self,
        hosts: Union[str, List[Union[str, Mapping[str, Union[str, int]]]], None] = 'http://localhost:9200',
        n_dim: int = 128,
        distance: str = 'cosine',
        index_name: str = 'Persisted',
        es_config: Optional[Dict[str, Any]] = None,
        index_text: bool = False,
        tag_indices: Optional[List[str]] = None,
        batch_size: int = 64,
        ef_construction: Optional[int] = None,
        m: Optional[int] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self._index = DocumentArray(
            storage='elasticsearch',
            config={
                'hosts': hosts,
                'n_dim': n_dim,
                'distance': distance,
                'index_name': index_name,
                'es_config': es_config or {},
                'index_text': index_text,
                'tag_indices': tag_indices or [],
                'batch_size': batch_size,
                'ef_construction': ef_construction,
                'm': m
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._index.extend(docs)

    @requests(on='/search')
    def search(
            self,
            docs: 'DocumentArray',
            **kwargs,
    ):
        """
        Perform a vector similarity search and retrieve the full Document match
        :param docs: the Documents to search with
        function. They overwrite the original match_args arguments.
        """
        docs.match(self._index)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """
        Delete entries from the index by id
        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """
        Update doc with the same id, if not present, append into storage
        :param docs: the documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray`: https://docarray.jina.ai/fundamentals/documentarray/find/#filter-with-query-operators
        :param parameters: parameters of the request
        """
        return self._index.find(parameters['query'])

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """
        retrieve embedding of Documents by id
        :param docs: DocumentArray to search with
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """
        clear the database
        """
        self._index.clear()

    def close(self) -> None:
        del self._index
