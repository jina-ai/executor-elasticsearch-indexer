from jina import Executor, requests
from typing import Optional, Dict, Any, List, Union, Mapping, Tuple
from docarray import DocumentArray
from jina.logging.logger import JinaLogger


class ElasticSearchIndexer(Executor):
    """ElasticSearchIndexer indexes Documents into an ElasticSearch service using DocumentArray  with `storage='elasticsearch'`"""
    def __init__(
        self,
        hosts: Union[str, List[Union[str, Mapping[str, Union[str, int]]]], None] = 'http://localhost:9200',
        n_dim: int = 128,
        distance: str = 'cosine',
        index_name: str = 'persisted',
        match_args: Optional[Dict] = None,
        es_config: Optional[Dict[str, Any]] = None,
        index_text: bool = False,
        tag_indices: Optional[List[str]] = None,
        batch_size: int = 64,
        ef_construction: Optional[int] = None,
        m: Optional[int] = None,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        **kwargs,
    ):
        """
        :param hosts: host configuration of the ElasticSearch node or cluster
        :param n_dim: number of dimensions
        :param distance: The distance metric used for the vector index and vector search
        :param index_name: ElasticSearch Index name used for the storage
        :param match_args: the arguments to `DocumentArray`'s match function
        :param es_config: ElasticSearch cluster configuration object
        :param index_text: If set to True, ElasticSearch will index the text attribute of each Document to allow text
            search
        :param tag_indices: Tag fields to be indexed in ElasticSearch to allow text search on them.
        :param batch_size: Batch size used to handle storage refreshes/updates.
        :param ef_construction: The size of the dynamic list for the nearest neighbors. Defaults to the default
            `ef_construction` value in ElasticSearch
        :param m: The maximum number of connections per element in all layers. Defaults to the default
            `m` in ElasticSearch.
        :param columns: precise columns for the Indexer (used for filtering).
        """

        super().__init__(**kwargs)
        self._match_args = match_args or {}

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
                'm': m,
                'columns': columns,
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        """Index new documents
        :param docs: the Documents to index
        """
        self._index.extend(docs)

    @requests(on='/search')
    def search(
            self,
            docs: 'DocumentArray',
            parameters: Dict = {},
            **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint
        """
        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )
        docs.match(self._index, **match_args)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters of the request

        Keys accepted:
            - 'ids': List of Document IDs to be deleted
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
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
        specifications in the `find` method of `DocumentArray` using ElasticSearch: https://docarray.jina.ai/advanced/document-store/elasticsearch/#search-by-filter-query
        :param parameters: parameters of the request, containing the `filter` query
        """
        return self._index.find(filter=parameters.get('filter', None))

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Fill embedding of Documents by id

        :param docs: DocumentArray to be filled with Embeddings from the index
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index"""
        self._index.clear()

    def close(self) -> None:
        super().close()
        self._index.sync()
        del self._index
