import numpy as np
import pytest
from docarray import Document, DocumentArray
from jina import Flow

from executor import ElasticSearchIndexer


def test_flow(docker_compose):
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test', 'n_dim': 2},
    )

    with f:
        f.post(
            on='/index',
            inputs=[
                Document(id='a', embedding=np.array([1, 3])),
                Document(id='b', embedding=np.array([1, 1])),
                Document(id='c', embedding=np.array([3, 1])),
                Document(id='d', embedding=np.array([2, 3])),
            ],
        )

        docs = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1, 1]))],
        )
        assert docs[0].matches[0].id == 'b'


def test_reload_keep_state(docker_compose):
    docs = DocumentArray([Document(embedding=np.random.rand(3)) for _ in range(2)])
    f = Flow().add(uses=ElasticSearchIndexer, uses_with={'index_name': 'test', 'n_dim': 3},)

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)

    with f:
        second_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == len(second_search[0].matches)