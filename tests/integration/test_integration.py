import operator

import numpy as np
import pytest
from docarray import Document, DocumentArray
from executor import ElasticSearchIndexer
from helper import assert_document_arrays_equal
from jina import Flow


def test_flow(docker_compose):
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test1', 'n_dim': 2},
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
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test2', 'n_dim': 3},
    )

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)

    with f:
        second_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == len(second_search[0].matches)


def test_persistence(docs, docker_compose):
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test3', 'distance': 'l2_norm'},
    )
    with f:
        f.index(docs)
    indexer = ElasticSearchIndexer(index_name='test3', distance='l2_norm')
    assert_document_arrays_equal(indexer._index, docs)
    indexer.close()


numeric_operators_elasticsearch = {
    'gte': operator.ge,
    'gt': operator.gt,
    'lte': operator.le,
    'lt': operator.lt,
}


@pytest.mark.parametrize('operator', list(numeric_operators_elasticsearch.keys()))
def test_pre_filtering(docker_compose, operator: str):
    n_dim = 2

    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={
            'n_dim': n_dim,
            'columns': [('price', 'float')],
        },
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    with f:

        f.index(docs)

        for threshold in [10, 20, 30]:
            filter_ = {
                'range': {
                    'price': {
                        operator: threshold,
                    }
                }
            }
            doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
            indexed_docs = f.search(doc_query, parameters={'filter': filter_})

            assert len(indexed_docs[0].matches) > 0

            assert all(
                [
                    numeric_operators_elasticsearch[operator](
                        r.tags['price'], threshold
                    )
                    for r in indexed_docs[0].matches
                ]
            )


@pytest.mark.parametrize('operator', list(numeric_operators_elasticsearch.keys()))
def test_filtering(docker_compose, operator: str):
    n_dim = 2
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={
            'n_dim': n_dim,
            'columns': [('price', 'float')],
        },
    )

    docs = DocumentArray([Document(id=f'r{i}', tags={'price': i}) for i in range(50)])
    with f:

        f.index(docs)

        for threshold in [10, 20, 30]:
            filter_ = {
                'range': {
                    'price': {
                        operator: threshold,
                    }
                }
            }
        indexed_docs = f.post('/filter', parameters={'filter': filter_})

        assert len(indexed_docs) > 0

        assert all(
            [
                numeric_operators_elasticsearch[operator](r.tags['price'], threshold)
                for r in indexed_docs
            ]
        )
