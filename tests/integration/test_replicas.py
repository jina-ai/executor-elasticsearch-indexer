from jina import Flow
from docarray import Document
import numpy as np

from executor import ElasticSearchIndexer


def test_replicas(docker_compose):
    n_dim = 10
    
    f = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test1', 'n_dim': n_dim},
    )
    
    docs_index = [Document(embedding=np.random.random(n_dim)) for _ in range(1000)]
    
    docs_query = docs_index[:100]
    
    with f:
        f.post(
            on='/index',
            inputs=docs_index,
        )
    
        docs_without_replicas = f.post(
            on='/search',
            inputs=docs_query,
        )
    
    f_with_replicas = Flow().add(
        uses=ElasticSearchIndexer,
        uses_with={'index_name': 'test1', 'n_dim': n_dim},
        replicas=4
    )
    
    with f_with_replicas :
        docs_with_replicas = f_with_replicas.post(
            on='/search',
            inputs=docs_query,
        )
    
    assert docs_without_replicas == docs_with_replicas
