import os
import time

import numpy as np
from docarray import Document, DocumentArray
import pytest
from elasticsearch import Elasticsearch

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))

@pytest.fixture(scope='module')
def docker_compose():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d --remove-orphans"
    )
    es = Elasticsearch(hosts='http://localhost:9200/')
    while not es.ping():
        time.sleep(0.5)
    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down --remove-orphans"
    )


@pytest.fixture(scope='module')
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(128)),
            Document(id='doc2', embedding=np.random.rand(128)),
            Document(id='doc3', embedding=np.random.rand(128)),
            Document(id='doc4', embedding=np.random.rand(128)),
            Document(id='doc5', embedding=np.random.rand(128)),
            Document(id='doc6', embedding=np.random.rand(128)),
        ]
    )