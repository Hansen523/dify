import os

from core.model_runtime.model_providers.inscode.text_embedding.text_embedding import InscodeTextEmbeddingModel


def test_embed_documents():
    emb = InscodeTextEmbeddingModel()
    credentials = {
        'inscode_api_url': os.environ.get('inscode_api_url'),
        'inscode_api_key': os.environ.get('inscode_api_key')
    }
    emb._invoke(
        model="gte-base-768",
        credentials=credentials,
        texts=["hello world"],
        user="test",
    )
