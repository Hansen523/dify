import json
import time
from typing import Optional

import requests

from core.model_runtime.entities.model_entities import PriceType
from core.model_runtime.entities.text_embedding_entities import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.text_embedding_model import (
    TextEmbeddingModel,
)
from core.model_runtime.model_providers.inscode._common import _CommonInscode


class InscodeTextEmbeddingModel(_CommonInscode, TextEmbeddingModel):

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
    ) -> TextEmbeddingResult:
        credentials_kwargs = self._to_credential_kwargs(credentials)

        embeddings, embedding_used_tokens = self.embed_documents(
            credentials_kwargs=credentials_kwargs,
            model=model,
            texts=texts
        )

        return TextEmbeddingResult(
            embeddings=embeddings,
            usage=self._calc_response_usage(model, credentials_kwargs, embedding_used_tokens),
            model=model
        )

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return:
        """
        if len(texts) == 0:
            return 0
        total_num_tokens = 0
        for text in texts:
            total_num_tokens += self._get_num_tokens_by_gpt2(text)

        return total_num_tokens

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            credentials_kwargs = self._to_credential_kwargs(credentials)

            self.embed_documents(credentials_kwargs=credentials_kwargs, model=model, texts=["ping"])
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @staticmethod
    def embed_documents(credentials_kwargs: dict, model: str, texts: list[str]) -> tuple[list[list[float]], int]:
        api_key = credentials_kwargs.get("inscode_api_key")
        api_url = credentials_kwargs.get("inscode_api_url",
                                         "https://inscode-ai-api.node.inscode.run") + "/api/v1/embeddings"

        embeddings = []
        embedding_used_tokens = 0
        for text in texts:
            try:
                params = {
                    "content": text,
                    "dim": 1024 if "_1024" in model else 768
                }
                response = requests.post(api_url, headers={
                    'Content-Type': 'application/json',
                    'Authorization': api_key
                }, json=params, stream=False)

                if response.status_code==200:
                    embeddings.append(json.loads(response.text)["data"][0])
            except Exception as e:
                print(e)

        return [list(map(float, e)) for e in embeddings], embedding_used_tokens

    def _calc_response_usage(
        self, model: str, credentials: dict, tokens: int
    ) -> EmbeddingUsage:

        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens
        )

        # transform usage
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at
        )

        return usage
