import os

import pytest

from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.inscode.inscode import InscodeProvider


def test_validate_provider_credentials():
    provider = InscodeProvider()

    with pytest.raises(CredentialsValidateFailedError):
        provider.validate_provider_credentials(
            credentials={}
        )

    provider.validate_provider_credentials(
        credentials={
            'inscode_api_key': os.environ.get('inscode_api_key')
        }
    )