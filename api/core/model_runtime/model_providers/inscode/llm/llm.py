import base64
import os
import tempfile
import uuid
import requests
import json
from collections.abc import Generator
from typing import Optional, Union, cast

from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)

from core.model_runtime.entities.llm_entities import LLMMode, LLMResult, LLMResultChunk, LLMResultChunkDelta
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel


class InscodeLargeLanguageModel(LargeLanguageModel):
    tokenizers = {}

    def _invoke(self, model: str, credentials: dict,
                prompt_messages: list[PromptMessage], model_parameters: dict,
                tools: Optional[list[PromptMessageTool]] = None, stop: Optional[list[str]] = None,
                stream: bool = True, user: Optional[str] = None) \
        -> Union[LLMResult, Generator]:
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def get_num_tokens(self, model: str, credentials: dict, prompt_messages: list[PromptMessage],
                       tools: Optional[list[PromptMessageTool]] = None) -> int:
        api_key = credentials.get("inscode_api_key")
        api_url = credentials.get("inscode_api_url",
                                  "https://inscode-ai-api.node.inscode.run") + "/api/v1/chat/token_size"
        params = {
            "messages": self._convert_prompt_messages_to_tongyi_messages(prompt_messages),
            "cate": model
        }
        response = requests.post(api_url, headers={
            'Content-Type': 'application/json',
            'Authorization': api_key
        }, json=params, stream=False)
        if response.status_code == 200:
            resp = json.loads(response.text).get("data")
            return resp.get("token_size")
        else:
            raise InvokeBadRequestError(f"model:{model} 获取token长度失败")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[
                    UserPromptMessage(content="ping"),
                ],
                model_parameters={
                    "temperature": 0.5,
                },
                stream=False
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(self, model: str, credentials: dict,
                  prompt_messages: list[PromptMessage], model_parameters: dict,
                  tools: Optional[list[PromptMessageTool]] = None,
                  stop: Optional[list[str]] = None,
                  stream: bool = True,
                  user: Optional[str] = None) -> Union[LLMResult, Generator]:

        api_key = credentials.get("inscode_api_key")
        api_url = credentials.get("inscode_api_url",
                                  "https://inscode-ai-api.node.inscode.run") + "/api/v1/chat/completions"
                      
        print(f"api url: {api_url}")

        mode = self.get_model_mode(model, credentials)

        params = {
            'cate': model,
            'user': user,
            'stream': 'true' if stream else 'false',
            'append': 'false',
            **model_parameters,
        }

        if mode == LLMMode.CHAT:
            params['messages'] = self._convert_prompt_messages_to_tongyi_messages(prompt_messages)
        else:
            params['messages'] = [{
                "role": "user",
                "content": prompt_messages[0].content.rstrip()
            }]
        response = requests.post(api_url, headers={
            'Content-Type': 'application/json',
            'Authorization': api_key
        }, json=params, stream=stream)
        if response.status_code != 200:
            if response.status_code == 400:
                raise InvokeBadRequestError(response.msg)
            else:
                raise ServiceUnavailableError(f"Failed to invoke model {model}, status code: {response.status_code}")
        if stream:
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)

        return self._handle_generate_response(model, credentials, response, prompt_messages)

    def _handle_generate_response(self, model: str, credentials: dict, response: requests.Response,
                                  prompt_messages: list[PromptMessage]) -> LLMResult:
        resp = json.loads(response.text).get("data")
        output = resp.get("output")
        meta = resp.get("meta")
        assistant_prompt_message = AssistantPromptMessage(
            content=output
        )

        usage = self._calc_response_usage(model, credentials, meta.get("in_tokens"), meta.get("gen_tokens"))

        result = LLMResult(
            model=model,
            message=assistant_prompt_message,
            prompt_messages=prompt_messages,
            usage=usage,
        )
        return result

    def _handle_generate_stream_response(self, model: str, credentials: dict,
                                         response: requests.Response,
                                         prompt_messages: list[PromptMessage]) -> Generator:
        for line in response.iter_lines():
            result = line.decode('utf-8')
            if len(line) == 0:
                continue
            if result.startswith("data:"):
                result = result[5:].strip()
            if '[Done]' in result or '[DONE]' in result:
                return
            try:
                result = json.loads(result)
                output = result.get("choices")[0].get("delta").get("content")
                index = result.get("choices")[0].get("index")
                in_tokens = result.get("meta").get("in_tokens")
                gen_tokens = result.get("meta").get("gen_tokens")
                usage = self._calc_response_usage(model, credentials, in_tokens, gen_tokens)

                assistant_prompt_message = AssistantPromptMessage(content=output)

                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=assistant_prompt_message,
                        usage=usage
                    )
                )
            except Exception as e:
                print(e)

    def _convert_prompt_messages_to_tongyi_messages(self, prompt_messages: list[PromptMessage],
                                                    rich_content: bool = False) -> list[dict]:
        """
        Convert prompt messages to tongyi messages

        :param prompt_messages: prompt messages
        :return: tongyi messages
        """
        tongyi_messages = []
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, SystemPromptMessage):
                tongyi_messages.append({
                    'role': 'system',
                    'content': prompt_message.content if not rich_content else [{"text": prompt_message.content}],
                })
            elif isinstance(prompt_message, UserPromptMessage):
                if isinstance(prompt_message.content, str):
                    tongyi_messages.append({
                        'role': 'user',
                        'content': prompt_message.content if not rich_content else [{"text": prompt_message.content}],
                    })
                else:
                    sub_messages = []
                    for message_content in prompt_message.content:
                        if message_content.type == PromptMessageContentType.TEXT:
                            message_content = cast(TextPromptMessageContent, message_content)
                            sub_message_dict = {
                                "text": message_content.data
                            }
                            sub_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.IMAGE:
                            message_content = cast(ImagePromptMessageContent, message_content)

                            image_url = message_content.data
                            if message_content.data.startswith("data:"):
                                # convert image base64 data to file in /tmp
                                image_url = self._save_base64_image_to_file(message_content.data)

                            sub_message_dict = {
                                "image": image_url
                            }
                            sub_messages.append(sub_message_dict)

                    # resort sub_messages to ensure text is always at last
                    sub_messages = sorted(sub_messages, key=lambda x: 'text' in x)

                    tongyi_messages.append({
                        'role': 'user',
                        'content': sub_messages
                    })
            elif isinstance(prompt_message, AssistantPromptMessage):
                content = prompt_message.content
                if not content:
                    content = ' '
                tongyi_messages.append({
                    'role': 'assistant',
                    'content': content if not rich_content else [{"text": content}],
                })
            elif isinstance(prompt_message, ToolPromptMessage):
                tongyi_messages.append({
                    "role": "tool",
                    "content": prompt_message.content,
                    "name": prompt_message.tool_call_id
                })
            else:
                raise ValueError(f"Got unknown type {prompt_message}")

        return tongyi_messages

    def _save_base64_image_to_file(self, base64_image: str) -> str:
        """
        Save base64 image to file
        'data:{upload_file.mime_type};base64,{encoded_string}'

        :param base64_image: base64 image data
        :return: image file path
        """
        # get mime type and encoded string
        mime_type, encoded_string = base64_image.split(',')[0].split(';')[0].split(':')[1], base64_image.split(',')[1]

        # save image to file
        temp_dir = tempfile.gettempdir()

        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{mime_type.split('/')[1]}")

        with open(file_path, "wb") as image_file:
            image_file.write(base64.b64decode(encoded_string))

        return f"file://{file_path}"

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [
                RequestFailure,
            ],
            InvokeServerUnavailableError: [
                ServiceUnavailableError,
            ],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [
                AuthenticationError,
            ],
            InvokeBadRequestError: [
                InvalidParameter,
                UnsupportedModel,
                UnsupportedHTTPMethod,
            ]
        }
