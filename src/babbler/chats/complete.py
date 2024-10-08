"""A module for completing chats with generative models."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override, Optional

import dotenv
import typer
from tqdm import tqdm
from typing_extensions import Annotated

import openai
import vertexai.generative_models
from google import generativeai as genai
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from babbler.chats.adapters import OpenAIChatAdapter, GoogleAIChatAdapter, Gemini15ChatAdapter
from babbler.files import JSONLWriter
from babbler.resources import Chat, Message, Role, Provider
from babbler.types import PathLike


class ChatCompleter(ABC):
    """A base class for completing chats."""

    @abstractmethod
    def complete(self, chat: Chat) -> Message:
        """Complete a chat and return the completion.

        :param chat: A chat to complete.
        :return: A Completion object.
        """
        raise NotImplementedError


class OpenAiChatCompleter(ChatCompleter):
    """Completes chats with OpenAI models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Create a new instance.

        If an API key is omitted, the client will use the `OPENAI_API_KEY` environment variable
        to set the API key.

        :param api_key: An API key to authenticate requests with.
        :param model: A model to complete chats with. Used if the chat doesn't have a model set.
        """
        self.model = model
        name = 'OPENAI_API_KEY'
        api_key = api_key or os.environ.get(name, None)
        if not api_key:
            logger.warning(
                'The API key is not set. Please set the following environment variable: ' f'{name}'
            )
        self.client = openai.OpenAI(api_key=api_key)
        self.chat_adapter = OpenAIChatAdapter()

    @override
    def complete(self, chat: Chat) -> Message:
        """Complete a chat and return the completion.

        :param chat: A chat to complete.
        :return: A Completion object.
        """
        messages: list[ChatCompletionMessageParam] = []
        if chat.system_message:
            messages.append(
                self.chat_adapter.adapt_message(
                    Message(role=Role.system, content=chat.system_message)
                )
            )
        self.chat_adapter.adapt_messages(chat.messages, target=messages)
        model_name = chat.model or self.model
        if model_name is None:
            raise ValueError(
                f'A default model must be set if the chat does not have a model set: {chat}'
            )
        chat_completion = self.client.chat.completions.create(
            max_completion_tokens=chat.max_tokens,
            messages=messages,
            model=model_name,
            n=1,
            seed=chat.seed,
            temperature=chat.temperature,
        )
        # Because "n = 1" only one choice is generated.
        choice = chat_completion.choices[0]
        return Message(
            role=Role.assistant,
            content=choice.message.content or '',
        )


class GoogleChatCompleter(ChatCompleter):
    """Completes chats with Google AI models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Create a new instance.

        If an API key is omitted, the client will use the `GOOGLE_API_KEY` environment variable
        to set the API key.

        :param api_key: An API key to authenticate requests with.
        :param model: A model to complete chats with. Used if the chat doesn't have a model set.
        """
        self.model = model
        name = 'GOOGLE_API_KEY'
        api_key = api_key or os.environ.get(name, None)
        if not api_key:
            logger.warning(
                'The API key is not set. Please set the following environment variable: ' f'{name}'
            )
        genai.configure(api_key=api_key)
        self.chat_adapter = GoogleAIChatAdapter()

    def complete(self, chat: Chat) -> Message:
        """Complete a chat and return the completion.

        :param chat: A chat to complete.
        :return: A Completion object.
        """
        # See: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        model_name = chat.model or self.model
        if model_name is None:
            raise ValueError(
                f'A default model must be set if the chat does not have a model set: {chat}'
            )
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=chat.system_message,
        )
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=chat.max_tokens,
            seed=chat.seed,
            temperature=chat.temperature,
        )
        contents = self.chat_adapter.adapt_messages(chat.messages)
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE',
                },
            ],
        )
        # The candidate count is set to 1, so only 1 is available.
        content = response.candidates[0].content
        text = content.parts[0].text
        return Message(
            role=Role.assistant,
            content=text,
        )


class Gemini15ChatCompleter(ChatCompleter):
    """Completes chats with Vertex AI Gemini 1.5 models."""

    def __init__(
        self,
        model: str | None = None,
    ):
        """Create a new instance.

        :param model: A model to complete chats with. Used if the chat doesn't have a model set.
        """
        self.model = model
        self.chat_adapter = Gemini15ChatAdapter()

    @override
    def complete(self, chat: Chat) -> Message:
        model_name = chat.model or self.model
        if model_name is None:
            raise ValueError(
                f'A default model must be set if the chat does not have a model set: {chat}'
            )
        model = vertexai.generative_models.GenerativeModel(
            model_name=model_name,
            system_instruction=chat.system_message,
        )
        generation_config = vertexai.generative_models.GenerationConfig(
            candidate_count=1,
            max_output_tokens=chat.max_tokens,
            seed=chat.seed,
            temperature=chat.temperature,
        )
        contents = self.chat_adapter.adapt_messages(chat.messages)
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
        )
        # The candidate count is set to 1, so only 1 is available.
        content = response.candidates[0].content
        text = content.parts[0].text
        return Message(
            role=Role.assistant,
            content=text,
        )


def complete_file(
    input_path: Annotated[
        Path,
        typer.Option(
            default='--input',
            help='The path to the input dataset.',
            exists=True,
            dir_okay=False,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            default='--output',
            help='The file path to save the output.',
            dir_okay=False,
        ),
    ],
    provider: Annotated[Provider, typer.Option(help='The model provider.')],
    model: Annotated[
        Optional[str],
        typer.Option(
            help='The model to complete chats with. Used if the model is not set on the chat.',
        ),
    ] = None,
    env: Annotated[
        Optional[Path],
        typer.Option(
            help='An optional path to a .env file.',
            dir_okay=False,
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            help='Toggle resuming completion from a previous run.',
        ),
    ] = False,
):
    """Complete a conversation with generative model.

    Completions can optionally be resumed. When resuming is enabled and the output file exists,
    keys from completions in the output file are read. Chats associated with the keys will not
    be completed and new completions will be appended to the output file. When resuming is
    disabled, the output file is overwritten.

    Requests to the model provider are authenticated by loading credentials from a .env file.
    A path to a .env file can be provided, otherwise a default location is used.
    """
    dotenv.load_dotenv(dotenv_path=env)
    chat_completer: ChatCompleter
    if provider == Provider.google_ai:
        chat_completer = GoogleChatCompleter(model=model)
    elif provider == Provider.openai:
        chat_completer = OpenAiChatCompleter(model=model)
    elif provider == Provider.vertexai:
        chat_completer = Gemini15ChatCompleter(model=model)
    else:
        raise ValueError(f'Unsupported provider: {provider}')
    keys: set[str] = set()
    if resume and output_path.exists():
        keys = _find_keys(output_path)
    with JSONLWriter(path=output_path, append=resume) as writer:
        for i, chat in enumerate(tqdm(Chat.yield_from_jsonl(input_path), desc='Completing chats')):
            # Skip chats that already exist in the output file.
            if chat.key and chat.key in keys:
                continue
            if not chat.messages:
                raise ValueError(f'No messages in chat at index {i}: {chat}')
            message = chat_completer.complete(chat=chat)
            chat.messages.append(message)
            writer.write(chat)


def _find_keys(path: PathLike) -> set[str]:
    """Find chat keys in a chat JSONL file.

    :param path: Path to a file.
    :return: A set of keys or an empty set if none are found.
    """
    keys: set[str] = set()
    no_keys = 0
    for chat in tqdm(Chat.yield_from_jsonl(path), desc='Reading chat keys'):
        key = chat.key
        if key:
            if key in keys:
                logger.warning(f'Duplicate key found: {key}')
            keys.add(key)
        else:
            no_keys += 1
    if no_keys > 0:
        logger.warning(
            f'The output file contains completions without keys. Total missing keys: {no_keys}'
        )
    logger.debug(f'Total completions that will be skipped: {len(keys)}')
    return keys
