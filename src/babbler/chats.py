"""Provides classes for completing chats with generative models."""

import os
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional
from typing import override

import dotenv
import google.generativeai as genai
import openai
import typer
from google.generativeai.types import ContentsType, ContentDict
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field
from tqdm import tqdm
from typing_extensions import Annotated

from babbler.resources import JsonModel, Provider


class Role(str, Enum):
    """The role of an entity in a conversation."""

    assistant = 'assistant'
    """A role the model adopts."""

    user = 'user'
    """A role the for entities external to the model, such as a user or agent."""


class Message(JsonModel):
    """A message in a conversation with a generative model."""

    role: Role
    content: str


class Chat(JsonModel):
    """A conversation with a generative model."""

    key: str | None = None
    model: str | None = None
    temperature: float | None = None
    system_message: str | None = None
    messages: list[Message] = Field(default_factory=list)


class Completion(JsonModel):
    """A completion from a generative model."""

    key: str | None
    message: Message


class ChatCompleter(ABC):
    """A base class for completing chats."""

    @abstractmethod
    def complete(self, chat: Chat) -> Completion:
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

    @override
    def complete(self, chat: Chat) -> Completion:
        """Complete a chat and return the completion.

        :param chat: A chat to complete.
        :return: A Completion object.
        """
        messages: list[ChatCompletionMessageParam] = []
        if chat.system_message:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role='system',
                    content=chat.system_message,
                )
            )
        for message in chat.messages:
            created = OpenAiChatCompleter.create_message(message)
            messages.append(created)
        model_name = chat.model or self.model
        if model_name is None:
            raise ValueError(
                f'A default model must be set if the chat does not have a model set: {chat}'
            )
        chat_completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=chat.temperature,
            n=1,
        )
        # Because "n = 1" only one choice is generated.
        choice = chat_completion.choices[0]
        message = Message(
            role=Role.assistant,
            content=choice.message.content or '',
        )
        return Completion(
            key=chat.key,
            message=message,
        )

    @staticmethod
    def create_message(message: Message) -> ChatCompletionMessageParam:
        """Create a message in a format for the provider.

        :param message: Input to map.
        :return: A new message.
        """
        role = message.role
        match role:
            case 'assistant':
                return ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content=message.content,
                )
            case 'user':
                return ChatCompletionUserMessageParam(
                    role='user',
                    content=message.content,
                )
            case _:
                raise ValueError(f'Unrecognized role {role}')


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

    def complete(self, chat: Chat) -> Completion:
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
            temperature=chat.temperature,
        )
        contents: list[ContentsType] = []
        for msg in chat.messages:
            contents.append(
                {
                    'role': msg.role,
                    'parts': msg.content,
                }
            )
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
        )
        # The candidate count is set to 1, so only 1 is available.
        content = response.candidates[0].content
        text = content.parts[0].text
        message = Message(
            role=content.role,
            content=text,
        )
        return Completion(
            key=chat.key,
            message=message,
        )

    @staticmethod
    def create_message(message: Message) -> ContentDict:
        """Create a message in a format for the provider.

        :param message: Input to map.
        :return: A new message.
        """
        role = message.role
        parts = [message.content]
        match role:
            case 'assistant':
                return ContentDict(
                    role='model',
                    parts=parts,
                )
            case 'user':
                return ContentDict(
                    role='user',
                    parts=parts,
                )
            case _:
                raise ValueError(f'Unrecognized role {role}')


def complete(
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
    else:
        raise ValueError(f'Unsupported provider: {provider}')
    mode = 'w'
    keys: set[str] = set()
    if resume and output_path.exists():
        mode = 'a'
        keys = _find_keys(output_path) if resume else set()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, mode) as file:
        for i, chat in enumerate(tqdm(Chat.yield_from_jsonl(input_path), desc='Completing chats')):
            # Skip chats that already exist in the output file.
            if chat.key and chat.key in keys:
                continue
            if not chat.messages:
                raise ValueError(f'No messages in chat at index {i}: {chat}')
            completion = chat_completer.complete(chat=chat)
            file.write(completion.model_dump_json())
            file.write('\n')


def _find_keys(path: PathLike) -> set[str]:
    """Find chat keys in a chat JSONL file.

    :param path: Path to a file.
    :return: A set of keys or an empty set if none are found.
    """
    keys: set[str] = set()
    no_keys = 0
    for completion in tqdm(Completion.yield_from_jsonl(path), desc='Reading keys'):
        key = completion.key
        if key:
            if key in keys:
                logger.warning(f'Duplicate key found: {key}')
            keys.add(key)
        else:
            no_keys += 1
    if no_keys > 0:
        logger.warning(
            'The output file contains completions without keys. ' f'Total missing keys: {no_keys}'
        )
    logger.debug(f'Total completions that will be skipped: {len(keys)}')
    return keys
