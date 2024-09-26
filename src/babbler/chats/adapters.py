"""A module for adapters that adapt messages into formats that are compatible with model providers."""

from abc import ABC, abstractmethod
from typing import Iterable, override

from pydantic import Field
from typing_extensions import TypedDict

from google.generativeai.types import ContentDict
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from babbler.files import JSONLWriter
from babbler.resources import Message, Chat, Role, JsonModel
from babbler.types import PathLike


class ChatAdapter[T](ABC):
    """Converts chats into a format suitable for a model provider."""

    @abstractmethod
    def adapt_fine_tune(self, input_path: PathLike, output_path: PathLike) -> None:
        """Adapt a chat file for fine-tuning.

        The output file is in a format suitable for the model provider.

        :param input_path: A path to a chat JSONL file.
        :param output_path: Path to save the output file.
        """
        raise NotImplementedError

    @abstractmethod
    def adapt_message(self, message: Message) -> T:
        """Adapt a message for a provider.

        :param message: A message to adapt.
        :return: An object in a provider format.
        """
        raise NotImplementedError

    def adapt_messages(self, messages: Iterable[Message], target: list[T] | None = None) -> list[T]:
        """Adapt messages for a provider.

        If the target list is not set, a new list is created and returned. Otherwise, the target
        list is appended to and returned.

        :param messages: Messages to adapt.
        :param target: A list to append to.
        :return: Objects in a provider format.
        """
        result: list[T] = [] if target is None else target
        for message in messages:
            result.append(self.adapt_message(message))
        return result


class GoogleAIChatAdapter(ChatAdapter[ContentDict]):
    """Adapts chats for Google AI."""

    @override
    def adapt_fine_tune(self, input_path: PathLike, output_path: PathLike) -> None:
        with JSONLWriter(path=output_path) as writer:
            for chat in Chat.yield_from_jsonl(input_path):
                if chat.system_message:
                    # TODO system messages are not supported yet.
                    pass
                if len(chat.messages) != 2:
                    raise ValueError(
                        f'Google AI only supports tuning single turn messages, but got chat: {chat}'
                    )
                example = {
                    'text_input': chat.messages[0].content,
                    'output': chat.messages[1].content,
                }
                writer.write(example)

    @override
    def adapt_message(self, message: Message) -> ContentDict:
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
                raise ValueError(f'Unsupported role {role}')


class OpenAIChatAdapter(ChatAdapter[ChatCompletionMessageParam]):
    """Adapts chats for OpenAI."""

    @override
    def adapt_fine_tune(self, input_path: PathLike, output_path: PathLike) -> None:
        with JSONLWriter(path=output_path) as writer:
            for chat in Chat.yield_from_jsonl(input_path):
                messages: list[ChatCompletionMessageParam] = []
                if chat.system_message:
                    messages.append(
                        self.adapt_message(
                            Message(
                                role=Role.system,
                                content=chat.system_message,
                            )
                        )
                    )
                self.adapt_messages(chat.messages, target=messages)
                writer.write({'messages': messages})

    @override
    def adapt_message(self, message: Message) -> ChatCompletionMessageParam:
        role = message.role
        match role:
            case Role.assistant:
                return ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content=message.content,
                )
            case Role.system:
                return ChatCompletionSystemMessageParam(
                    role='system',
                    content=message.content,
                )
            case Role.user:
                return ChatCompletionUserMessageParam(
                    role='user',
                    content=message.content,
                )
            case _:
                raise ValueError(f'Unsupported role {role}')


class TextPart(TypedDict):
    """The text part of a conversation."""

    text: str


class Gemini15Message(TypedDict):
    """A message in a Gemini 1.5 chat."""

    role: str
    parts: list[TextPart]


class Gemini15TuneChat(JsonModel):
    """A chat in a format for tuning Gemini 1.5 models."""

    # Gemini 1.5 text datasets use camel case.
    system_instruction: Gemini15Message | None = Field(alias='systemInstruction')
    contents: list[Gemini15Message]


class Gemini15ChatAdapter(ChatAdapter[Gemini15Message]):
    """Adapts chats for Gemini 1.5 on Google Cloud Platform's Vertex AI."""

    @override
    def adapt_fine_tune(self, input_path: PathLike, output_path: PathLike) -> None:
        with JSONLWriter(path=output_path) as writer:
            for chat in Chat.yield_from_jsonl(input_path):
                messages: list[Gemini15Message] = []
                system_instruction: Gemini15Message | None = None
                if chat.system_message:
                    message = Message(role=Role.system, content=chat.system_message)
                    system_instruction = self.adapt_message(message)
                self.adapt_messages(chat.messages, target=messages)
                tune_chat = Gemini15TuneChat(
                    systemInstruction=system_instruction,
                    contents=messages,
                )
                writer.write(tune_chat)

    @override
    def adapt_message(self, message: Message) -> Gemini15Message:
        role: str
        match message.role:
            case Role.assistant:
                role = 'model'
            case Role.user:
                role = 'user'
            case Role.system:
                role = 'system'
            case _:
                raise ValueError(f'Unsupported role: {message.role}')
        return Gemini15Message(
            role=role,
            parts=[TextPart(text=message.content)],
        )
