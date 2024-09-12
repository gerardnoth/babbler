import babbler
from babbler.chats import Chat, GoogleAIChatAdapter, Message, Role, OpenAIChatAdapter


def test_read_chats():
    path = babbler.utils.project_root() / 'test-data/test_chats/chats.jsonl'
    chats = Chat.from_jsonl(path)
    assert len(chats) == 2
    for chat in chats:
        assert chat.messages
        for message in chat.messages:
            assert message.role
            assert message.content


def test_adapt_chat_for_google_ai():
    chat_adapter = GoogleAIChatAdapter()
    messages = [
        Message(role=Role.user, content='message 1'),
        Message(role=Role.assistant, content='message 2'),
    ]
    results = chat_adapter.adapt_messages(messages)
    assert len(results) == 2
    assert results[0]['role'] == 'user'
    assert results[0]['parts'] == ['message 1']

    assert results[1]['role'] == 'model'
    assert results[1]['parts'] == ['message 2']


def test_adapt_chat_for_openai():
    chat_adapter = OpenAIChatAdapter()
    messages = [
        Message(role=Role.user, content='message 1'),
        Message(role=Role.assistant, content='message 2'),
    ]
    results = chat_adapter.adapt_messages(messages)
    assert len(results) == 2
    assert results[0]['role'] == 'user'
    assert results[0]['content'] == 'message 1'

    assert results[1]['role'] == 'assistant'
    assert results[1]['content'] == 'message 2'
