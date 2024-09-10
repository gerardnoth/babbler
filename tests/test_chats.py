import babbler
from babbler.chats import Chat


def test_read_chats():
    path = babbler.utils.project_root() / 'test-data/test_chats/chats.jsonl'
    chats = Chat.from_jsonl(path)
    assert len(chats) == 2
    for chat in chats:
        assert chat.messages
        for message in chat.messages:
            assert message.role
            assert message.content
