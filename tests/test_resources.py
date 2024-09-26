import babbler
from babbler.resources import JsonModel, Chat


def test_write_read_json(tmp_path):
    path = tmp_path / 'object.json'

    class ExampleObject(JsonModel):
        value: str

    ExampleObject(value='hello').write_json(path)
    obj = ExampleObject.from_json(path)
    assert obj.value == 'hello'


def test_read_chats():
    path = babbler.utils.project_root() / 'test-data/test_chats/chats.jsonl'
    chats = Chat.from_jsonl(path)
    assert len(chats) == 2
    for chat in chats:
        assert chat.messages
        for message in chat.messages:
            assert message.role
            assert message.content
