import json

from babbler.resources import JsonModel


def test_read_json(tmp_path):
    path = tmp_path / 'object.json'

    class ExampleObject(JsonModel):
        value: str

    with open(path, 'w') as file:
        json.dump({'value': 'hello'}, file)
    obj = ExampleObject.from_json(path)
    assert obj.value == 'hello'
