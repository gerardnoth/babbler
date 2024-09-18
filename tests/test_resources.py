from babbler.resources import JsonModel


def test_write_read_json(tmp_path):
    path = tmp_path / 'object.json'

    class ExampleObject(JsonModel):
        value: str

    ExampleObject(value='hello').write_json(path)
    obj = ExampleObject.from_json(path)
    assert obj.value == 'hello'
