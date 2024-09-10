import json

from babbler import files


def test_read_lines(tmp_path):
    path = tmp_path / 'texts.txt'
    count = 3
    with open(path, 'w', encoding='utf-8') as file:
        for i in range(count):
            file.write(f'{i}\n')

    lines = list(files.yield_lines(path))
    assert len(lines) == count
    for i, line in enumerate(lines):
        assert f'{i}\n' == line


def test_read_jsonl(tmp_path):
    path = tmp_path / 'blobs.jsonl'
    count = 3
    with open(path, 'w', encoding='utf-8') as file:
        for i in range(count):
            line = json.dumps({str(i): i}) + '\n'
            file.write(line)

    blobs = list(files.yield_jsonl(path))
    assert len(blobs) == count
    for i, blob in enumerate(blobs):
        assert blob[str(i)] == i
