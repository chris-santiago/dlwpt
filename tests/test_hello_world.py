from dlwpt.hello_world import hello_world


def test_hello_world():
    actual = hello_world()
    expected = 'Hello World.'
    assert actual == expected
