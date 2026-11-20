from ec.judge.classifiers import parse_harmbench, parse_wildguard


def test_parse_wildguard():
    r = parse_wildguard("Harmful request: yes\nResponse refusal: no\nHarmful response: yes")
    assert r == {"harmful_request": True, "refusal": False, "harmful": True}
    assert parse_wildguard("garbage")["refusal"] is None


def test_parse_harmbench():
    assert parse_harmbench("Yes") is True
    assert parse_harmbench(" no") is False
    assert parse_harmbench("maybe") is None
