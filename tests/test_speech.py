from tsmarker.speech.llm_client import OpenAIClient


def _client():
    return OpenAIClient(api_key="test", base_url="https://example.invalid", model="test")


def test_parse_single_response_plain():
    assert _client()._parse_single_response("AD: 0.95 理由") == 0.95


def test_parse_single_response_brackets():
    assert _client()._parse_single_response("AD: [0.0] 理由") == 0.0


def test_parse_single_response_numbered_brackets():
    assert _client()._parse_single_response("1. AD: [0.3] 理由") == 0.3


def test_parse_single_response_multiline():
    assert _client()._parse_single_response("foo\nAD: [0.5] bar\n") == 0.5


def test_parse_single_response_clamped():
    assert _client()._parse_single_response("AD: 1.5 理由") == 1.0
