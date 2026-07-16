import pytest

from daiya_training_harness.prompting import PromptField, PromptTemplate


@pytest.fixture
def prompt_template():
    return PromptTemplate(
        template="Language: {language}\nTerms: {terms}\nAudio: {transcript}",
        fields=(
            PromptField("language", required=False, default="mixed"),
            PromptField("terms", required=False),
            PromptField("transcript"),
        ),
    )


def test_prompt_round_trip_and_render(prompt_template):
    restored = PromptTemplate.from_dict(prompt_template.to_dict())
    assert restored == prompt_template
    assert restored.render({"terms": "Daiya", "transcript": "สวัสดี hello"}) == (
        "Language: mixed\nTerms: Daiya\nAudio: สวัสดี hello"
    )


def test_prompt_rejects_contract_mismatch():
    with pytest.raises(ValueError, match="undeclared placeholders"):
        PromptTemplate("{text} {speaker}", (PromptField("text"),))


def test_prompt_rejects_missing_and_unexpected_fields(prompt_template):
    with pytest.raises(ValueError, match="missing required"):
        prompt_template.render({})
    with pytest.raises(ValueError, match="unexpected prompt fields"):
        prompt_template.render({"transcript": "hello", "typo": "value"})
