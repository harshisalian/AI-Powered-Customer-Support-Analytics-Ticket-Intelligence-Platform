from app.ml.preprocessing import clean_text, preprocess_ticket, preprocess_text


def test_clean_text_removes_noise() -> None:
    raw_text = "Hello! My payment failed at https://example.com. Email me@test.com ASAP."

    cleaned_text = clean_text(raw_text)

    assert cleaned_text == "hello my payment failed at email asap"


def test_preprocess_text_removes_stop_words_and_lemmatizes() -> None:
    processed_text = preprocess_text("My cards were charged twice and payments failed.")

    assert "my" not in processed_text
    assert "were" not in processed_text
    assert "charge" in processed_text
    assert "payment" in processed_text
    assert "fail" in processed_text


def test_preprocess_ticket_combines_subject_and_description() -> None:
    processed_text = preprocess_ticket(
        "Password reset link not working",
        "I cannot access my account because the login code expired.",
    )

    assert "password" in processed_text
    assert "reset" in processed_text
    assert "login" in processed_text
    assert "account" in processed_text
