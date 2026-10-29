"""Exercise user-visible Streamlit states against the trained local model."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def open_app():
    application = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not application.exception
    return application


def click(application, label):
    next(button for button in application.button if button.label == label).click().run(timeout=30)
    assert not application.exception


def test_examples_score_and_input_edit_clears_old_result():
    application = open_app()
    click(application, "Positive example")
    assert "delicious" in application.text_area[0].value
    click(application, "Analyze sentiment")
    assert any(item.value == "Positive sentiment" for item in application.subheader)
    assert len(application.metric) == 1
    application.text_area[0].set_value("A different review waiting to be scored.").run(timeout=30)
    assert not application.exception
    assert len(application.metric) == 0
    assert not any(item.value == "Positive sentiment" for item in application.subheader)
    click(application, "Negative example")
    click(application, "Analyze sentiment")
    assert any(item.value == "Negative sentiment" for item in application.subheader)


def test_unknown_vocabulary_is_not_presented_as_a_binary_sentiment():
    application = open_app()
    application.text_area[0].set_value("zxqvzzzx qxzzvvqqz").run(timeout=30)
    click(application, "Analyze sentiment")
    assert any(item.value == "No vocabulary match" for item in application.info)
    assert len(application.metric) == 0


def test_empty_text_has_actionable_error():
    application = open_app()
    click(application, "Analyze sentiment")
    assert any("containing words" in item.value for item in application.error)
    assert len(application.metric) == 0


def test_model_report_renders_recorded_metrics_and_scope():
    application = open_app()
    application.radio[0].set_value("Model report").run(timeout=30)
    assert not application.exception
    assert len(application.metric) == 7
    captions = "\n".join(item.value for item in application.caption)
    assert "All held-out rows" in captions
    assert "zero vocabulary matches: 1" in captions
    assert "Rows without alphabetic words removed: 2" in captions
    assert "not a random-guess baseline" in captions
    tables = [item.value for item in application.dataframe]
    source_table = next(table for table in tables if "Source" in table.columns)
    assert source_table.columns.tolist() == ["Source", "Test sentences", "Accuracy", "Positive F1"]


def test_batch_page_requires_upload_and_offers_a_demo_file():
    application = open_app()
    application.radio[0].set_value("Batch review").run(timeout=30)
    assert not application.exception
    assert next(button for button in application.button if button.label == "Analyze CSV").disabled
    assert any("Choose a CSV" in item.value for item in application.info)
