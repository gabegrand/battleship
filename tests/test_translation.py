from battleship.translation import Translator


def test_translate():
    translator = Translator()
    translation = translator.translate("def add_one(x):")
    assert isinstance(translation, str)
