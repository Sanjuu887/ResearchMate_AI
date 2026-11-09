import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitUISmokeTests(unittest.TestCase):
    def test_all_research_modes_render_without_streamlit_exceptions(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.text_area)
        self.assertTrue(any(button.label.startswith("Start research") for button in app.button))

        app.radio[0].set_value("Quick Answer").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input)

        app.radio[0].set_value("Deep Research").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.text_area)
        self.assertTrue(
            any(button.label.startswith("Start deep research") for button in app.button)
        )


if __name__ == "__main__":
    unittest.main()
