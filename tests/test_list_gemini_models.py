import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import list_gemini_models


class GeminiModelListTests(unittest.TestCase):
    def test_lists_only_models_supporting_generate_content(self):
        client = Mock()
        client.models.list.return_value = [
            SimpleNamespace(
                name="models/gemini-test-flash",
                supported_actions=["generateContent", "countTokens"],
            ),
            SimpleNamespace(
                name="models/gemini-test-tts",
                supported_actions=["predict"],
            ),
            SimpleNamespace(name="models/gemini-test-unknown", supported_actions=None),
        ]
        with patch.object(list_gemini_models.genai, "Client", return_value=client):
            names = list_gemini_models.list_compatible_models("test-only-key")
        self.assertEqual(names, ["gemini-test-flash"])

    def test_cli_prints_ids_without_printing_the_key(self):
        with patch.object(
            list_gemini_models,
            "get_config",
            return_value=SimpleNamespace(gemini_api_key="test-only-secret"),
        ), patch.object(
            list_gemini_models, "list_compatible_models", return_value=["gemini-test-flash"]
        ), patch("builtins.print") as print_mock:
            result = list_gemini_models.main()

        self.assertEqual(result, 0)
        rendered = " ".join(
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
        )
        self.assertIn("gemini-test-flash", rendered)
        self.assertNotIn("test-only-secret", rendered)


if __name__ == "__main__":
    unittest.main()
