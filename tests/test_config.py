import os
import unittest
from unittest.mock import patch

import config


class ConfigurationTests(unittest.TestCase):
    def test_reads_streamlit_secrets_when_environment_is_unset(self):
        secrets = {
            "GEMINI_API_KEY": "cloud-gemini-key",
            "TAVILY_API_KEY": "cloud-tavily-key",
            "GEMINI_MODEL": "gemini-test-model",
        }
        with patch.object(config, "load_dotenv"), patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "",
                "TAVILY_API_KEY": "",
                "GEMINI_MODEL": "",
                "TAVILY_SEARCH_DEPTH": "",
                "MCP_HOST": "",
                "MCP_PORT": "",
            },
            clear=True,
        ):
            settings = config.load_config(secrets)

        self.assertEqual(settings.gemini_api_key, "cloud-gemini-key")
        self.assertEqual(settings.tavily_api_key, "cloud-tavily-key")
        self.assertEqual(settings.gemini_model, "gemini-test-model")
        self.assertNotIn("cloud-gemini-key", repr(settings))
        self.assertNotIn("cloud-tavily-key", repr(settings))

    def test_environment_values_override_streamlit_secrets(self):
        with patch.object(config, "load_dotenv"), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "environment-key"},
            clear=True,
        ):
            settings = config.load_config({"GEMINI_API_KEY": "cloud-key"})

        self.assertEqual(settings.gemini_api_key, "environment-key")

    def test_request_scoped_config_is_restored_after_use(self):
        default = config.AppConfig(gemini_model="default-model")
        override = config.AppConfig(gemini_model="request-model")
        token = config._active_config.set(default)
        try:
            with config.use_config(override):
                self.assertIs(config.get_config(), override)
            self.assertIs(config.get_config(), default)
        finally:
            config._active_config.reset(token)

    def test_rejects_invalid_mcp_port(self):
        with patch.object(config, "load_dotenv"), patch.dict(
            os.environ, {"MCP_PORT": "not-a-port"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "MCP_PORT"):
                config.load_config()


if __name__ == "__main__":
    unittest.main()
