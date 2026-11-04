"""Test suite verifying compliance of submission.zip with Google ADK specification."""

import hashlib
import unittest
import zipfile
from pathlib import Path
import yaml


class AdkLoader(yaml.SafeLoader):
    pass

AdkLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))


class TestSubmissionCompliance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zip_path = Path("submission.zip")
        cls.assertTrue(cls.zip_path.exists(), "submission.zip must exist in workspace root")

    def test_zip_integrity(self):
        with zipfile.ZipFile(self.zip_path) as z:
            self.assertIsNone(z.testzip(), "submission.zip must not be corrupt")
            namelist = z.namelist()
            self.assertIn("agent.yaml", namelist)
            self.assertIn("configs/sampling.yaml", namelist)
            self.assertIn("eval_config.yaml", namelist)
            self.assertIn("prompts/system.md", namelist)

    def test_agent_yaml_schema(self):
        with zipfile.ZipFile(self.zip_path) as z:
            content = z.read("agent.yaml").decode("utf-8")
            data = yaml.load(content, Loader=AdkLoader)
            self.assertEqual(data.get("name"), "swe_agent")
            self.assertEqual(data.get("model"), "gemma-4-31b-it-qat-w4a16-ct")
            tools = data.get("tools", [])
            expected_tools = [
                "run_command", "read_file", "edit_file", "write_file",
                "get_status", "submit_patch", "get_code_neighbors",
                "search_similar_code", "get_code_subgraph"
            ]
            self.assertEqual(sorted(tools), sorted(expected_tools))

    def test_sampling_yaml_championship_bounds(self):
        with zipfile.ZipFile(self.zip_path) as z:
            content = z.read("configs/sampling.yaml").decode("utf-8")
            data = yaml.load(content, Loader=AdkLoader)
            self.assertEqual(data.get("max_output_tokens"), 8192, "Must be 8192 to prevent MAX_TOKENS truncation")
            self.assertEqual(data.get("top_p"), 0.95)
            self.assertEqual(data.get("temperature"), 0.2)
            thinking = data.get("thinking_config", {})
            self.assertEqual(thinking.get("thinking_budget"), 4096, "Must be 4096 for deep reasoning")
            self.assertTrue(thinking.get("include_thoughts"), "Must include thoughts")

    def test_eval_config_yaml_verifier_timeout(self):
        with zipfile.ZipFile(self.zip_path) as z:
            content = z.read("eval_config.yaml").decode("utf-8")
            data = yaml.load(content, Loader=AdkLoader)
            eval_dict = data.get("evaluation", {})
            self.assertEqual(eval_dict.get("timeout_seconds"), 240, "Timeout must be 240 to prevent killed tests")
            self.assertEqual(eval_dict.get("max_tool_calls"), 28, "Max tool calls must be 28")
            self.assertEqual(eval_dict.get("max_time_minutes"), 8, "Max time must be 8 minutes")
            self.assertEqual(eval_dict.get("max_turns"), 80)

    def test_system_prompt_winning_tricks(self):
        with zipfile.ZipFile(self.zip_path) as z:
            prompt = z.read("prompts/system.md").decode("utf-8")
            # Trick 1: Repository Root Awareness
            self.assertIn("src/requests/", prompt, "Must map requests to src/requests/")
            self.assertIn("src/ahttpx/", prompt, "Must map httpx to src/ahttpx/")
            # Trick 2: FastAPI tutorial path
            self.assertIn("docs_src", prompt, "Must mention docs_src")
            # Trick 3: 2-3 lines surrounding context
            self.assertTrue("2–3 lines" in prompt or "2-3 lines" in prompt, "Must specify 2-3 surrounding lines")
            # Trick 4: Turn 12 edit deadline
            self.assertIn("12th", prompt, "Must enforce 12th call edit deadline")
            # Trick 5: Anti-tampering auto-revert
            self.assertIn("git checkout -- tests/", prompt, "Must auto-revert tests modifications")


if __name__ == "__main__":
    unittest.main()
