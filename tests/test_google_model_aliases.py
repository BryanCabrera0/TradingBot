import unittest

from bot.google_model_aliases import (
    GOOGLE_GEMINI_FLASH_MODEL,
    GOOGLE_GEMINI_PRO_MODEL,
    candidate_google_models,
    reset_google_model_cache,
    remember_google_model_success,
)


class GoogleModelAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_google_model_cache()

    def tearDown(self) -> None:
        reset_google_model_cache()

    def test_primary_model_candidates_include_known_fallbacks(self) -> None:
        candidates = candidate_google_models(GOOGLE_GEMINI_PRO_MODEL)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], GOOGLE_GEMINI_PRO_MODEL)
        self.assertIn("gemini-3.1-pro-preview", candidates)

    def test_flash_model_candidates_include_known_fallbacks(self) -> None:
        candidates = candidate_google_models(GOOGLE_GEMINI_FLASH_MODEL)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0], GOOGLE_GEMINI_FLASH_MODEL)
        self.assertIn("gemini-3-flash-preview", candidates)

    def test_cached_resolution_is_prioritized(self) -> None:
        remember_google_model_success(
            GOOGLE_GEMINI_PRO_MODEL,
            "gemini-3.1-pro-preview",
        )
        candidates = candidate_google_models(GOOGLE_GEMINI_PRO_MODEL)
        self.assertEqual(candidates[0], "gemini-3.1-pro-preview")


if __name__ == "__main__":
    unittest.main()
