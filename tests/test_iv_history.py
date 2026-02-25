import tempfile
import unittest
from pathlib import Path

from bot.iv_history import IVHistory


class IVHistoryTests(unittest.TestCase):
    def test_update_and_rank_tracks_percentile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "iv_history.json"
            history = IVHistory(path)

            history.update_and_rank("SPY", 20.0, as_of="2026-01-01")
            history.update_and_rank("SPY", 25.0, as_of="2026-01-02")
            rank = history.update_and_rank("SPY", 30.0, as_of="2026-01-03")

            self.assertGreaterEqual(rank, 90.0)

    def test_same_day_update_replaces_existing_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "iv_history.json"
            history = IVHistory(path)

            history.update_and_rank("QQQ", 18.0, as_of="2026-01-10")
            history.update_and_rank("QQQ", 22.0, as_of="2026-01-10")

            rank = history.percentile_rank("QQQ", 22.0)
            self.assertEqual(rank, 100.0)


if __name__ == "__main__":
    unittest.main()
