import os
import stat
import tempfile
import unittest
from pathlib import Path

from bot.file_security import (
    atomic_write_private,
    tighten_file_permissions,
    validate_sensitive_file,
)


class FileSecurityTests(unittest.TestCase):
    def test_atomic_write_private_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "state.json"

            atomic_write_private(path, '{"ok": true}', label="test file")

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok": true}')
            if os.name == "posix":
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode & 0o077, 0)

    def test_validate_sensitive_file_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "target.txt"
            target.write_text("data", encoding="utf-8")
            link = Path(tmp_dir) / "link.txt"
            link.symlink_to(target)

            with self.assertRaises(RuntimeError):
                validate_sensitive_file(
                    link,
                    label="test file",
                    allow_missing=False,
                )

    def test_validate_sensitive_file_rejects_broken_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_target = Path(tmp_dir) / "missing.txt"
            link = Path(tmp_dir) / "broken-link.txt"
            link.symlink_to(missing_target)

            with self.assertRaises(RuntimeError):
                validate_sensitive_file(link, label="test file", allow_missing=True)

    def test_tighten_file_permissions_removes_group_other_access(self) -> None:
        if os.name != "posix":
            self.skipTest("Permission mode checks are POSIX-specific.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "secret.txt"
            path.write_text("secret", encoding="utf-8")
            path.chmod(0o644)

            tighten_file_permissions(path, label="secret file")

            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode & 0o077, 0)


if __name__ == "__main__":
    unittest.main()
