import unittest

from bot.number_utils import safe_float, safe_int


class NumberUtilsTests(unittest.TestCase):
    def test_safe_float_parses_numeric_inputs(self) -> None:
        self.assertEqual(safe_float("1.25"), 1.25)
        self.assertEqual(safe_float(2), 2.0)

    def test_safe_float_falls_back_on_invalid_values(self) -> None:
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float("abc", default=9.5), 9.5)

    def test_safe_int_parses_numeric_inputs(self) -> None:
        self.assertEqual(safe_int("7"), 7)
        self.assertEqual(safe_int("7.9"), 7)

    def test_safe_int_falls_back_on_invalid_values(self) -> None:
        self.assertEqual(safe_int(None), 0)
        self.assertEqual(safe_int("abc", default=3), 3)


if __name__ == "__main__":
    unittest.main()
