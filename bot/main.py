"""Console-script entry point that reuses the existing root main module."""

from main import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
