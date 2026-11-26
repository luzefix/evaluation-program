"""Entry point: ``python -m evalprog`` or the ``evalprog`` command."""

import sys


def main() -> int:
    from evalprog.gui.app import run

    return run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
