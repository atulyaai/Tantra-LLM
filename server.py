"""Compatibility entry point for the Tantra FastAPI server.

The implementation lives in :mod:`Tantra.server`; keeping this small module
preserves the documented ``python server.py`` command and existing imports.
"""

from Tantra.server import TANTRA_API_KEY, app, start_server

__all__ = ["TANTRA_API_KEY", "app", "start_server"]


if __name__ == "__main__":
    start_server()
