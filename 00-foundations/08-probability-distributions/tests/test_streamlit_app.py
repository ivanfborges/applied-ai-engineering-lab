"""Smoke test for the Day 8 Streamlit dashboard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))


class StreamlitDashboardTests(unittest.TestCase):
    """Ensure the default dashboard view renders without an exception."""

    def test_default_view_loads(self) -> None:
        app = AppTest.from_file(
            str(TOPIC_DIRECTORY / "interactive_dashboard.py"),
            default_timeout=30,
        )

        app.run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.title), 1)


if __name__ == "__main__":
    unittest.main()
