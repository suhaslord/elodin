"""Minimal, self-contained Voyager chapter 1 experiment."""

from pathlib import Path

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"

# Keep the first pass isolated from main.py and run it only on the agreed
# maneuver-free window. Add the basic gravity sim here next.
