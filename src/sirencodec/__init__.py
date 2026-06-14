"""Shared paths for the historical SirenCodec package."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["REPO_ROOT"]
