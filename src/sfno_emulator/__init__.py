"""SFNO climate emulators built on NVIDIA Makani.

This package provides the command-line front end for training, inference, and
ensemble runs. The modelling, distributed training, and scoring machinery is
supplied by Makani, which is vendored under ``makani/`` and importable as
``makani``; see the repository README for installation.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
