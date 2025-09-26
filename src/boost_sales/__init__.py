# SPDX-License-Identifier: MIT
"""
Lightweight package init: don't import heavy submodules here.
This avoids side-effect imports when scripts only need config or utils.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
