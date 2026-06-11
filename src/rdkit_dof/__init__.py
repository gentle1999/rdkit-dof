"""
Author: TMJ
Date: 2025-12-01 12:37:38
LastEditors: TMJ
LastEditTime: 2026-06-11 21:55:35
Description: 请填写简介
"""

import importlib.metadata

from .config import DofDrawSettings, dofconfig
from .core import MolsToGridDofImage, MolToDofImage

try:
    __version__ = importlib.metadata.version("rdkit-dof")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["MolsToGridDofImage", "MolToDofImage", "DofDrawSettings", "dofconfig"]
