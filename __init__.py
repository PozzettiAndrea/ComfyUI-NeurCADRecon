# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI-NeurCADRecon - Neural Implicit CAD Reconstruction

Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra

Paper: "NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces" (SIGGRAPH 2024)
"""

import sys

if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
