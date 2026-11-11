# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""pygrm - the Global Relational Model engine.

    from pygrm import GlobalRelationalNetwork
    GlobalRelationalNetwork(nodes, relations, units="planck")   # si | atomic | planck | ticks
"""
from .grm import GlobalRelationalNetwork
from . import units
from .units import express, convert_network, MODES

__all__ = ["GlobalRelationalNetwork", "units", "express", "convert_network", "MODES"]
__version__ = "0.2.0"
