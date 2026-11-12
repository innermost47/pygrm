# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""pygrm - the Global Relational Model engine.

    from pygrm import GlobalRelationalNetwork
    GlobalRelationalNetwork(nodes, relations, units="planck")   # si | atomic | planck | ticks
    discover.find_relations(rows)    # the relations a network can find by itself in what it has seen
    discover.hidden_node(residuals)  # is a node missing?
"""
from .grm import GlobalRelationalNetwork
from . import units
from .units import express, convert_network, MODES
from . import discover
from . import derive

__all__ = ["GlobalRelationalNetwork", "units", "express", "convert_network", "MODES", "discover", "derive"]
__version__ = "0.5.0"
