# Init.py — FreeCAD runs this file on EVERY startup, including headless
# `freecadcmd` (no GUI). So we do NOTHING heavy here: no Qt imports, no
# network, no model loading. Anything GUI-related belongs in InitGui.py.
#
# Why keep the file at all? FreeCAD looks for it to recognize this folder
# as an addon. An empty-ish Init.py is normal and correct.
