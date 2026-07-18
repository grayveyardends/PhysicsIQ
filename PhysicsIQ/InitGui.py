# Copyright (c) 2026 Gabrial Alex. MIT license, see LICENSE.
# InitGui.py — FreeCAD runs this file on GUI startup (NOT in freecadcmd).
#
# NOTE on how FreeCAD loads addons:
#   - Every folder inside <appdata>/Mod/ is put on sys.path automatically.
#     That is why `import physicsiq` below works: the package lives right next
#     to this file (PhysicsIQ/physicsiq/).
#   - This file is exec()'d by FreeCAD, not imported. In some FreeCAD builds
#     `__file__` is NOT defined here, so never rely on it in this file.
#   - Keep this file tiny. Real code lives in the physicsiq package so it can
#     be reloaded/tested without restarting FreeCAD.

import FreeCAD
import FreeCADGui


class PhysicsIQWorkbench(FreeCADGui.Workbench):
    """The workbench entry FreeCAD shows in the workbench dropdown."""

    MenuText = "PhysicsIQ"
    ToolTip = "PhysicsIQ"

    def __init__(self):
        # Icon path: resolve through the package (which DOES have __file__).
        try:
            import physicsiq
            self.__class__.Icon = physicsiq.icon_path()
        except Exception:
            pass  # no icon is ugly but not fatal

    def Initialize(self):
        # Called ONCE, the first time the user activates the workbench.
        # We import lazily here so FreeCAD startup stays fast even when
        # the user never opens PhysicsIQ.
        from physicsiq import commands
        self.appendToolbar("PhysicsIQ", commands.TOOLBAR_COMMANDS)
        self.appendMenu("PhysicsIQ", commands.MENU_COMMANDS)

    def Activated(self):
        # Called every time the user switches TO this workbench.
        # Open (or re-show) the chat dock panel.
        from physicsiq import panel_manager
        panel_manager.show_chat_panel()

    def Deactivated(self):
        # Called when the user switches AWAY. We keep the panel alive —
        # people like chatting while using other workbenches.
        pass

    def GetClassName(self):
        # Tells FreeCAD this is a pure-Python workbench.
        return "Gui::PythonWorkbench"


FreeCADGui.addWorkbench(PhysicsIQWorkbench())
