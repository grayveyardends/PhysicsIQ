"""commands.py — the buttons/menu entries FreeCAD shows for this workbench.

Each command is a tiny class with GetResources (icon + text) and
Activated (what happens on click). FreeCADGui.addCommand registers it
under a globally-unique string name.
"""

import FreeCADGui

from inferio import icon_path


class ShowChatCommand:
    def GetResources(self):
        return {"Pixmap": icon_path(),
                "MenuText": "Inferio Chat",
                "ToolTip": "Open the AI chat panel"}

    def Activated(self):
        from inferio import panel_manager
        panel_manager.show_chat_panel()

    def IsActive(self):
        return True


class OpenPdfCommand:
    def GetResources(self):
        return {"Pixmap": icon_path(),
                "MenuText": "Open PDF for AI",
                "ToolTip": "View a PDF and send pages to the vision model"}

    def Activated(self):
        from PySide6 import QtWidgets
        from inferio import panel_manager
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            panel_manager.main_window(), "Open PDF", "", "PDF files (*.pdf)")
        if path:
            panel_manager.show_pdf_panel(path)

    def IsActive(self):
        return True


class RunPinnCommand:
    def GetResources(self):
        return {"Pixmap": icon_path(),
                "MenuText": "Analyze stress (PINN)",
                "ToolTip": "Run the break-point elasticity PINN on the "
                           "selected part and overlay the stress heatmap"}

    def Activated(self):
        from inferio import panel_manager
        panel = panel_manager.show_chat_panel()
        panel.controller.request_pinn_analysis()

    def IsActive(self):
        import FreeCAD
        return FreeCAD.ActiveDocument is not None


FreeCADGui.addCommand("Inferio_ShowChat", ShowChatCommand())
FreeCADGui.addCommand("Inferio_OpenPdf", OpenPdfCommand())
FreeCADGui.addCommand("Inferio_RunPinn", RunPinnCommand())

TOOLBAR_COMMANDS = ["Inferio_ShowChat", "Inferio_OpenPdf", "Inferio_RunPinn"]
MENU_COMMANDS = TOOLBAR_COMMANDS
