#
# Automatic Atomic Column Characterizer (AACC).
# ----------------------------------------
# By Haakon Tvedt.
# ----------------------------------------
# Master project in technical physics at NTNU. Supervised by Prof. Randi Holmestad. Co-supervisors Calin Maroiara and
# Jesper Friis at SINTEF.
# ----------------------------------------
# This file is the entry point of the software.
# ----------------------------------------
#

import GUI
import GUI_settings
from PyQt5 import QtWidgets
import sys

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    if GUI_settings.theme == 'dark':
        app.setPalette(GUI_settings.dark_palette)
    else:
        pass
    program = GUI.MainUI()
    sys.exit(app.exec_())

