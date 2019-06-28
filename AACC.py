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
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
program = GUI.MainUI()
sys.exit(app.exec_())

