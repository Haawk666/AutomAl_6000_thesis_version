"""Entry module for the GUI. Reads 'config.ini' and launches the PyQt5 application loop."""

#
# Automatic Atomic Column Characterizer (AACC).
# ----------------------------------------
# By Haakon Tvedt.
# ----------------------------------------
# Master project in technical physics at NTNU. Supervised by Prof. Randi Holmestad. Co-supervisors Calin Maroiara and
# Jesper Friis at SINTEF.
#

# Program imports:
import GUI
import GUI_settings
# External imports:
from PyQt5 import QtWidgets
import sys
import os
import configparser

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Check for existence of config file:
    if not os.path.isfile('config.ini'):
        with open('config.ini', 'w') as f:
            f.write(GUI_settings.default_config_string)

    # Import configurations from config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    GUI_settings.theme = config.get('theme', 'theme')
    GUI_settings.tooltips = config.getboolean('tooltips', 'tooltips')

    # Set theme
    if GUI_settings.theme == 'dark':
        app.setPalette(GUI_settings.dark_palette)
    else:
        pass

    # Start app
    program = GUI.MainUI(settings_file=config)
    sys.exit(app.exec_())

