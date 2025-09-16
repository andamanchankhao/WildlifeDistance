########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
### Launcher created by Gemini #########################
########################################################

import sys
import subprocess
import os
from datetime import date
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class AppLauncher(QWidget):
    """
    A simple launcher window to open the Annotation and Calculator tools.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wildlife Distance')
        self.setGeometry(300, 300, 300, 320)
        
        # Use the helper function to find the icon
        icon_path = resource_path('icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.init_ui()

    def init_ui(self):
        """Initializes the user interface of the launcher."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Add Image Display ---
        image_label = QLabel(self)
        # Use the helper function to find the image
        image_path = resource_path('icon.png')
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(360, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
        else:
            image_label.setText("Image not found\n(Place 'icon.png' in the same folder)")
            image_label.setFont(QFont("Arial", 10))
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumHeight(150)

        layout.addWidget(image_label)
        layout.addSpacing(15)

        # Button to launch the annotation and training tool
        self.annotate_button = QPushButton("📍Annotate Images & Train Model")
        self.annotate_button.setFont(QFont("Arial", 13))
        self.annotate_button.setToolTip("Open a tool to draw boxes on images, assign distances, and train a new model.")
        self.annotate_button.clicked.connect(self.open_annotation_tool)
        layout.addWidget(self.annotate_button)

        # Button to launch the distance calculation tool
        self.calculate_button = QPushButton("🎯Calculate Distances from Model")
        self.calculate_button.setFont(QFont("Arial", 13))
        self.calculate_button.setToolTip("Open a tool to load a pre-trained model and calculate distances by clicking on images.")
        self.calculate_button.clicked.connect(self.open_calculator_tool)
        layout.addWidget(self.calculate_button)

        layout.addStretch(1)

        # --- Add Footnote ---
        version = "v1.1"
        today = date.today().strftime("%B %d, %Y")
        footnote_text = f"{version} - {today}"
        
        footnote_label = QLabel(footnote_text)
        footnote_label.setFont(QFont("Arial", 8))
        footnote_label.setStyleSheet("color: grey;")
        footnote_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(footnote_label)


    def run_script(self, script_name, tool_name):
        """
        Launches a Python script using the correct path, whether bundled or not.
        """
        loading_dialog = QMessageBox(self)
        loading_dialog.setWindowTitle("Loading")
        loading_dialog.setText(f"Opening the {tool_name}...\nThis may take a moment.")
        loading_dialog.setStandardButtons(QMessageBox.NoButton)
        loading_dialog.setModal(True)

        def restore_ui():
            loading_dialog.accept()
            QApplication.restoreOverrideCursor()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            loading_dialog.show()
            QApplication.processEvents()

            # Use the helper function to get the correct path to the script
            script_path = resource_path(script_name)
            
            print(f"Launching {script_path}...")
            # Run the script from its correct location
            subprocess.Popen([sys.executable, script_path])

            QTimer.singleShot(3000, restore_ui)

        except FileNotFoundError:
            restore_ui()
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setText(f"Error: Script file not found.")
            error_msg.setInformativeText(f"Ensure '{script_name}' is correctly bundled in the .spec file.")
            error_msg.setWindowTitle("File Not Found Error")
            error_msg.exec_()
        except Exception as e:
            restore_ui()
            print(f"An error occurred while trying to launch {script_name}: {e}")

    def open_annotation_tool(self):
        """Handler to open the annotation tool script."""
        self.run_script('annotate-train-DPT.py', "Annotation Tool")

    def open_calculator_tool(self):
        """Handler to open the calculator tool script."""
        self.run_script('calculator-DPT.py', "Calculator Tool")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = AppLauncher()
    launcher.show()
    sys.exit(app.exec_())

