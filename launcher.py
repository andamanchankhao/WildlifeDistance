########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
### Launcher created by Gemini #########################
########################################################

import sys
import subprocess
import os
from datetime import date # Import the date object
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

class AppLauncher(QWidget):
    """
    A simple launcher window to open the Annotation and Calculator tools.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wildlife Distance')
        # Adjusted geometry for the new footnote
        self.setGeometry(300, 300, 300, 320)
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(script_dir, 'icon.png')
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
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # You should create an image named 'wildlife_icon.png' and place it in the same folder.
        image_path = os.path.join(script_dir, 'icon.png')
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale pixmap to fit nicely in the window
            image_label.setPixmap(pixmap.scaled(360, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
        else:
            # Placeholder text if the image is not found
            image_label.setText("Image not found\n(Place 'icon.png' in the same folder)")
            image_label.setFont(QFont("Arial", 10))
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumHeight(150) # Give it some space

        layout.addWidget(image_label)
        layout.addSpacing(15) # Add some space between the image and the buttons

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

        # Add a stretch to push the footnote to the bottom
        layout.addStretch(1)

        # --- Add Footnote ---
        version = "v1.1"
        today = date.today().strftime("%B %d, %Y") # Format date as Month Day, Year
        footnote_text = f"{version} - {today}"
        
        footnote_label = QLabel(footnote_text)
        footnote_label.setFont(QFont("Arial", 8))
        footnote_label.setStyleSheet("color: grey;")
        footnote_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(footnote_label)


    def run_script(self, script_name, tool_name):
        """
        Launches a Python script, shows a loading dialog, and changes the cursor.
        
        Args:
            script_name (str): The filename of the script to run.
            tool_name (str): The user-friendly name of the tool for the message.
        """
        # Create and configure the loading dialog
        loading_dialog = QMessageBox(self)
        loading_dialog.setWindowTitle("Loading")
        loading_dialog.setText(f"Opening the {tool_name}...\nThis may take a moment.")
        loading_dialog.setStandardButtons(QMessageBox.NoButton) # No buttons on the dialog
        loading_dialog.setModal(True)

        def restore_ui():
            """A helper function to close the dialog and restore the normal cursor."""
            loading_dialog.accept()
            QApplication.restoreOverrideCursor()

        try:
            # Change cursor to a waiting cursor and show the dialog
            QApplication.setOverrideCursor(Qt.WaitCursor)
            loading_dialog.show()
            QApplication.processEvents()  # Ensure the UI updates to show the dialog immediately

            # Use subprocess.Popen to run the script in a new process
            print(f"Launching {script_name}...")
            subprocess.Popen([sys.executable, script_name])

            # Schedule the UI to be restored after a delay, giving the app time to open
            QTimer.singleShot(3000, restore_ui)  # Close dialog and restore cursor after 3 seconds

        except FileNotFoundError:
            restore_ui() # Restore UI immediately on error
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setText(f"Error: Script file not found.")
            error_msg.setInformativeText(f"Ensure '{script_name}' is in the same directory as this launcher.")
            error_msg.setWindowTitle("File Not Found Error")
            error_msg.exec_()
        except Exception as e:
            restore_ui() # Restore UI immediately on error
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

