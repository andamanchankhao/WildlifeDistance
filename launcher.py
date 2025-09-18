########################################################
### Dev : Andaman Chankhao andmanchankhao@gmail.com ####
### Launcher created by Gemini #########################
########################################################

import sys
import os
from datetime import date
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt

# --- MODIFIED: Import the main window classes from your other scripts ---
from annotate_train_DPT import AnnotationTool 
from calculator_DPT import DistanceCalculator

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class AppLauncher(QWidget):
    """
    The main launcher window to open the Annotation and Calculator tools.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wildlife Distance')
        self.setGeometry(300, 300, 300, 320)
        
        icon_path = resource_path('icon.icns')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # --- MODIFIED: Keep track of the tool windows ---
        self.annotation_window = None
        self.calculator_window = None

        self.init_ui()

    def init_ui(self):
        """Initializes the user interface of the launcher."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setAlignment(Qt.AlignCenter)

        # --- Logo ---
        logo_label = QLabel(self)
        logo_path = resource_path('icon.icns')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        # --- Title ---
        title = QLabel("Wildlife Distance Tool", self)
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(20)

        # --- Buttons ---
        annotate_btn = QPushButton("Annotate & Train Tool", self)
        annotate_btn.setFont(QFont('Arial', 12))
        annotate_btn.setMinimumHeight(40)
        annotate_btn.clicked.connect(self.open_annotation_tool)
        layout.addWidget(annotate_btn)

        calculator_btn = QPushButton("Distance Calculator", self)
        calculator_btn.setFont(QFont('Arial', 12))
        calculator_btn.setMinimumHeight(40)
        calculator_btn.clicked.connect(self.open_calculator_tool)
        layout.addWidget(calculator_btn)
        
        layout.addStretch()

        # --- Footer ---
        year = date.today().year
        footer = QLabel(f"© {year} Andaman Chankhao. All Rights Reserved.", self)
        footer.setFont(QFont('Arial', 8))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

    def open_annotation_tool(self):
        """--- MODIFIED: Create and show the annotation tool window ---"""
        # Create an instance of the window if it doesn't exist yet
        if not self.annotation_window:
            self.annotation_window = AnnotationTool()
        self.annotation_window.show()
        self.annotation_window.activateWindow() # Bring to front

    def open_calculator_tool(self):
        """--- MODIFIED: Create and show the calculator tool window ---"""
        # Create an instance of the window if it doesn't exist yet
        if not self.calculator_window:
            self.calculator_window = DistanceCalculator()
        self.calculator_window.show()
        self.calculator_window.activateWindow() # Bring to front

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = AppLauncher()
    launcher.show()
    sys.exit(app.exec_())
