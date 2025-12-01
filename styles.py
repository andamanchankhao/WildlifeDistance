
from PyQt5.QtGui import QPalette, QColor, QLinearGradient, QBrush
from PyQt5.QtCore import Qt

def apply_theme(app):
    """
    Applies the 'Liquid Glass' theme to the QApplication instance.
    This includes a dark gradient palette and a comprehensive QSS stylesheet.
    """
    app.setStyle("Fusion")

    # --- 1. Dark Gradient Palette (Fallback / Base) ---
    dark_palette = QPalette()
    
    # Base background color (deep dark blue/gray)
    base_color = QColor(20, 25, 35)
    dark_palette.setColor(QPalette.Window, base_color)
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 35, 45))
    dark_palette.setColor(QPalette.AlternateBase, base_color)
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, base_color)
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)

    # --- 2. Liquid Glass Stylesheet ---
    # We use semi-transparent backgrounds (rgba) and blur-like colors.
    # Note: True backdrop-filter blur isn't supported in standard QSS, 
    # so we simulate it with semi-transparent layers.

    glass_style = """
    /* Main Window Background */
    QMainWindow, QDialog {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 #1a1c29, stop:1 #2d3447);
    }

    /* Generic Widget Reset */
    QWidget {
        color: #e0e0e0;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 14px;
    }

    /* --- Glass Panels --- */
    /* Use this class or ID for container widgets to give them the glass look */
    QWidget#GlassPanel {
        background-color: rgba(255, 255, 255, 15); /* Very subtle white tint */
        border: 1px solid rgba(255, 255, 255, 30);
        border-radius: 15px;
    }
    
    /* GroupBoxes as Glass Containers */
    QGroupBox {
        background-color: rgba(255, 255, 255, 10);
        border: 1px solid rgba(255, 255, 255, 20);
        border-radius: 10px;
        margin-top: 20px; /* Leave space for title */
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 10px;
        color: #a0a0a0;
        background-color: transparent;
    }

    /* --- Buttons (Liquid Style) --- */
    QPushButton {
        background-color: rgba(255, 255, 255, 20);
        border: 1px solid rgba(255, 255, 255, 40);
        border-radius: 8px;
        padding: 8px 16px;
        color: white;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 40); /* Brighter on hover */
        border: 1px solid rgba(255, 255, 255, 80);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 10);
        border: 1px solid rgba(255, 255, 255, 20);
    }
    QPushButton:disabled {
        background-color: rgba(255, 255, 255, 5);
        color: #606060;
        border: 1px solid rgba(255, 255, 255, 10);
    }

    /* --- Input Fields & Lists --- */
    QLineEdit, QListWidget, QTableWidget, QTextEdit {
        background-color: rgba(0, 0, 0, 40); /* Darker semi-transparent background */
        border: 1px solid rgba(255, 255, 255, 20);
        border-radius: 6px;
        color: white;
        padding: 4px;
        selection-background-color: rgba(42, 130, 218, 150);
    }
    QListWidget::item:hover, QTableWidget::item:hover {
        background-color: rgba(255, 255, 255, 10);
    }
    QListWidget::item:selected, QTableWidget::item:selected {
        background-color: rgba(42, 130, 218, 200); /* More opaque for better visibility */
        border: 1px solid rgba(42, 130, 218, 255);
        border-radius: 4px;
        color: white;
    }

    /* --- Scrollbars (Slim & Modern) --- */
    QScrollBar:vertical {
        border: none;
        background: rgba(0, 0, 0, 20);
        width: 8px;
        margin: 0px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: rgba(255, 255, 255, 40);
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* --- Header Views (Tables) --- */
    QHeaderView::section {
        background-color: rgba(255, 255, 255, 10);
        color: #e0e0e0;
        padding: 4px;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 20);
        border-right: 1px solid rgba(255, 255, 255, 10);
    }

    /* --- Labels --- */
    QLabel {
        color: #e0e0e0;
        background-color: transparent;
    }
    
    /* Specific Highlights */
    QLabel#TitleLabel {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    
    QLabel#StatusLabel {
        color: #a0a0a0;
        font-style: italic;
    }
    """
    
    app.setStyleSheet(glass_style)
