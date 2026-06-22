from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

def apply_theme(app):
    """
    Applies a 'Minimal Red' theme inspired by BirdNET Analyzer.
    Features a clean white/light gray background with Red accents.
    """
    app.setStyle("Fusion")

    # --- 1. Light Palette with Red Accents ---
    palette = QPalette()
    
    # Base Colors
    base_color = QColor(245, 245, 245) # Light Gray Background
    text_color = QColor(30, 30, 30)    # Almost Black Text
    accent_color = QColor(200, 40, 40) # Red Accent
    
    palette.setColor(QPalette.Window, base_color)
    palette.setColor(QPalette.WindowText, text_color)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.AlternateBase, QColor(230, 230, 230))
    
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, text_color)
    palette.setColor(QPalette.Text, text_color)
    
    palette.setColor(QPalette.Button, Qt.white) # White buttons
    palette.setColor(QPalette.ButtonText, text_color)
    palette.setColor(QPalette.Link, accent_color)
    palette.setColor(QPalette.Highlight, accent_color)
    palette.setColor(QPalette.HighlightedText, Qt.white)
    
    app.setPalette(palette)

    # Detect platform font to avoid alias lookup delay and console warning
    import os
    import platform
    from PyQt5.QtGui import QFontDatabase

    system_font = '".AppleSystemUIFont"' if platform.system() == "Darwin" else '"Segoe UI"'
    
    regular_font_path = os.path.join(os.path.dirname(__file__), "fonts", "Inter-Regular.ttf")
    bold_font_path = os.path.join(os.path.dirname(__file__), "fonts", "Inter-Bold.ttf")

    if os.path.exists(regular_font_path):
        font_id = QFontDatabase.addApplicationFont(regular_font_path)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                system_font = f'"{families[0]}"'
                
    if os.path.exists(bold_font_path):
        QFontDatabase.addApplicationFont(bold_font_path)

    # Set the native application font to match layout size calculations
    font_family_name = system_font.strip('"')
    app.setFont(QFont(font_family_name, 10))

    # --- 2. Minimal Stylesheet ---
    style_sheet = """
    /* Main Window */
    QMainWindow, QDialog {
        background-color: #f5f5f5;
    }
    
    QWidget {
        font-family: REPLACE_FONT, "Helvetica Neue", Arial, sans-serif;
        font-size: 14px;
        color: #1e1e1e;
    }

    /* --- Tab Widget (The Star of the Show) --- */
    QTabWidget::pane {
        border: none;
        border-top: 1px solid #e0e0e0; /* Subtle separator line only */
        background: white;
    }
    
    QTabWidget::tab-bar {
        alignment: left;
    }
    
    QTabBar::tab {
        background: transparent;
        color: #555;
        padding: 10px 32px; /* Increased horizontal padding to prevent cutoff */
        border-bottom: 3px solid transparent; 
        font-weight: 500;
        margin-right: 8px;
        min-width: 150px;
    }
    
    QTabBar::tab:hover {
        color: #c82828;
        background: rgba(200, 40, 40, 0.05);
    }
    
    QTabBar::tab:selected {
        color: #c82828; /* Red Text */
        border-bottom: 3px solid #c82828; /* Thick Red Underline */
        font-weight: bold;
    }

    /* --- Buttons --- */
    QPushButton {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 6px 12px;
        color: #333;
    }
    QPushButton:hover {
        background-color: #f9f9f9;
        border: 1px solid #c82828; /* Red border on hover */
        color: #c82828;
    }
    QPushButton:pressed {
        background-color: #c82828;
        color: white;
    }
    QPushButton:checked {
        background-color: #c82828;
        color: white;
        border: 1px solid #c82828;
    }
    QPushButton:disabled {
        background-color: #f0f0f0;
        color: #aaa;
        border: 1px solid #eee;
    }

    /* --- Input Fields & Lists --- */
    QLineEdit, QListWidget, QTableWidget, QTextEdit {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 4px;
    }
    
    /* Highlight selection in lists/tables */
    QListWidget::item:selected, QTableWidget::item:selected {
        background-color: rgba(200, 40, 40, 0.1); /* Very light red */
        color: #c82828; /* Red text */
        border: none;
    }
    QListWidget::item:hover, QTableWidget::item:hover {
        background-color: #fafafa;
    }

    /* --- Headers --- */
    QHeaderView::section {
        background-color: #f9f9f9;
        color: #555;
        padding: 6px;
        border: none;
        border-bottom: 1px solid #ddd;
        border-right: 1px solid #eee;
        font-weight: bold;
    }

    /* --- Custom ID Hooks --- */
    /* GlassPanel now translates to a clean card in minimal theme */
    QWidget#GlassPanel {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
    
    /* Status Label */
    QLabel#StatusLabel {
        color: #666;
        padding: 4px 8px;
        border-top: 1px solid #eee;
        background-color: #fcfcfc;
    }

    /* --- Scrollbars --- */
    QScrollBar:vertical {
        border: none;
        background: #f0f0f0;
        width: 10px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #ccc;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: #aaa;
    }
    """
    
    app.setStyleSheet(style_sheet.replace('REPLACE_FONT', system_font))


class UploadPlaceholder(QFrame):
    clicked = pyqtSignal()
    files_dropped = pyqtSignal(list)

    def __init__(self, text="Drop Image Folder Here\n- or -\nClick to Browse", parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("UploadPlaceholder")
        self.text = text
        self.is_drag_over = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        # Upload Icon
        self.icon_label = QLabel("📤")
        self.icon_label.setFont(QFont("Arial", 44))
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Instruction Text
        self.text_label = QLabel(self.text)
        self.text_label.setFont(QFont("Arial", 13))
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.update_style()

    def update_style(self):
        border_color = "#c82828" if self.is_drag_over else "#d8d8d8"
        bg_color = "rgba(200, 40, 40, 0.04)" if self.is_drag_over else "white"
        self.setStyleSheet(f"""
            QFrame#UploadPlaceholder {{
                border: 2px dashed {border_color};
                border-radius: 8px;
                background-color: {bg_color};
            }}
            QLabel {{
                color: #666666;
                background: transparent;
                border: none;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.is_drag_over = True
            self.update_style()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.is_drag_over = False
        self.update_style()
        event.accept()

    def dropEvent(self, event):
        self.is_drag_over = False
        self.update_style()
        urls = event.mimeData().urls()
        if urls:
            paths = [url.toLocalFile() for url in urls]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

