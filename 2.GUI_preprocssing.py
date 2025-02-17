import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, 
    QFileDialog, QSlider, QLabel, QTableWidget, QTableWidgetItem, QStackedWidget, QGraphicsDropShadowEffect, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
from scipy.signal import savgol_filter

# ---------------------------------------------------
# Styling Class: StyleManager
# ---------------------------------------------------
class StyleManager:
    @staticmethod
    def style_main_window(widget, gradient_start="#f0f0f0", gradient_end="#ffffff", border_radius=0):
        style = f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {gradient_start}, stop:1 {gradient_end});
            border-radius: {border_radius}px;
        """
        widget.setStyleSheet(style)

    @staticmethod
    def style_sidebar(widget, gradient_start="#2c3e50", gradient_end="#34495e", border_radius=10):
        style = f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {gradient_start}, stop:1 {gradient_end});
            border-radius: {border_radius}px;
        """
        widget.setStyleSheet(style)

    @staticmethod
    def style_sidebar_button(button, bg_color="#34495e", hover_color="#3c5d7c", text_color="white", border_radius=5, shadow=True):
        style = f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 10px;
                border-radius: {border_radius}px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """
        button.setStyleSheet(style)
        if shadow:
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(15)
            shadow_effect.setXOffset(0)
            shadow_effect.setYOffset(4)
            shadow_effect.setColor(QColor(0, 0, 0, 160))
            button.setGraphicsEffect(shadow_effect)

# ---------------------------------------------------
# Main GUI Class: SpectralPreprocessingTool
# ---------------------------------------------------
class SpectralPreprocessingTool(QMainWindow):
    def __init__(self):
        super().__init__()

        # Dynamically set window size to 100% of the user's screen resolution
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        self.setWindowTitle("Spectral Preprocessing Tool")

        # Data storage and loaded file path
        self.data = None
        self.loaded_file_path = None

        # Create central widget with a horizontal layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Create sidebar (15% width) and main content (85% width)
        self.create_sidebar()
        self.create_main_content()

        # Apply styling via the StyleManager
        self.apply_styles()

    def create_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
        self.sidebar_layout.setSpacing(15)

        # Preprocessing Button (centered at the top)
        self.preproc_button = QPushButton("Preprocessing")
        self.preproc_button.setCheckable(True)
        self.preproc_button.clicked.connect(self.toggle_preprocessing_options)
        self.sidebar_layout.addWidget(self.preproc_button, alignment=Qt.AlignCenter)

        # Expandable container for extra preprocessing options
        self.preproc_options = QWidget()
        self.preproc_options_layout = QVBoxLayout(self.preproc_options)
        self.preproc_options_layout.setSpacing(10)

        # Button: Load Excel File
        self.btn_load_excel = QPushButton("Load Excel File")
        self.btn_load_excel.clicked.connect(self.load_excel_file)
        self.preproc_options_layout.addWidget(self.btn_load_excel)

        # Button: Plot Data
        self.btn_plot_data = QPushButton("Plot Data")
        self.btn_plot_data.clicked.connect(self.plot_data)
        self.preproc_options_layout.addWidget(self.btn_plot_data)

        # Button: Savitzky-Golay Filter
        self.btn_savgol_filter = QPushButton("Savitzky-Golay Filter")
        self.btn_savgol_filter.clicked.connect(self.activate_filter_mode)
        self.preproc_options_layout.addWidget(self.btn_savgol_filter)

        # Container for filter controls (sliders + dynamic finish button)
        self.filter_controls = QWidget()
        self.filter_controls_layout = QVBoxLayout(self.filter_controls)
        self.filter_controls_layout.setSpacing(10)

        # Slider: Savitzky-Golay Window Size
        self.lbl_window = QLabel("Savitzky-Golay Window Size")
        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setMinimum(3)
        self.slider_window.setMaximum(21)
        self.slider_window.setSingleStep(2)
        self.slider_window.setValue(7)
        self.slider_window.setTickPosition(QSlider.TicksBelow)
        self.slider_window.setTickInterval(2)
        self.filter_controls_layout.addWidget(self.lbl_window)
        self.filter_controls_layout.addWidget(self.slider_window)

        # Slider: Polynomial Order
        self.lbl_poly = QLabel("Polynomial Order")
        self.slider_poly = QSlider(Qt.Horizontal)
        self.slider_poly.setMinimum(1)
        self.slider_poly.setMaximum(5)
        self.slider_poly.setValue(2)
        self.slider_poly.setTickPosition(QSlider.TicksBelow)
        self.slider_poly.setTickInterval(1)
        self.filter_controls_layout.addWidget(self.lbl_poly)
        self.filter_controls_layout.addWidget(self.slider_poly)

        # Dynamic Button: Finished filtering choices
        self.btn_finish_filtering = QPushButton("Finished filtering choices")
        self.btn_finish_filtering.clicked.connect(self.finished_filtering)
        self.filter_controls_layout.addWidget(self.btn_finish_filtering)

        # Initially, hide the filter controls
        self.filter_controls.setVisible(False)
        self.preproc_options_layout.addWidget(self.filter_controls)

        self.preproc_options_layout.addStretch()
        self.preproc_options.setVisible(False)
        self.sidebar_layout.addWidget(self.preproc_options)
        self.sidebar_layout.addStretch()

        self.main_layout.addWidget(self.sidebar, stretch=15)

    def create_main_content(self):
        self.content_area = QStackedWidget()
        self.main_layout.addWidget(self.content_area, stretch=85)

        # Data table view for Excel contents
        self.table_widget = QTableWidget()
        self.content_area.addWidget(self.table_widget)

        # Plot view using Plotly embedded in QWebEngineView
        self.plot_view = QWebEngineView()
        self.content_area.addWidget(self.plot_view)

    def apply_styles(self):
        # Main window styling
        StyleManager.style_main_window(self.central_widget, gradient_start="#f0f0f0", gradient_end="#ffffff")
        # Sidebar styling
        StyleManager.style_sidebar(self.sidebar, gradient_start="#2c3e50", gradient_end="#34495e")
        # Style sidebar buttons
        StyleManager.style_sidebar_button(self.preproc_button, bg_color="#e74c3c", hover_color="#c0392b", text_color="white", border_radius=15)
        StyleManager.style_sidebar_button(self.btn_load_excel)
        StyleManager.style_sidebar_button(self.btn_plot_data)
        StyleManager.style_sidebar_button(self.btn_savgol_filter)
        StyleManager.style_sidebar_button(self.btn_finish_filtering)

    def toggle_preprocessing_options(self):
        # Toggle the visibility of the entire extra options panel
        is_checked = self.preproc_button.isChecked()
        self.preproc_options.setVisible(is_checked)
        # Also, hide filter controls if not in filtering mode
        if not is_checked:
            self.filter_controls.setVisible(False)

    def load_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            try:
                self.data = pd.read_excel(file_path)
                self.loaded_file_path = file_path
                print("Excel file loaded successfully.")
                self.display_data()  # Auto-display the data
            except Exception as e:
                print(f"Error loading file: {e}")

    def display_data(self):
        if self.data is not None:
            self.table_widget.clear()
            self.table_widget.setRowCount(len(self.data))
            self.table_widget.setColumnCount(len(self.data.columns))
            self.table_widget.setHorizontalHeaderLabels(self.data.columns.tolist())
            for i in range(len(self.data)):
                for j, col in enumerate(self.data.columns):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(self.data.iloc[i][col])))
            self.content_area.setCurrentWidget(self.table_widget)
        else:
            print("No data loaded.")

    def plot_data(self):
        if self.data is not None:
            # For now, we assume the first column is 'wavelength' and plot the next available column.
            # In the future, a popup can let the user choose which columns to plot.
            if self.data.columns[0].lower() == "wavelength" and len(self.data.columns) > 1:
                y_col = self.data.columns[1]
                fig = px.line(self.data, x=self.data.columns[0], y=y_col, title="Spectral Data",
                              labels={self.data.columns[0]: "Wavelength", y_col: y_col})
                html = fig.to_html(include_plotlyjs='cdn')
                self.plot_view.setHtml(html)
                self.content_area.setCurrentWidget(self.plot_view)
            else:
                print("Ensure the first column is named 'wavelength' and at least one data column exists.")
        else:
            print("No data loaded.")

    def activate_filter_mode(self):
        if self.data is not None:
            # Show filter controls (sliders and finish button)
            self.filter_controls.setVisible(True)
            # Optionally, you could switch the main content view to a preview plot.
            # For now, we just enable live filtering previews using the sliders.
            print("Filter mode activated. Adjust the sliders, then click 'Finished filtering choices'.")
        else:
            print("Load an Excel file first.")

    def finished_filtering(self):
        if self.data is None:
            print("No data loaded.")
            return

        # Retrieve slider values for filter parameters
        window = self.slider_window.value()
        poly_order = self.slider_poly.value()

        # Ensure window is odd and greater than poly_order
        if window % 2 == 0:
            window += 1
        if window <= poly_order:
            window = poly_order + 2 if (poly_order + 2) % 2 != 0 else poly_order + 3

        # Apply the filter to all columns except the 'wavelength' column (assumed to be first)
        for col in self.data.columns:
            if col.lower() != "wavelength":
                try:
                    self.data[col] = savgol_filter(self.data[col], window_length=window, polyorder=poly_order)
                except Exception as e:
                    print(f"Error filtering column '{col}': {e}")

        # Save the filtered data to a new Excel file
        if self.loaded_file_path:
            base, ext = os.path.splitext(self.loaded_file_path)
            output_path = base + "_savitzky_filtered" + ext
            try:
                self.data.to_excel(output_path, index=False)
                print(f"Filtered data saved to {output_path}")
                QMessageBox.information(self, "Filtering Complete", f"Filtered data saved to:\n{output_path}")
            except Exception as e:
                print(f"Error saving filtered file: {e}")
        else:
            print("Original file path not found.")

        # Optionally, update the display with the filtered data
        self.display_data()
        # Hide the filter controls after finishing
        self.filter_controls.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectralPreprocessingTool()
    window.show()
    sys.exit(app.exec_())
