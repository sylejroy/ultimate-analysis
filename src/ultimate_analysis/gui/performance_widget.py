#!/usr/bin/env python3
"""
Performance monitoring widget for displaying live runtime metrics.
"""

import time
import psutil
from collections import deque, defaultdict
from typing import Dict, List, Optional, Deque

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFont

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class PerformanceMetrics:
    """Container for performance metrics data."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.processing_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.cpu_usage: Deque[float] = deque(maxlen=self.history_size)
        self.gpu_usage: Deque[float] = deque(maxlen=self.history_size)
        self.gpu_memory: Deque[float] = deque(maxlen=self.history_size)
        self.memory_usage: Deque[float] = deque(maxlen=self.history_size)
        self.timestamps: Deque[float] = deque(maxlen=self.history_size)
    
    def add_processing_time(self, process_name: str, duration_ms: float):
        """Add a processing time measurement."""
        self.processing_times[process_name].append(duration_ms)
    
    def add_system_metrics(self, cpu_percent: float, memory_mb: float, gpu_percent: float = 0.0, gpu_memory_mb: float = 0.0):
        """Add system resource usage metrics."""
        current_time = time.time()
        self.timestamps.append(current_time)
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_mb)
        self.gpu_usage.append(gpu_percent)
        self.gpu_memory.append(gpu_memory_mb)
    
    def get_stats(self, process_name: str) -> Dict[str, float]:
        """Get statistics for a specific process."""
        times = self.processing_times[process_name]
        if not times:
            return {"last": 0.0, "avg": 0.0, "max": 0.0}
        
        return {
            "last": times[-1] if times else 0.0,
            "avg": sum(times) / len(times) if times else 0.0,
            "max": max(times) if times else 0.0
        }


class MiniGraphWidget(QWidget):
    """Small graph widget for displaying time series data."""
    
    def __init__(self, title: str, color: QColor = QColor(100, 150, 255), max_value: float = 100.0):
        super().__init__()
        self.title = title
        self.color = color
        self.max_value = max_value
        self.data: Deque[float] = deque(maxlen=60)  # 60 data points
        self.timestamps: Deque[float] = deque(maxlen=60)
        self.setFixedSize(150, 80)
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
    
    def add_data_point(self, value: float):
        """Add a new data point."""
        self.data.append(value)
        self.timestamps.append(time.time())
        self.update()
    
    def paintEvent(self, event):
        """Draw the mini graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(42, 42, 42))
        
        # Draw title
        painter.setPen(QPen(QColor(200, 200, 200)))
        font = QFont("Arial", 8)
        painter.setFont(font)
        painter.drawText(5, 15, self.title)
        
        # Draw current value with appropriate units
        current_value = self.data[-1] if self.data else 0.0
        if "RAM" in self.title:
            painter.drawText(5, self.height() - 5, f"{current_value:.1f}GB")
        elif "CPU" in self.title or "GPU" in self.title:
            painter.drawText(5, self.height() - 5, f"{current_value:.0f}%")
        else:
            painter.drawText(5, self.height() - 5, f"{current_value:.1f}")
        
        # Draw graph
        if len(self.data) > 1:
            painter.setPen(QPen(self.color, 1.5))
            
            width = self.width() - 10
            height = self.height() - 30
            y_offset = 20
            
            points = []
            for i, value in enumerate(self.data):
                x = 5 + (i * width / max(1, len(self.data) - 1))
                y = y_offset + height - (value / self.max_value * height)
                y = max(y_offset, min(y_offset + height, y))
                points.append((int(x), int(y)))
            
            # Draw lines between points
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])


class PerformanceWidget(QWidget):
    """Widget for displaying live performance metrics."""
    
    metrics_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.metrics = PerformanceMetrics()
        self.process = psutil.Process()  # Current process
        self.init_ui()
        self.init_timers()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Performance Metrics Group (no title)
        perf_group = QGroupBox("")
        perf_layout = QVBoxLayout()
        
        # Processing times table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Process", "Last (ms)", "Avg (ms)", "Max (ms)"])
        self.table.setAlternatingRowColors(True)
        
        # Disable scrollbars and make table size to content
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                alternate-background-color: #3a3a3a;
                color: #ffffff;
                gridline-color: #555555;
                font-size: 12px;
                border: 1px solid #555555;
            }
            QTableWidget::item {
                padding: 2px 4px;
                border: none;
            }
            QHeaderView::section {
                background-color: #444444;
                color: #ffffff;
                padding: 2px 4px;
                font-size: 12px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """)
        
        # Initialize table rows
        self._init_table_rows()
        
        # Set column widths for better space usage
        self.table.setColumnWidth(0, 120)  # Process name (wider)
        self.table.setColumnWidth(1, 70)   # Last (ms)
        self.table.setColumnWidth(2, 70)   # Avg (ms)
        self.table.setColumnWidth(3, 70)   # Max (ms)
        
        # Make table use available space and remove empty column
        self.table.horizontalHeader().setStretchLastSection(True)  # Stretch last column to fill
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setShowGrid(True)  # Keep grid lines
        
        # Set a reasonable maximum width for the table
        self.table.setMaximumWidth(350)  # Limit table width
        
        perf_layout.addWidget(self.table)
        
        # System resource graphs
        graphs_layout = QHBoxLayout()
        
        # CPU Usage Graph (normalized per-core)
        self.cpu_graph = MiniGraphWidget("CPU/Core", QColor(100, 150, 255), 100.0)
        self.cpu_graph.setToolTip("CPU usage per core (normalized)\nShows app CPU usage divided by number of cores")
        graphs_layout.addWidget(self.cpu_graph)
        
        # GPU Usage Graph (if available)
        if GPU_AVAILABLE:
            self.gpu_graph = MiniGraphWidget("GPU %", QColor(255, 150, 100), 100.0)
            self.gpu_graph.setToolTip("GPU utilization percentage")
            graphs_layout.addWidget(self.gpu_graph)
        else:
            self.gpu_graph = None
        
        # Memory Usage Graph (convert MB to percentage of system memory)
        system_memory_gb = psutil.virtual_memory().total / (1024**3)  # Total system memory in GB
        self.memory_graph = MiniGraphWidget("App RAM", QColor(150, 255, 100), system_memory_gb)
        self.memory_graph.setToolTip(f"App memory usage in GB\nSystem has {system_memory_gb:.1f}GB total RAM")
        graphs_layout.addWidget(self.memory_graph)
        
        perf_layout.addLayout(graphs_layout)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        self.setLayout(layout)
    
    def _init_table_rows(self):
        """Initialize the table with processing function rows."""
        processes = [
            "Inference", 
            "Tracking", 
            "Player ID - Preprocessing",
            "Player ID - EasyOCR", 
            "Field Segmentation",
            "Line Extraction",
            "Homography Calculation",
            "Homography Display", 
            "Visualization", 
            "Total Runtime"
        ]
        self.table.setRowCount(len(processes))
        
        for i, process in enumerate(processes):
            self.table.setItem(i, 0, QTableWidgetItem(process))
            self.table.setItem(i, 1, QTableWidgetItem("0.0"))
            self.table.setItem(i, 2, QTableWidgetItem("0.0"))
            self.table.setItem(i, 3, QTableWidgetItem("0.0"))
            
            # Style the Total Runtime row differently for emphasis
            if process == "Total Runtime":
                for col in range(4):
                    item = self.table.item(i, col)
                    if item:
                        item.setBackground(QColor(60, 80, 100))  # Darker blue background
            
            # Style the Player ID rows with a subtle background
            elif "Player ID" in process:
                for col in range(4):
                    item = self.table.item(i, col)
                    if item:
                        item.setBackground(QColor(50, 70, 50))  # Subtle green tint
                        
            # Style the Homography rows with a subtle background
            elif "Homography" in process or "Line Extraction" in process:
                for col in range(4):
                    item = self.table.item(i, col)
                    if item:
                        item.setBackground(QColor(70, 50, 70))  # Subtle purple tint
        
        # Auto-resize table height to fit all rows
        self._resize_table_to_content()
    
    def _resize_table_to_content(self):
        """Resize table to fit all content without scrollbars."""
        # Calculate height needed for all rows + header
        header_height = self.table.horizontalHeader().height()
        row_height = self.table.rowHeight(0) if self.table.rowCount() > 0 else 25
        total_height = header_height + (row_height * self.table.rowCount()) + 2  # +2 for borders
        
        # Set the exact height needed
        self.table.setFixedHeight(total_height)
    
    def init_timers(self):
        """Initialize update timers."""
        # System metrics timer (every 500ms)
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self._update_system_metrics)
        self.system_timer.start(500)
        
        # UI update timer (every 200ms)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(200)
    
    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage (for this process) - normalize to per-core percentage
            cpu_percent_raw = self.process.cpu_percent()
            cpu_cores = psutil.cpu_count()
            cpu_percent = min(100.0, cpu_percent_raw / cpu_cores * 100) if cpu_cores > 0 else cpu_percent_raw
            
            # Memory usage (convert to GB for the graph)
            memory_info = self.process.memory_info()
            memory_gb = memory_info.rss / (1024**3)  # Convert bytes to GB
            
            # Get system memory for context
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_info.rss / system_memory.total) * 100
            
            # GPU metrics
            gpu_percent = 0.0
            gpu_memory = 0.0
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_percent = gpu.load * 100
                        gpu_memory = gpu.memoryUsed
                except:
                    pass
            
            # Add to metrics (store memory as percentage for consistency)
            self.metrics.add_system_metrics(cpu_percent, memory_percent, gpu_percent, gpu_memory)
            
            # Update graphs
            self.cpu_graph.add_data_point(cpu_percent)
            self.memory_graph.add_data_point(memory_gb)  # Display as GB in graph
            if self.gpu_graph:
                self.gpu_graph.add_data_point(gpu_percent)
                
        except Exception as e:
            print(f"[PERFORMANCE] Error updating system metrics: {e}")
    
    def _update_ui(self):
        """Update the UI with latest metrics."""
        processes = [
            "Inference", 
            "Tracking", 
            "Player ID - Preprocessing",
            "Player ID - EasyOCR", 
            "Field Segmentation",
            "Line Extraction",
            "Homography Calculation",
            "Homography Display", 
            "Visualization", 
            "Total Runtime"
        ]
        
        for i, process in enumerate(processes):
            stats = self.metrics.get_stats(process)
            self.table.item(i, 1).setText(f"{stats['last']:.1f}")
            self.table.item(i, 2).setText(f"{stats['avg']:.1f}")
            self.table.item(i, 3).setText(f"{stats['max']:.1f}")
    
    def add_processing_measurement(self, process_name: str, duration_ms: float):
        """Add a processing time measurement."""
        self.metrics.add_processing_time(process_name, duration_ms)
