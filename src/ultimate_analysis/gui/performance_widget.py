#!/usr/bin/env python3
"""
Performance monitoring widget for displaying live runtime metrics.

Changes:
- Replaced flat table with hierarchical, expandable tree grouped by category.
- Removed CPU/GPU/memory graphs and system metrics.
- Kept backward-compatible add_processing_measurement(process_name, duration_ms).
"""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PerformanceMetrics:
    """Container for processing time series (hierarchical support via keys)."""

    def __init__(self, history_size: int = 120):
        self.history_size = history_size
        # Keyed by (category, subcategory) where subcategory can be None for top-level timings
        self.series: Dict[Tuple[str, Optional[str]], Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    def add(self, category: str, duration_ms: float, subcategory: Optional[str] = None) -> None:
        self.series[(category, subcategory)].append(duration_ms)

    def get_stats(self, category: str, subcategory: Optional[str] = None) -> Dict[str, float]:
        seq = self.series.get((category, subcategory))
        if not seq:
            return {"last": 0.0, "avg": 0.0, "max": 0.0}
        return {
            "last": seq[-1],
            "avg": (sum(seq) / len(seq)) if seq else 0.0,
            "max": max(seq) if seq else 0.0,
        }

    def children_of(self, category: str) -> List[str]:
        subs = sorted({sub for (cat, sub) in self.series.keys() if cat == category and sub})
        return [s for s in subs if s is not None]


class PerformanceWidget(QWidget):
    """Hierarchical runtime table with expandable categories."""

    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.metrics = PerformanceMetrics()
        # Known categories in display order
        self.categories_order: List[str] = [
            "Frame I/O",
            "Inference",
            "Tracking",
            "Player Identification",
            "Field Segmentation",
            "Cache",
            "Homography",
            "Visualization",
            "UI Display",
            "Total Runtime",
        ]
        # Known subcategories by category to always display rows (even if no data for current frame)
        self.known_children: Dict[str, List[str]] = {
            "Player Identification": [
                "Preprocessing",
                "Optical Character Recognition",
                "Jersey Number Filtering",
            ],
            "Homography": [
                "Calculation",
                "Display",
            ],
            "Field Segmentation": [
                "Line Extraction",
                "Mask Unification",
            ],
            "Cache": [
                "Lookup",
                "Store",
            ],
        }
        # Track which metrics were updated in the current frame
        self._touched: Set[Tuple[str, Optional[str]]] = set()
        self._init_ui()
        self._init_timer()

    # UI setup
    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        group = QGroupBox("")
        vbox = QVBoxLayout()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Process", "Last (ms)", "Avg (ms)", "Max (ms)"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                background-color: #2a2a2a;
                alternate-background-color: #3a3a3a;
                color: #ffffff;
                gridline-color: #555555;
                font-size: 12px;
                border: 1px solid #555555;
            }
            QHeaderView::section {
                background-color: #444444;
                color: #ffffff;
                padding: 2px 4px;
                font-size: 12px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """
        )

        # Build initial top-level categories (collapsed by default)
        self.category_items: Dict[str, QTreeWidgetItem] = {}
        for cat in self.categories_order:
            item = QTreeWidgetItem([cat, "0.0", "0.0", "0.0"])
            # Highlight Total Runtime
            if cat == "Total Runtime":
                item.setBackground(0, QColor(60, 80, 100))
                item.setBackground(1, QColor(60, 80, 100))
                item.setBackground(2, QColor(60, 80, 100))
                item.setBackground(3, QColor(60, 80, 100))
            self.tree.addTopLevelItem(item)
            item.setExpanded(False)
            self.category_items[cat] = item

        # Column widths
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 80)
        self.tree.setColumnWidth(2, 80)
        self.tree.setColumnWidth(3, 80)

        # Ensure the tree is tall enough to avoid scroll. Allow extra space for expanded rows.
        # Approx: Header ~28px + top-level (~10 * 22px) + subrows (~10 * 18px) + padding
        self.tree.setFixedHeight(28 + 10 * 22 + 10 * 18 + 24)

        vbox.addWidget(self.tree)
        group.setLayout(vbox)
        layout.addWidget(group)
        self.setLayout(layout)

    def _init_timer(self) -> None:
        # Refresh UI every 200ms
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh_tree)
        self.ui_timer.start(200)

    # Public API (backward compatible)
    def begin_frame(self) -> None:
        """Mark the start of a new frame; resets touched flags.

        Call once per frame before adding measurements to ensure 'Last' shows 0 for
        steps that didn't execute in this frame.
        """
        self._touched.clear()

    def add_processing_measurement(self, process_name: str, duration_ms: float) -> None:
        category, sub = self._categorize(process_name)
        self.metrics.add(category, duration_ms, sub)
        self._touched.add((category, sub))

    # Helpers
    def _categorize(self, name: str) -> Tuple[str, Optional[str]]:
        n = name.strip()
        # Frame I/O
        if n == "Frame I/O":
            return "Frame I/O", None

        # Cache operations
        if n.startswith("Cache"):
            sub = None
            if " - " in n:
                sub = n.split(" - ", 1)[1].strip()
            return "Cache", sub
        # Player ID mappings
        if n.startswith("Player ID"):
            sub = None
            if " - " in n:
                sub_raw = n.split(" - ", 1)[1].strip()
                if sub_raw.lower() == "easyocr":
                    sub = "Optical Character Recognition"
                elif "preprocess" in sub_raw.lower():
                    sub = "Preprocessing"
                elif "filter" in sub_raw.lower():
                    sub = "Jersey Number Filtering"
                else:
                    sub = sub_raw
            return "Player Identification", sub

        # Homography group
        if n.startswith("Homography"):
            if "Calculation" in n:
                return "Homography", "Calculation"
            if "Display" in n:
                return "Homography", "Display"
            return "Homography", None

        # Field Segmentation subcategories
        if n == "Line Extraction":
            return "Field Segmentation", "Line Extraction"
        if n == "Mask Unification":
            return "Field Segmentation", "Mask Unification"

        # All others are their own categories
        return n, None

    def _aggregate_category(self, category: str) -> Dict[str, float]:
        """
        Aggregate stats for a category.
        For Field Segmentation, always sum only its subcategories (never add its own timing),
        to prevent double counting and ensure correct aggregation.
        For other categories, prefer own series if present, else sum of children.
        """
        if category == "Field Segmentation":
            # Always sum only subcategories, ignore parent timing
            total = {"last": 0.0, "avg": 0.0, "max": 0.0}
            for sub in self.metrics.children_of(category):
                s = self.metrics.get_stats(category, sub)
                total["last"] += s["last"]
                total["avg"] += s["avg"]
                total["max"] += s["max"]
            return total

        own = self.metrics.get_stats(category, None)
        has_own = (category, None) in self.metrics.series and len(
            self.metrics.series[(category, None)]
        ) > 0
        if has_own:
            return own

        # Sum across children (default behavior)
        total = {"last": 0.0, "avg": 0.0, "max": 0.0}
        for sub in self.metrics.children_of(category):
            s = self.metrics.get_stats(category, sub)
            total["last"] += s["last"]
            total["avg"] += s["avg"]
            total["max"] += s["max"]
        return total

    def _ensure_child(self, category: str, child_label: str) -> QTreeWidgetItem:
        parent = self.category_items[category]
        # Search existing child
        for i in range(parent.childCount()):
            ch = parent.child(i)
            if ch.text(0) == child_label:
                return ch
        # Create new child
        child = QTreeWidgetItem([child_label, "0.0", "0.0", "0.0"])
        # Subtle category-based tinting
        if category == "Player Identification":
            for c in range(4):
                child.setBackground(c, QColor(50, 70, 50))
        elif category == "Homography":
            for c in range(4):
                child.setBackground(c, QColor(70, 50, 70))
        parent.addChild(child)
        return child

    # UI refresh
    def _refresh_tree(self) -> None:
        # Update top-level categories
        for cat in self.categories_order:
            item = self.category_items[cat]
            stats = self._aggregate_category(cat)

            # Determine 'last' for display: 0 if not touched in this frame
            # For categories with own series, check (cat, None). Otherwise, check any child touched
            display_last = stats["last"]
            has_own_series = (cat, None) in self.metrics.series and len(
                self.metrics.series[(cat, None)]
            ) > 0
            if has_own_series:
                if (cat, None) not in self._touched:
                    display_last = 0.0
            else:
                # Sum-of-children category: if none of the children were touched this frame, show 0
                any_child_touched = any(
                    (cat, sub) in self._touched
                    for sub in self.metrics.children_of(cat) or self.known_children.get(cat, [])
                )
                if not any_child_touched:
                    display_last = 0.0

            item.setText(1, f"{display_last:.1f}")
            item.setText(2, f"{stats['avg']:.1f}")
            item.setText(3, f"{stats['max']:.1f}")

            # Update children for categories that have them
            if cat in ("Player Identification", "Homography", "Field Segmentation", "Cache"):
                # Ensure all known children rows exist
                subs = set(self.metrics.children_of(cat)) | set(self.known_children.get(cat, []))
                for sub in sorted(subs):
                    child = self._ensure_child(cat, sub)
                    s = self.metrics.get_stats(cat, sub)
                    # Show 0 for 'last' if not touched this frame
                    child_last = s["last"] if (cat, sub) in self._touched else 0.0
                    child.setText(1, f"{child_last:.1f}")
                    child.setText(2, f"{s['avg']:.1f}")
                    child.setText(3, f"{s['max']:.1f}")
