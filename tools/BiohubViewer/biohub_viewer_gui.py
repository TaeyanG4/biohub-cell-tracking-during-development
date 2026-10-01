from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from PyQt6.QtCore import QDir, QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QFileSystemModel, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "Biohub Cell Tracking Viewer"


@dataclass
class GraphData:
    node_ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        self.node_index = {int(node_id): i for i, node_id in enumerate(self.node_ids.tolist())}
        self.nodes_by_t: dict[int, np.ndarray] = {}
        for tv in np.unique(self.t):
            self.nodes_by_t[int(tv)] = np.flatnonzero(self.t == tv)

        self.edges_by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for source, target in self.edges.tolist():
            s_idx = self.node_index.get(int(source))
            t_idx = self.node_index.get(int(target))
            if s_idx is None or t_idx is None:
                continue
            edge_time = int(self.t[t_idx])
            self.edges_by_time[edge_time].append((s_idx, t_idx))


class ImageCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 640)
        self.setStyleSheet("background: #111;")
        self._pixmap: Optional[QPixmap] = None
        self._image_shape = (1, 1)
        self._nodes: list[tuple[float, float, str]] = []
        self._edges: list[tuple[float, float, float, float]] = []
        self._show_labels = False

    def set_content(
        self,
        image_u8: np.ndarray,
        nodes: list[tuple[float, float, str]],
        edges: list[tuple[float, float, float, float]],
        show_labels: bool,
    ) -> None:
        image_u8 = np.ascontiguousarray(image_u8)
        height, width = image_u8.shape
        qimage = QImage(
            image_u8.data,
            width,
            height,
            int(image_u8.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._image_shape = (height, width)
        self._nodes = nodes
        self._edges = edges
        self._show_labels = show_labels
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._pixmap is None:
            painter.setPen(Qt.GlobalColor.lightGray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a .zarr dataset")
            return

        img_h, img_w = self._image_shape
        if img_w <= 0 or img_h <= 0:
            return

        scale = min(self.width() / img_w, self.height() / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        left = (self.width() - draw_w) / 2.0
        top = (self.height() - draw_h) / 2.0
        target = QRectF(left, top, draw_w, draw_h)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        def map_xy(x: float, y: float) -> QPointF:
            return QPointF(left + x * scale, top + y * scale)

        edge_pen = QPen(Qt.GlobalColor.cyan)
        edge_pen.setWidthF(max(1.0, 1.2 * scale))
        edge_pen.setCosmetic(True)
        painter.setPen(edge_pen)
        for x1, y1, x2, y2 in self._edges:
            painter.drawLine(map_xy(x1, y1), map_xy(x2, y2))

        node_pen = QPen(Qt.GlobalColor.red)
        node_pen.setWidth(2)
        node_pen.setCosmetic(True)
        painter.setPen(node_pen)
        radius = max(4.0, 3.0 * min(scale, 2.0))
        for x, y, label in self._nodes:
            point = map_xy(x, y)
            painter.drawEllipse(point, radius, radius)
            if self._show_labels:
                painter.drawText(point + QPointF(radius + 2, -radius - 2), label)


class BiohubViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1520, 900)
        self.setAcceptDrops(True)

        self.zarr_path: Optional[Path] = None
        self.geff_path: Optional[Path] = None
        self.image_array = None
        self.graph: Optional[GraphData] = None

        self.canvas = ImageCanvas()

        self.fs_model = QFileSystemModel(self)
        self.fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Drives)
        self.fs_model.setRootPath("")
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.fs_model)
        self.file_tree.setRootIndex(self.fs_model.index(""))
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setAnimated(False)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setIndentation(18)
        self.file_tree.setMinimumWidth(270)
        for column in range(1, 4):
            self.file_tree.hideColumn(column)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Type or paste a folder path...")
        self.path_go_btn = QPushButton("Go")
        self.project_data_btn = QPushButton("Project data")
        self.auto_open_check = QCheckBox("Open .zarr on single click")
        self.auto_open_check.setChecked(True)

        self.open_zarr_btn = QPushButton("Open Zarr...")
        self.open_geff_btn = QPushButton("Open GEFF...")
        self.auto_gt_btn = QPushButton("Find Matching GT")

        self.dataset_label = QLabel("Dataset: -")
        self.shape_label = QLabel("Shape: -")
        self.gt_label = QLabel("GEFF: -")

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 0)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_spin = QSpinBox()
        self.z_spin.setRange(0, 0)

        self.mip_check = QCheckBox("Z maximum projection (MIP)")
        self.nodes_check = QCheckBox("Show nodes")
        self.nodes_check.setChecked(True)
        self.edges_check = QCheckBox("Show tracks / edges")
        self.edges_check.setChecked(True)
        self.labels_check = QCheckBox("Show node IDs")

        self.z_tolerance = QSpinBox()
        self.z_tolerance.setRange(0, 10)
        self.z_tolerance.setValue(1)
        self.z_tolerance.setSuffix(" slices")

        self.trail_length = QSpinBox()
        self.trail_length.setRange(1, 20)
        self.trail_length.setValue(2)
        self.trail_length.setSuffix(" frames")

        self.low_pct = QSpinBox()
        self.low_pct.setRange(0, 20)
        self.low_pct.setValue(1)
        self.low_pct.setSuffix(" %")
        self.high_pct = QSpinBox()
        self.high_pct.setRange(80, 100)
        self.high_pct.setValue(99)
        self.high_pct.setSuffix(" %")

        self._build_ui()
        self._wire_events()
        self._build_menu()

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Drop a .zarr folder here, or click Open Zarr...")

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        explorer = QWidget()
        explorer_layout = QVBoxLayout(explorer)
        explorer_layout.setContentsMargins(0, 0, 0, 0)

        location_row = QHBoxLayout()
        location_row.addWidget(self.path_edit, 1)
        location_row.addWidget(self.path_go_btn)
        explorer_layout.addLayout(location_row)
        explorer_layout.addWidget(self.project_data_btn)
        explorer_layout.addWidget(self.auto_open_check)

        explorer_hint = QLabel("Browse drives and folders. Select a .zarr dataset to preview it immediately.")
        explorer_hint.setWordWrap(True)
        explorer_layout.addWidget(explorer_hint)
        explorer_layout.addWidget(self.file_tree, 1)

        splitter.addWidget(explorer)
        splitter.addWidget(self.canvas)

        side = QWidget()
        side.setMaximumWidth(360)
        side_layout = QVBoxLayout(side)

        file_group = QGroupBox("Files")
        file_layout = QVBoxLayout(file_group)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_zarr_btn)
        buttons.addWidget(self.open_geff_btn)
        file_layout.addLayout(buttons)
        file_layout.addWidget(self.auto_gt_btn)
        file_layout.addWidget(self.dataset_label)
        file_layout.addWidget(self.shape_label)
        file_layout.addWidget(self.gt_label)
        side_layout.addWidget(file_group)

        nav_group = QGroupBox("Navigation")
        nav_layout = QGridLayout(nav_group)
        nav_layout.addWidget(QLabel("Time (T)"), 0, 0)
        nav_layout.addWidget(self.time_slider, 0, 1)
        nav_layout.addWidget(self.time_spin, 0, 2)
        nav_layout.addWidget(QLabel("Depth (Z)"), 1, 0)
        nav_layout.addWidget(self.z_slider, 1, 1)
        nav_layout.addWidget(self.z_spin, 1, 2)
        nav_layout.addWidget(self.mip_check, 2, 0, 1, 3)
        side_layout.addWidget(nav_group)

        overlay_group = QGroupBox("Ground truth overlay")
        overlay_layout = QGridLayout(overlay_group)
        overlay_layout.addWidget(self.nodes_check, 0, 0, 1, 2)
        overlay_layout.addWidget(self.edges_check, 1, 0, 1, 2)
        overlay_layout.addWidget(self.labels_check, 2, 0, 1, 2)
        overlay_layout.addWidget(QLabel("Z tolerance"), 3, 0)
        overlay_layout.addWidget(self.z_tolerance, 3, 1)
        overlay_layout.addWidget(QLabel("Track trail"), 4, 0)
        overlay_layout.addWidget(self.trail_length, 4, 1)
        side_layout.addWidget(overlay_group)

        contrast_group = QGroupBox("Display contrast")
        contrast_layout = QGridLayout(contrast_group)
        contrast_layout.addWidget(QLabel("Low percentile"), 0, 0)
        contrast_layout.addWidget(self.low_pct, 0, 1)
        contrast_layout.addWidget(QLabel("High percentile"), 1, 0)
        contrast_layout.addWidget(self.high_pct, 1, 1)
        side_layout.addWidget(contrast_group)

        help_text = QLabel(
            "Shortcuts\n"
            "Left / Right : previous / next time\n"
            "Up / Down    : previous / next Z\n"
            "M            : toggle MIP\n"
            "N            : toggle nodes\n"
            "E            : toggle edges\n\n"
            "Tip: You can drag a .zarr or .geff folder onto this window."
        )
        help_text.setWordWrap(True)
        side_layout.addWidget(help_text)
        side_layout.addStretch(1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 880, 320])

        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _wire_events(self) -> None:
        self.open_zarr_btn.clicked.connect(self.choose_zarr)
        self.open_geff_btn.clicked.connect(self.choose_geff)
        self.auto_gt_btn.clicked.connect(self.find_and_load_gt)

        self.file_tree.clicked.connect(self._tree_clicked)
        self.file_tree.doubleClicked.connect(self._tree_double_clicked)
        self.path_edit.returnPressed.connect(self._go_to_path)
        self.path_go_btn.clicked.connect(self._go_to_path)
        self.project_data_btn.clicked.connect(self._go_to_project_data)

        self.time_slider.valueChanged.connect(self.time_spin.setValue)
        self.time_spin.valueChanged.connect(self.time_slider.setValue)
        self.z_slider.valueChanged.connect(self.z_spin.setValue)
        self.z_spin.valueChanged.connect(self.z_slider.setValue)

        for widget in (
            self.time_slider,
            self.z_slider,
            self.mip_check,
            self.nodes_check,
            self.edges_check,
            self.labels_check,
            self.z_tolerance,
            self.trail_length,
            self.low_pct,
            self.high_pct,
        ):
            if isinstance(widget, QSlider):
                widget.valueChanged.connect(self.refresh_view)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.refresh_view)
            else:
                widget.valueChanged.connect(self.refresh_view)

        self.mip_check.toggled.connect(lambda checked: self.z_slider.setEnabled(not checked))
        self.mip_check.toggled.connect(lambda checked: self.z_spin.setEnabled(not checked))

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_zarr = QAction("Open Zarr...", self)
        open_zarr.setShortcut(QKeySequence.StandardKey.Open)
        open_zarr.triggered.connect(self.choose_zarr)
        file_menu.addAction(open_zarr)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        shortcuts = [
            (Qt.Key.Key_Left, lambda: self._step_time(-1)),
            (Qt.Key.Key_Right, lambda: self._step_time(1)),
            (Qt.Key.Key_Up, lambda: self._step_z(-1)),
            (Qt.Key.Key_Down, lambda: self._step_z(1)),
            (Qt.Key.Key_M, lambda: self.mip_check.toggle()),
            (Qt.Key.Key_N, lambda: self.nodes_check.toggle()),
            (Qt.Key.Key_E, lambda: self.edges_check.toggle()),
        ]
        for key, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def _step_time(self, delta: int) -> None:
        self.time_slider.setValue(self.time_slider.value() + delta)

    def _step_z(self, delta: int) -> None:
        if not self.mip_check.isChecked():
            self.z_slider.setValue(self.z_slider.value() + delta)

    def _path_from_index(self, index) -> Optional[Path]:
        try:
            raw = self.fs_model.filePath(index)
            if not raw:
                return None
            return Path(raw)
        except Exception:
            return None

    def _tree_clicked(self, index) -> None:
        path = self._path_from_index(index)
        if path is None:
            return
        self.path_edit.setText(str(path))
        if self.auto_open_check.isChecked() and path.name.lower().endswith(".zarr"):
            if self.zarr_path is None or path.resolve() != self.zarr_path.resolve():
                self.load_zarr(path)

    def _tree_double_clicked(self, index) -> None:
        path = self._path_from_index(index)
        if path is None:
            return
        if path.name.lower().endswith(".zarr"):
            self.load_zarr(path)
            return
        if path.name.lower().endswith(".geff"):
            self.load_geff(path)
            return

    def _select_path_in_tree(self, path: Path) -> bool:
        try:
            path = path.resolve()
        except Exception:
            path = Path(path)
        if not path.exists() or not path.is_dir():
            return False
        index = self.fs_model.index(str(path))
        if not index.isValid():
            return False
        self.file_tree.setCurrentIndex(index)
        self.file_tree.scrollTo(index)
        self.file_tree.expand(index)
        self.path_edit.setText(str(path))
        return True

    def _go_to_path(self) -> None:
        text = self.path_edit.text().strip().strip('"')
        if not text:
            return
        path = Path(text)
        if not self._select_path_in_tree(path):
            QMessageBox.information(self, APP_NAME, f"Folder not found:\n{text}")

    def _project_data_path(self) -> Optional[Path]:
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).resolve().parent
        else:
            app_dir = Path(__file__).resolve().parent
        candidates = [
            app_dir.parent.parent / "data",
            app_dir.parent / "data",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _go_to_project_data(self) -> None:
        path = self._project_data_path()
        if path is None:
            QMessageBox.information(self, APP_NAME, "Project data folder was not found automatically.")
            return
        self._select_path_in_tree(path)

    def choose_zarr(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Biohub .zarr folder")
        if path:
            self.load_zarr(Path(path))

    def choose_geff(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Biohub .geff folder")
        if path:
            self.load_geff(Path(path))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.name.lower().endswith(".zarr"):
                self.load_zarr(path)
                return
            if path.name.lower().endswith(".geff"):
                self.load_geff(path)
                return

    def _find_image_array(self, root):
        if hasattr(root, "shape"):
            return root
        if "0" in root:
            candidate = root["0"]
            if hasattr(candidate, "shape"):
                return candidate
        for key in root.keys():
            candidate = root[key]
            if hasattr(candidate, "shape") and len(candidate.shape) in (3, 4):
                return candidate
        raise ValueError("No 3D/4D image array found inside the Zarr group.")

    def load_zarr(self, path: Path) -> None:
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            root = zarr.open(str(path), mode="r")
            arr = self._find_image_array(root)
            if len(arr.shape) not in (3, 4):
                raise ValueError(f"Expected (T,Z,Y,X) or (Z,Y,X), got {arr.shape}")

            self.zarr_path = path
            self.image_array = arr
            self.dataset_label.setText(f"Dataset: {path.stem}")
            self.shape_label.setText(f"Shape: {tuple(arr.shape)}   dtype={arr.dtype}")

            if len(arr.shape) == 4:
                t_count, z_count = int(arr.shape[0]), int(arr.shape[1])
            else:
                t_count, z_count = 1, int(arr.shape[0])
            self.time_slider.setRange(0, max(0, t_count - 1))
            self.time_spin.setRange(0, max(0, t_count - 1))
            self.z_slider.setRange(0, max(0, z_count - 1))
            self.z_spin.setRange(0, max(0, z_count - 1))
            self.time_slider.setValue(0)
            self.z_slider.setValue(z_count // 2)

            self.graph = None
            self.geff_path = None
            self.gt_label.setText("GEFF: not loaded")
            self.statusBar().showMessage(f"Loaded {path.name}")
            self.path_edit.setText(str(path))
            self.refresh_view()
            self.find_and_load_gt(silent=True)
        except Exception as exc:
            self._show_error("Could not open Zarr", exc)
        finally:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def _candidate_geff_paths(self) -> list[Path]:
        if self.zarr_path is None:
            return []
        stem = self.zarr_path.stem
        candidates = [
            self.zarr_path.with_name(stem + ".geff"),
            self.zarr_path.parent / (stem + ".geff"),
            self.zarr_path.parent.parent / (stem + ".geff"),
        ]
        for parent in [self.zarr_path.parent, *self.zarr_path.parents]:
            if parent.name.lower() == "data":
                candidates.extend(
                    [
                        parent / "visible_gt" / "train" / (stem + ".geff"),
                        parent / "full_train_gt" / "train" / (stem + ".geff"),
                        parent / "train_gt" / (stem + ".geff"),
                    ]
                )
                break
        unique = []
        seen = set()
        for item in candidates:
            key = str(item.resolve()) if item.exists() else str(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def find_and_load_gt(self, silent: bool = False) -> None:
        for candidate in self._candidate_geff_paths():
            if candidate.exists():
                self.load_geff(candidate)
                return
        if not silent:
            QMessageBox.information(
                self,
                APP_NAME,
                "No matching .geff folder was found automatically.\nUse Open GEFF... to select one.",
            )

    def load_geff(self, path: Path) -> None:
        try:
            root = zarr.open(str(path), mode="r")
            required = [
                "nodes/ids",
                "nodes/props/t/values",
                "nodes/props/z/values",
                "nodes/props/y/values",
                "nodes/props/x/values",
            ]
            missing = [key for key in required if key not in root]
            if missing:
                raise ValueError("GEFF is missing: " + ", ".join(missing))

            node_ids = np.asarray(root["nodes/ids"][:])
            t = np.asarray(root["nodes/props/t/values"][:])
            z = np.asarray(root["nodes/props/z/values"][:])
            y = np.asarray(root["nodes/props/y/values"][:])
            x = np.asarray(root["nodes/props/x/values"][:])
            if "edges/ids" in root:
                edges = np.asarray(root["edges/ids"][:])
            else:
                edges = np.empty((0, 2), dtype=np.int64)

            self.graph = GraphData(node_ids=node_ids, t=t, z=z, y=y, x=x, edges=edges)
            self.geff_path = path
            self.gt_label.setText(f"GEFF: {path.name}   nodes={len(node_ids):,}  edges={len(edges):,}")
            self.statusBar().showMessage(f"Loaded GT: {path.name}")
            self.refresh_view()
        except Exception as exc:
            self._show_error("Could not open GEFF", exc)

    def _read_plane(self, t: int, z_index: int, mip: bool) -> np.ndarray:
        arr = self.image_array
        if len(arr.shape) == 4:
            if mip:
                return np.asarray(arr[t]).max(axis=0)
            return np.asarray(arr[t, z_index])
        if mip:
            return np.asarray(arr[:]).max(axis=0)
        return np.asarray(arr[z_index])

    def _normalize(self, plane: np.ndarray) -> np.ndarray:
        plane = np.asarray(plane, dtype=np.float32)
        low_p = min(self.low_pct.value(), self.high_pct.value() - 1)
        high_p = max(self.high_pct.value(), low_p + 1)
        low, high = np.percentile(plane, [low_p, high_p])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(plane))
            high = float(np.nanmax(plane))
        if high <= low:
            return np.zeros(plane.shape, dtype=np.uint8)
        scaled = (plane - low) / (high - low)
        return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

    def _overlay_data(self, t: int, z_index: int, mip: bool):
        if self.graph is None:
            return [], []
        graph = self.graph
        nodes: list[tuple[float, float, str]] = []
        edges: list[tuple[float, float, float, float]] = []
        z_tol = self.z_tolerance.value()

        if self.nodes_check.isChecked():
            indices = graph.nodes_by_t.get(t, np.empty(0, dtype=np.int64))
            for idx in indices.tolist():
                if mip or abs(float(graph.z[idx]) - z_index) <= z_tol:
                    nodes.append((float(graph.x[idx]), float(graph.y[idx]), str(int(graph.node_ids[idx]))))

        if self.edges_check.isChecked():
            trail = self.trail_length.value()
            start_t = max(0, t - trail + 1)
            for edge_time in range(start_t, t + 1):
                for s_idx, target_idx in graph.edges_by_time.get(edge_time, []):
                    if not mip:
                        z_ok = (
                            abs(float(graph.z[s_idx]) - z_index) <= z_tol
                            or abs(float(graph.z[target_idx]) - z_index) <= z_tol
                        )
                        if not z_ok:
                            continue
                    edges.append(
                        (
                            float(graph.x[s_idx]),
                            float(graph.y[s_idx]),
                            float(graph.x[target_idx]),
                            float(graph.y[target_idx]),
                        )
                    )
        return nodes, edges

    def refresh_view(self, *_args) -> None:
        if self.image_array is None:
            return
        try:
            t = self.time_slider.value()
            z_index = self.z_slider.value()
            mip = self.mip_check.isChecked()
            plane = self._read_plane(t, z_index, mip)
            image_u8 = self._normalize(plane)
            nodes, edges = self._overlay_data(t, z_index, mip)
            self.canvas.set_content(image_u8, nodes, edges, self.labels_check.isChecked())
            mode = "MIP" if mip else f"Z={z_index}"
            self.statusBar().showMessage(
                f"T={t}  {mode}  nodes shown={len(nodes):,}  edges shown={len(edges):,}"
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Render error: {exc}")

    def _show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    viewer = BiohubViewer()
    viewer.show()

    args = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("-")]
    for path in args:
        if path.name.lower().endswith(".zarr"):
            viewer.load_zarr(path)
        elif path.name.lower().endswith(".geff"):
            viewer.load_geff(path)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
