"""Simulation dialog and worker for running Stim/Sinter simulations from the GUI."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QComboBox, QPushButton, QProgressBar, QGroupBox,
    QFormLayout, QFrame, QTabWidget, QWidget, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal

from ..codes.css_code import CSSCode
from ..codes.base import NoiseModel
from .graph_model import GraphData


class SimulationWorker(QThread):
    """Background worker that runs a sinter simulation."""

    progress = Signal(int, int)     # (shots_so_far, total)
    finished = Signal(object)       # sinter.TaskStats
    error = Signal(str)

    def __init__(self, circuit, shots, decoder="bposd"):
        super().__init__()
        self.circuit = circuit
        self.shots = shots
        self.decoder = decoder

    def run(self):
        try:
            import sinter

            task = sinter.Task(
                circuit=self.circuit,
                decoder=self.decoder,
                json_metadata={"source": "weave_gui"},
            )

            results = sinter.collect(
                num_workers=1,
                tasks=[task],
                max_shots=self.shots,
                decoders=[self.decoder],
            )

            if results:
                self.finished.emit(results[0])
            else:
                self.error.emit("Sinter returned no results.")
        except Exception as e:
            self.error.emit(str(e))


class NoiseConfigWidget(QWidget):
    """Widget for configuring the 5-channel NoiseModel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.channels: dict[str, QDoubleSpinBox] = {}
        channel_names = [
            ("data", "Data qubit noise"),
            ("z_check", "Z-check noise"),
            ("x_check", "X-check noise"),
            ("circuit", "Circuit (2Q gate) noise"),
            ("crossing", "Crossing noise"),
        ]

        for key, label_text in channel_names:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 0.5)
            spin.setDecimals(5)
            spin.setSingleStep(0.0001)
            spin.setValue(0.001)
            self.channels[key] = spin
            layout.addRow(f"{label_text}:", spin)

    def get_noise_model(self) -> NoiseModel:
        return NoiseModel(
            data=self.channels["data"].value(),
            z_check=self.channels["z_check"].value(),
            x_check=self.channels["x_check"].value(),
            circuit=self.channels["circuit"].value(),
            crossing=self.channels["crossing"].value(),
        )

    def get_config_dict(self) -> dict[str, float]:
        return {key: spin.value() for key, spin in self.channels.items()}

    def set_from_dict(self, config: dict[str, float]) -> None:
        for key, value in config.items():
            if key in self.channels:
                self.channels[key].setValue(value)


class LogicalSelectionWidget(QWidget):
    """Widget for selecting which logical operators to simulate."""

    def __init__(self, code: CSSCode, parent=None):
        super().__init__(parent)
        self.code = code
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        k = code.k
        if k == 0:
            layout.addWidget(QLabel("No logical qubits."))
            self.checkboxes = []
            return

        label = QLabel(f"Select logical qubits to simulate ({k} available):")
        layout.addWidget(label)

        self.checkboxes: list[QCheckBox] = []

        try:
            x_logicals, z_logicals = code.find_logicals()
        except Exception:
            x_logicals = np.zeros((0, code.HX.shape[1]), dtype=int)
            z_logicals = np.zeros((0, code.HZ.shape[1]), dtype=int)

        for i in range(k):
            # Build support info string.
            support_info = ""
            if i < x_logicals.shape[0]:
                x_support = np.nonzero(x_logicals[i])[0]
                support_info = f"  (data qubits: {', '.join(str(q) for q in x_support)})"

            cb = QCheckBox(f"Logical {i}{support_info}")
            cb.setChecked(True)
            self.checkboxes.append(cb)
            layout.addWidget(cb)

    def selected_indices(self) -> list[int]:
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]


class SimulationDialog(QDialog):
    """Dialog for configuring and running a simulation on a CSSCode."""

    def __init__(
        self,
        code: CSSCode,
        parent=None,
        graph_data: GraphData | None = None,
    ):
        super().__init__(parent)
        self.code = code
        self.graph_data = graph_data
        self.worker = None
        self._start_time = None

        self.setWindowTitle("Configure & Simulate")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: Code & Configuration ---
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)

        # Code info.
        info_group = QGroupBox("Code Parameters")
        info_layout = QFormLayout()

        num_data = len(self.code.data_qubits)
        k = self.code.k
        crossings = len(self.code.crossings)
        info_layout.addRow("Code:", QLabel(f"[[{num_data}, {k}]]"))
        info_layout.addRow("Data qubits (n):", QLabel(str(num_data)))
        info_layout.addRow("Logical qubits (k):", QLabel(str(k)))
        info_layout.addRow("Z checks:", QLabel(str(len(self.code.z_check_qubits))))
        info_layout.addRow("X checks:", QLabel(str(len(self.code.x_check_qubits))))
        info_layout.addRow("Crossings:", QLabel(str(crossings)))

        info_group.setLayout(info_layout)
        config_layout.addWidget(info_group)

        # Noise config.
        noise_group = QGroupBox("Noise Configuration")
        noise_layout = QVBoxLayout()
        self.noise_widget = NoiseConfigWidget()
        noise_layout.addWidget(self.noise_widget)

        # Pre-fill from graph_data if available.
        if self.graph_data and self.graph_data.noise_config:
            self.noise_widget.set_from_dict(self.graph_data.noise_config)

        noise_group.setLayout(noise_layout)
        config_layout.addWidget(noise_group)

        # Experiment selector.
        experiment_group = QGroupBox("Experiment")
        experiment_layout = QFormLayout()
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(["z_memory", "x_memory"])
        experiment_layout.addRow("Type:", self.experiment_combo)
        experiment_group.setLayout(experiment_layout)
        config_layout.addWidget(experiment_group)

        # Logical selection.
        logical_group = QGroupBox("Logical Selection")
        logical_layout = QVBoxLayout()
        self.logical_widget = LogicalSelectionWidget(self.code)
        logical_layout.addWidget(self.logical_widget)

        # Pre-select from graph_data if available.
        if self.graph_data and self.graph_data.logical_indices is not None:
            for i, cb in enumerate(self.logical_widget.checkboxes):
                cb.setChecked(i in self.graph_data.logical_indices)

        logical_group.setLayout(logical_layout)
        config_layout.addWidget(logical_group)

        config_layout.addStretch()
        self.tabs.addTab(config_tab, "Configuration")

        # --- Tab 2: Simulation & Results ---
        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)

        # Simulation parameters.
        param_group = QGroupBox("Simulation Parameters")
        param_layout = QFormLayout()

        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 100)
        self.rounds_spin.setValue(3)
        param_layout.addRow("Rounds:", self.rounds_spin)

        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(100, 10_000_000)
        self.shots_spin.setSingleStep(1000)
        self.shots_spin.setValue(10000)
        param_layout.addRow("Shots:", self.shots_spin)

        self.decoder_combo = QComboBox()
        try:
            import pymatching  # noqa: F401
            self.decoder_combo.addItem("pymatching")
        except ImportError:
            pass
        self.decoder_combo.addItem("bposd")
        param_layout.addRow("Decoder:", self.decoder_combo)

        param_group.setLayout(param_layout)
        sim_layout.addWidget(param_group)

        # Run button.
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._on_run)
        button_layout.addWidget(self.run_button)
        sim_layout.addLayout(button_layout)

        # Progress bar.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        sim_layout.addWidget(self.progress_bar)

        # Results.
        self.results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        self.result_ler = QLabel("-")
        self.result_shots = QLabel("-")
        self.result_errors = QLabel("-")
        self.result_time = QLabel("-")
        results_layout.addRow("Logical error rate:", self.result_ler)
        results_layout.addRow("Shots:", self.result_shots)
        results_layout.addRow("Errors:", self.result_errors)
        results_layout.addRow("Elapsed time:", self.result_time)
        self.results_group.setLayout(results_layout)
        self.results_group.setVisible(False)
        sim_layout.addWidget(self.results_group)

        sim_layout.addStretch()
        self.tabs.addTab(sim_tab, "Simulation")

        # --- Close button ---
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        close_layout.addWidget(self.close_button)
        layout.addLayout(close_layout)

    def _on_run(self):
        """Build circuit with user parameters and start simulation."""
        noise = self.noise_widget.get_noise_model()
        rounds = self.rounds_spin.value()
        experiment = self.experiment_combo.currentText()
        shots = self.shots_spin.value()
        decoder = self.decoder_combo.currentText()
        logical_indices = self.logical_widget.selected_indices()

        # Store config back to graph_data if available.
        if self.graph_data:
            self.graph_data.noise_config = self.noise_widget.get_config_dict()
            self.graph_data.logical_indices = logical_indices

        # Build logical parameter for CSSCode.
        logical = logical_indices if logical_indices else None

        code = CSSCode(
            HX=self.code.HX,
            HZ=self.code.HZ,
            rounds=rounds,
            noise=noise,
            experiment=experiment,
            logical=logical,
        )

        if self.code.pos is not None:
            code.embed(pos=self.code.pos)

        circuit = code.circuit

        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.results_group.setVisible(False)

        self._start_time = time.monotonic()

        self.worker = SimulationWorker(circuit, shots, decoder)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, stats):
        elapsed = time.monotonic() - self._start_time

        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.results_group.setVisible(True)

        shots = stats.shots
        errors = stats.errors
        ler = errors / shots if shots > 0 else 0.0

        self.result_ler.setText(f"{ler:.6f}")
        self.result_shots.setText(str(shots))
        self.result_errors.setText(str(errors))
        self.result_time.setText(f"{elapsed:.2f}s")

    def _on_error(self, message):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Simulation Error", message)
