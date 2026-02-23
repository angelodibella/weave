"""Simulation dialog and worker for running Stim/Sinter simulations from the GUI."""

import time

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QComboBox, QPushButton, QProgressBar, QGroupBox,
    QFormLayout, QDialogButtonBox, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal

from ..codes.css_code import CSSCode
from ..codes.base import NoiseModel


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


class SimulationDialog(QDialog):
    """Dialog for configuring and running a simulation on a CSSCode."""

    def __init__(self, code: CSSCode, parent=None):
        super().__init__(parent)
        self.code = code
        self.worker = None
        self._start_time = None

        self.setWindowTitle("Simulate Code")
        self.setMinimumWidth(380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Code info ---
        info_group = QGroupBox("Code Parameters")
        info_layout = QFormLayout()

        num_data = len(self.code.data_qubits)
        k = self.code.k
        crossings = len(self.code.crossings)
        info_layout.addRow("Data qubits (n):", QLabel(str(num_data)))
        info_layout.addRow("Logical qubits (k):", QLabel(str(k)))
        info_layout.addRow("Crossings:", QLabel(str(crossings)))

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- Simulation parameters ---
        param_group = QGroupBox("Simulation Parameters")
        param_layout = QFormLayout()

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setDecimals(4)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(0.001)
        param_layout.addRow("Noise rate:", self.noise_spin)

        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 100)
        self.rounds_spin.setValue(3)
        param_layout.addRow("Rounds:", self.rounds_spin)

        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(["z_memory", "x_memory"])
        param_layout.addRow("Experiment:", self.experiment_combo)

        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(100, 10_000_000)
        self.shots_spin.setSingleStep(1000)
        self.shots_spin.setValue(10000)
        param_layout.addRow("Shots:", self.shots_spin)

        self.decoder_combo = QComboBox()
        # Check for pymatching availability.
        try:
            import pymatching  # noqa: F401
            self.decoder_combo.addItem("pymatching")
        except ImportError:
            pass
        self.decoder_combo.addItem("bposd")
        param_layout.addRow("Decoder:", self.decoder_combo)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Results ---
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
        layout.addWidget(self.results_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        button_layout.addWidget(self.run_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _on_run(self):
        """Build circuit with user parameters and start simulation."""
        noise_rate = self.noise_spin.value()
        rounds = self.rounds_spin.value()
        experiment = self.experiment_combo.currentText()
        shots = self.shots_spin.value()
        decoder = self.decoder_combo.currentText()

        # Rebuild CSSCode with user's noise/rounds/experiment.
        noise = NoiseModel(
            data=noise_rate,
            circuit=noise_rate,
            crossing=noise_rate,
            z_check=noise_rate,
            x_check=noise_rate,
        )

        code = CSSCode(
            HX=self.code.HX,
            HZ=self.code.HZ,
            rounds=rounds,
            noise=noise,
            experiment=experiment,
        )

        # Re-embed with original positions for crossing detection.
        if self.code.pos is not None:
            code.embed(pos=self.code.pos)

        circuit = code.circuit

        # Update UI.
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.results_group.setVisible(False)

        self._start_time = time.monotonic()

        self.worker = SimulationWorker(circuit, shots, decoder)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, stats):
        """Handle simulation completion."""
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
        """Handle simulation error."""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Simulation Error", message)
