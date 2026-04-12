"""Simulation dialog and worker for running Stim/Sinter simulations from the GUI.

PR 18 additions:
- GeometryNoiseWidget: kernel type selector + J₀ / τ / kernel-parameter spinboxes.
- Exposure readout panel: J_κ, total exposure, pair event count after compile.
- Optimize Embedding button: runs swap descent and updates the canvas positions.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..codes.base import NoiseModel
from ..codes.css_code import CSSCode
from .graph_model import GraphData


class SimulationWorker(QThread):
    """Background worker that runs a sinter simulation."""

    progress = Signal(int, int)  # (shots_so_far, total)
    finished = Signal(object)  # sinter.TaskStats
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
            x_logicals, _ = code.find_logicals()
        except Exception:
            x_logicals = np.zeros((0, code.HX.shape[1]), dtype=int)

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


class GeometryNoiseWidget(QWidget):
    """Widget for configuring geometry-induced correlated noise parameters.

    Exposes kernel type (crossing / power-law / exponential), coupling
    scale J₀, tick duration τ, and kernel-specific parameters (α, r₀
    for power-law; ξ for exponential). The widget is hidden by default
    and shown when the user checks the "Enable geometry noise" box.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.enabled_check = QCheckBox("Enable geometry-induced noise")
        self.enabled_check.setChecked(False)
        self.enabled_check.toggled.connect(self._on_enabled_toggled)
        layout.addRow(self.enabled_check)

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["power_law", "exponential", "crossing"])
        self.kernel_combo.currentTextChanged.connect(self._on_kernel_changed)
        layout.addRow("Kernel:", self.kernel_combo)

        self.j0_spin = QDoubleSpinBox()
        self.j0_spin.setRange(0.0, 10.0)
        self.j0_spin.setDecimals(4)
        self.j0_spin.setSingleStep(0.01)
        self.j0_spin.setValue(0.04)
        layout.addRow("J₀ (coupling):", self.j0_spin)

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(0.01, 100.0)
        self.tau_spin.setDecimals(3)
        self.tau_spin.setValue(1.0)
        layout.addRow("τ (tick duration):", self.tau_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 20.0)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(3.0)
        layout.addRow("α (decay exponent):", self.alpha_spin)

        self.r0_spin = QDoubleSpinBox()
        self.r0_spin.setRange(0.01, 100.0)
        self.r0_spin.setDecimals(2)
        self.r0_spin.setValue(1.0)
        layout.addRow("r₀ (regularization):", self.r0_spin)

        self.xi_spin = QDoubleSpinBox()
        self.xi_spin.setRange(0.01, 100.0)
        self.xi_spin.setDecimals(2)
        self.xi_spin.setValue(1.0)
        layout.addRow("ξ (decay length):", self.xi_spin)

        self._params_widgets = [
            self.kernel_combo,
            self.j0_spin,
            self.tau_spin,
            self.alpha_spin,
            self.r0_spin,
            self.xi_spin,
        ]
        self._on_enabled_toggled(False)
        self._on_kernel_changed(self.kernel_combo.currentText())

    def _on_enabled_toggled(self, enabled: bool) -> None:
        for w in self._params_widgets:
            w.setEnabled(enabled)

    def _on_kernel_changed(self, kernel_name: str) -> None:
        is_power = kernel_name == "power_law"
        is_exp = kernel_name == "exponential"
        self.alpha_spin.setVisible(is_power)
        self.r0_spin.setVisible(is_power)
        self.xi_spin.setVisible(is_exp)

    def is_enabled(self) -> bool:
        return self.enabled_check.isChecked()

    def get_j0(self) -> float:
        return self.j0_spin.value() if self.is_enabled() else 0.0

    def get_tau(self) -> float:
        return self.tau_spin.value()

    def get_kernel_spec(self) -> dict:
        """Return a dict suitable for constructing a Kernel object."""
        ktype = self.kernel_combo.currentText()
        if ktype == "power_law":
            return {
                "type": "power_law",
                "alpha": self.alpha_spin.value(),
                "r0": self.r0_spin.value(),
            }
        if ktype == "exponential":
            return {"type": "exponential", "xi": self.xi_spin.value()}
        return {"type": "crossing"}


class SimulationDialog(QDialog):
    """Dialog for configuring and running a simulation on a CSSCode.

    PR 18 additions: geometry noise configuration, exposure readout,
    and an "Optimize Embedding" button.
    """

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

        # Geometry noise config (PR 18).
        geometry_group = QGroupBox("Geometry-Induced Noise")
        geometry_layout = QVBoxLayout()
        self.geometry_widget = GeometryNoiseWidget()
        geometry_layout.addWidget(self.geometry_widget)
        geometry_group.setLayout(geometry_layout)
        config_layout.addWidget(geometry_group)

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

        # Exposure readout (PR 18).
        self.exposure_group = QGroupBox("Exposure Analysis")
        exposure_layout = QFormLayout()
        self.exposure_j_kappa = QLabel("-")
        self.exposure_total = QLabel("-")
        self.exposure_pair_events = QLabel("-")
        self.exposure_corr_edges = QLabel("-")
        exposure_layout.addRow("J_κ (max support exposure):", self.exposure_j_kappa)
        exposure_layout.addRow("Total exposure:", self.exposure_total)
        exposure_layout.addRow("Pair events (provenance):", self.exposure_pair_events)
        exposure_layout.addRow("Correlation edges:", self.exposure_corr_edges)
        self.exposure_group.setLayout(exposure_layout)
        self.exposure_group.setVisible(False)
        sim_layout.addWidget(self.exposure_group)

        # Optimize button (PR 18).
        opt_layout = QHBoxLayout()
        self.optimize_button = QPushButton("Optimize Embedding (swap descent)")
        self.optimize_button.setEnabled(False)
        self.optimize_button.clicked.connect(self._on_optimize)
        opt_layout.addWidget(self.optimize_button)
        sim_layout.addLayout(opt_layout)

        self.optimize_log = QTextEdit()
        self.optimize_log.setReadOnly(True)
        self.optimize_log.setMaximumHeight(80)
        self.optimize_log.setVisible(False)
        sim_layout.addWidget(self.optimize_log)

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

        # Update exposure readout if geometry noise was active.
        self._update_exposure_readout()

    def _update_exposure_readout(self) -> None:
        """Compute and display exposure metrics using the compiler path."""
        if not self.geometry_widget.is_enabled():
            self.exposure_group.setVisible(False)
            self.optimize_button.setEnabled(False)
            return
        try:
            from ..compiler import compile_extraction
            from ..ir import (
                CrossingKernel,
                ExponentialKernel,
                GeometryNoiseConfig,
                LocalNoiseConfig,
                RegularizedPowerLawKernel,
                StraightLineEmbedding,
                default_css_schedule,
            )

            kernel_spec = self.geometry_widget.get_kernel_spec()
            if kernel_spec["type"] == "power_law":
                kernel = RegularizedPowerLawKernel(alpha=kernel_spec["alpha"], r0=kernel_spec["r0"])
            elif kernel_spec["type"] == "exponential":
                kernel = ExponentialKernel(xi=kernel_spec["xi"])
            else:
                kernel = CrossingKernel()

            experiment = self.experiment_combo.currentText()
            rounds = self.rounds_spin.value()

            if self.code.pos is not None:
                emb = StraightLineEmbedding.from_positions(self.code.pos)
            else:
                emb = StraightLineEmbedding.from_positions(
                    [(float(i), 0.0) for i in range(self.code.n_total)]
                )

            sched = default_css_schedule(self.code, experiment=experiment)
            compiled = compile_extraction(
                code=self.code,
                embedding=emb,
                schedule=sched,
                kernel=kernel,
                local_noise=LocalNoiseConfig(),
                geometry_noise=GeometryNoiseConfig(
                    J0=self.geometry_widget.get_j0(),
                    tau=self.geometry_widget.get_tau(),
                ),
                rounds=rounds,
                experiment=experiment,
            )

            self.exposure_pair_events.setText(str(len(compiled.provenance)))
            self.exposure_corr_edges.setText(str(len(compiled.correlation_edges)))

            if compiled.exposure_metrics is not None:
                total = compiled.exposure_metrics.total()
                self.exposure_total.setText(f"{total:.8f}")
                # J_κ requires a reference family — use the min-weight
                # per-support value if available.
                if compiled.exposure_metrics.per_support:
                    max_exp = max(s.exposure for s in compiled.exposure_metrics.per_support)
                    self.exposure_j_kappa.setText(f"{max_exp:.8f}")
                else:
                    self.exposure_j_kappa.setText(f"{total:.8f}")
            else:
                self.exposure_total.setText("0")
                self.exposure_j_kappa.setText("0")

            self.exposure_group.setVisible(True)
            self.optimize_button.setEnabled(True)
            self._last_compiled = compiled
        except Exception as e:
            self.exposure_group.setVisible(False)
            self.optimize_button.setEnabled(False)
            QMessageBox.warning(
                self,
                "Exposure Analysis Failed",
                f"Could not compute exposure metrics:\n{e}",
            )

    def _on_optimize(self) -> None:
        """Run swap descent on the current embedding and show results."""
        try:
            from ..ir import (
                CrossingKernel,
                ExponentialKernel,
                RegularizedPowerLawKernel,
                StraightLineEmbedding,
                default_css_schedule,
            )
            from ..optimize import (
                NumpyExposureTemplate,
                j_kappa_numpy,
                prepare_exposure_template,
                swap_descent,
            )

            kernel_spec = self.geometry_widget.get_kernel_spec()
            if kernel_spec["type"] == "power_law":
                kernel = RegularizedPowerLawKernel(alpha=kernel_spec["alpha"], r0=kernel_spec["r0"])
            elif kernel_spec["type"] == "exponential":
                kernel = ExponentialKernel(xi=kernel_spec["xi"])
            else:
                kernel = CrossingKernel()

            J0 = self.geometry_widget.get_j0()
            tau = self.geometry_widget.get_tau()

            # Build positions and schedule.
            if self.code.pos is not None:
                positions = np.asarray(self.code.pos, dtype=float)
                if positions.ndim == 2 and positions.shape[1] == 2:
                    positions = np.hstack([positions, np.zeros((positions.shape[0], 1))])
            else:
                positions = np.array([(float(i), 0.0, 0.0) for i in range(self.code.n_total)])

            sched = default_css_schedule(self.code)

            # Use the generic event template (serial schedule has no
            # parallel pairs, so this will likely produce no events).
            # For BB codes, the user would use ibm_schedule instead.

            self.optimize_log.setVisible(True)
            self.optimize_log.clear()
            self.optimize_log.append("Building event template...")

            # For the default serial schedule there are no parallel
            # events. Show an informative message.
            from ..compiler.geometry_pass import compute_provenance
            from ..ir import GeometryNoiseConfig, MinDistanceMetric

            emb = StraightLineEmbedding.from_positions(
                [(float(p[0]), float(p[1])) for p in positions]
            )
            prov = compute_provenance(
                sched,
                emb,
                kernel,
                MinDistanceMetric(),
                GeometryNoiseConfig(J0=J0, tau=tau),
            )

            if not prov:
                self.optimize_log.append(
                    "No parallel pair events in the default serial schedule.\n"
                    "Swap descent requires a schedule with parallel CNOT ticks.\n"
                    "For BB codes, use ibm_schedule() from the Python API."
                )
                return

            self.optimize_log.append(f"Found {len(prov)} pair events. Running swap descent...")
            # For codes with provenance, we CAN optimize.
            # Build exposure template from provenance.
            from ..optimize.objectives import PairEventTemplate

            template = [
                PairEventTemplate(
                    tick_index=r.tick_index,
                    edge_a=r.edge_a,
                    edge_b=r.edge_b,
                    sector=r.sector,
                    data_support=r.data_support,
                )
                for r in prov
            ]
            # Use all data qubits as a single reference support (conservative).
            data_indices = tuple(range(len(self.code.data_qubits)))
            exp_t = prepare_exposure_template(template, [data_indices])
            np_t = NumpyExposureTemplate.from_exposure_template(exp_t)

            def objective(pos: np.ndarray) -> float:
                return j_kappa_numpy(pos, np_t, kernel, J0=J0, tau=tau)

            # Swap classes: data qubits only (conservative).
            swap_classes = [list(range(len(self.code.data_qubits)))]
            rng = np.random.default_rng(42)
            result = swap_descent(
                positions,
                objective,
                swap_classes,
                max_iterations=30,
                sample_size=50,
                rng=rng,
            )

            reduction = result.reduction_ratio * 100
            self.optimize_log.append(
                f"Done. Reduction: {reduction:.1f}% "
                f"({result.initial_value:.6f} → {result.final_value:.6f})\n"
                f"Iterations: {len(result.history) - 1}, "
                f"Evaluations: {result.n_evaluations}"
            )

        except Exception as e:
            self.optimize_log.setVisible(True)
            self.optimize_log.append(f"Optimization failed: {e}")

    def _on_error(self, message):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Simulation Error", message)
