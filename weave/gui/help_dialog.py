"""Help dialog with keyboard shortcuts and quick-start guide."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_HELP_HTML = """
<style>
    body { font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 13px; }
    h2 { margin-top: 16px; margin-bottom: 8px; color: #1976D2; }
    h3 { margin-top: 12px; margin-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 12px; }
    th, td { padding: 6px 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f5f5f5; font-weight: 600; }
    code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
    .tip { background: #e3f2fd; border-left: 4px solid #1976D2; padding: 8px 12px; margin: 8px 0; }
</style>

<h2>Weave Editor &mdash; Quick Reference</h2>

<h3>Getting Started</h3>
<div class="tip">
    <strong>Fastest path:</strong> Open the hamburger menu (top-right) &rarr;
    <em>New Code from Template...</em> &rarr; pick a code &rarr; click <em>Load</em>.
    Then <em>Simulate...</em> to run.
</div>

<h3>Canvas Navigation</h3>
<table>
    <tr><th>Action</th><th>How</th></tr>
    <tr><td>Pan</td><td>Click &amp; drag empty space</td></tr>
    <tr><td>Zoom in/out</td><td>Mouse wheel &nbsp;/&nbsp; <code>Ctrl+=</code> / <code>Ctrl+-</code></td></tr>
    <tr><td>Reset zoom</td><td><code>Ctrl+0</code></td></tr>
    <tr><td>Toggle grid snap</td><td><code>G</code></td></tr>
</table>

<h3>Node &amp; Edge Operations</h3>
<table>
    <tr><th>Action</th><th>How</th></tr>
    <tr><td>Create node</td><td>Right-click canvas &rarr; <em>New Quantum Node</em> / <em>New Bit</em></td></tr>
    <tr><td>Create edge</td><td><code>Ctrl+click</code> source node, then click target node</td></tr>
    <tr><td>Select node</td><td>Click node</td></tr>
    <tr><td>Multi-select</td><td><code>Shift+drag</code> to draw a selection rectangle</td></tr>
    <tr><td>Select all</td><td><code>Ctrl+A</code></td></tr>
    <tr><td>Delete selection</td><td><code>Delete</code> or <code>Backspace</code></td></tr>
    <tr><td>Copy / Paste</td><td><code>Ctrl+C</code> / <code>Ctrl+V</code></td></tr>
    <tr><td>Deselect all</td><td><code>Escape</code></td></tr>
</table>

<h3>Graphs &amp; Codes</h3>
<table>
    <tr><th>Action</th><th>How</th></tr>
    <tr><td>Detect graph</td><td>Right-click a node &rarr; <em>Detect</em></td></tr>
    <tr><td>Simulate graph</td><td>Right-click detected graph &rarr; <em>Configure &amp; Simulate...</em></td></tr>
    <tr><td>Load template code</td><td>Hamburger menu &rarr; <em>New Code from Template...</em></td></tr>
</table>

<h3>File Operations</h3>
<table>
    <tr><th>Action</th><th>How</th></tr>
    <tr><td>Save canvas</td><td><code>Ctrl+S</code></td></tr>
    <tr><td>Load canvas</td><td><code>Ctrl+O</code></td></tr>
    <tr><td>Export code (CSV)</td><td>Hamburger menu &rarr; <em>Export Code...</em></td></tr>
</table>

<h3>Simulation Dialog</h3>
<p>The simulation dialog has two tabs:</p>
<ul>
    <li><strong>Configuration:</strong> Set noise channels, experiment type,
        logical operators, and geometry-induced noise (kernel, J&#x2080;, &tau;).</li>
    <li><strong>Simulation:</strong> Choose rounds, shots, decoder. Click
        <em>Run</em> to start. Results show LER, error count, elapsed time,
        and (if geometry noise is on) exposure metrics J&#x1D458; and pair event count.</li>
</ul>

<h3>Geometry Noise Parameters</h3>
<table>
    <tr><th>Parameter</th><th>Symbol</th><th>Meaning</th></tr>
    <tr><td>J&#x2080;</td><td>J<sub>0</sub></td><td>Microscopic coupling scale. Larger &rarr; stronger correlated noise.</td></tr>
    <tr><td>&tau;</td><td>&tau;</td><td>Tick duration. Scales the coupling linearly.</td></tr>
    <tr><td>&alpha;</td><td>&alpha;</td><td>Power-law decay exponent. Larger &rarr; faster fall-off with distance.</td></tr>
    <tr><td>r&#x2080;</td><td>r<sub>0</sub></td><td>Regularization length. Prevents divergence at zero separation.</td></tr>
    <tr><td>&xi;</td><td>&xi;</td><td>Exponential decay length. Characteristic distance scale.</td></tr>
</table>
<p>The pair probability at routed distance <em>d</em> is:
<code>p(d) = sin&sup2;(&tau; J&#x2080; &kappa;(d))</code>,
where <code>&kappa;</code> is the kernel.</p>

<h3>About Weave</h3>
<p>Weave is a geometry-aware compiler for CSS syndrome extraction
with correlated noise from routed embeddings. It targets the
<a href="https://quantum-journal.org/">Quantum Journal</a>.</p>
"""


class HelpDialog(QDialog):
    """Modal dialog showing keyboard shortcuts and a quick-start guide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Weave Editor — Help")
        self.setMinimumSize(520, 500)
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        label = QLabel(_HELP_HTML)
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        content_layout.addWidget(label)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
