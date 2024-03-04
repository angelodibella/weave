import model

rep_model = model.StabilizerModel(
    "repetition",
    distance=15,
    rounds=10,
    noise_circuit=0.0,
    noise_data=[0.01, 0.03, 0.06],
    noise_x_check=0.02,
    noise_z_check=0.02,
)

rep_model.reset_data_qubits()

rep_model.display_samples()
rep_model.display_detector_samples()

rep_model.circuit.diagram()
