import model

rep_model = model.StabilizerModel(rounds=5)

rep_model.set_noise_qubits(0.02)
rep_model.set_noise_circuit(0.01)

rep_model.repetition_code(8)
rep_model.measure()

rep_model.display_samples()
rep_model.display_detector_samples()

rep_model.circuit.diagram()
