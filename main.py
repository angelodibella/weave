import model
import numpy as np

rep_model = model.StabilizerModel(
    "repetition_code",
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

print()
adjacency_matrix = np.array(
    [
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
    ]
)
pos = {0: (0, 0), 1: (1, 1), 2: (2, 0.5), 3: (1, 0), 4: (0, 1), 5: (0.5, 0.5)}

adjacency_matrix = np.array(
    [[0, 1, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
)

pos = {0: (0, -0.1), 1: (1, 0), 2: (2, 0), 3: (1, 1), 4: (1, -1)}
print(model.intersecting_edges(adjacency_matrix, pos))

print()

clist_hamming = ["B", "C", 0, 1, 5, 6, "B", "B", "C", 1, 2, 4, 5, "B", "B", "B", "C", 3, 4, 5, 6, "B"]
hamming_pcm = model.classical_pcm(clist_hamming)
print(hamming_pcm, "\n")

clist_rep = ["B", "C", 0, 1, "B", "C", 1, 2, "B"]
rep_pcm = model.classical_pcm(clist_rep)
print(rep_pcm, "\n")

hp_pcm = model.hypergraph_pcm(rep_pcm, hamming_pcm)
print(hp_pcm[0], "\n")
print(hp_pcm[1])
