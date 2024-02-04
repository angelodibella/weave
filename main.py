import stim

import circuit


def save_diagram(circuit: stim.Circuit, filepath: str, **kwargs) -> None:
    diagram = circuit.diagram(**kwargs)
    with open(filepath, "w") as file:
        file.write(str(diagram))


def display_samples(circuit: stim.Circuit, shots: int = 1, rounds=1) -> None:
    sampler = circuit.compile_sampler().sample(shots)
    for i, shot in enumerate(sampler, start=1):
        round_list = []
        for j, outcome in enumerate(shot):
            round_list.append("x" if outcome else "_")
            if (j + 1) % (len(shot) / rounds) == 0:
                round_list[j] += "\n"
        print(f"Shot {i}:\n" + "".join(round_list))
    print("\n")


bell = circuit.simple_circuit()
save_diagram(bell, "temp_results/bell_diagram.svg", type="timeline-svg")
bell.to_file("temp_results/bell.stim")

sampler = bell.compile_sampler()
print(sampler.sample(shots=10))

rep_3 = circuit.repetition_code(4, 5, noise_data=[0.1, 0.01])
print(rep_3, "\n")

display_samples(rep_3, shots=3, rounds=5)
