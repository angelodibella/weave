import stim

import circuit

def save_diagram_svg(circuit: stim.Circuit, filepath: str) -> None:
    diagram = circuit.diagram('timeline-svg')
    with open(filepath, 'w') as file:
        file.write(str(diagram))

bell = circuit.simple_circuit()
save_diagram_svg(bell, 'temp_results/bell_diagram.svg')

sampler = bell.compile_sampler()
print(sampler.sample(shots=10))
