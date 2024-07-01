
from bqskit import Circuit, compile
from bqskit.compiler import Compiler

from bqskit.compiler import Workflow
from bqskit.passes import QuickPartitioner, ForEachBlockPass, ScanningGateRemovalPass, UnfoldPass
from bqskit.compiler.compile import get_instantiate_options
from bqskit import enable_logging
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
#enable_logging(True)

# Specify the full path to the QASM file

qasm_file_path = 'test.qasm'
basic_gate_deletion_workflow = Workflow([
    QuickPartitioner(3),  # Partition into 3-qubit blocks
    ForEachBlockPass(ScanningGateRemovalPass()),  # Apply gate deletion to each block (in parallel)
    UnfoldPass(),  # Unfold the blocks back into the original circuit
])

# QSearch synthesis with high-quality settings followed by gate deletion.
instantiate_options = get_instantiate_options(optimization_level=2)
qsearch_workflow = [
    QuickPartitioner(3), 
    ForEachBlockPass(
        QSearchSynthesisPass(instantiate_options=instantiate_options),
        # ScanningGateRemovalPass(instantiate_options=instantiate_options)
    ),
    UnfoldPass(),
]

# Load a circuit from QASM
circuit = Circuit.from_file(qasm_file_path)
compiler = Compiler(runtime_log_level=10, log_file='log.txt', num_workers=2)

out_circuit = compiler.compile(circuit, workflow=qsearch_workflow,)

compiler.close()
# Print new circuit statistics
print("Original Circuit Statistics:")
print("  Gates:", circuit.gate_set)
print("  Gate Counts:", circuit.gate_counts)
print("  Connectivity:", circuit.coupling_graph)

print("Compiled Circuit Statistics:")
print("  Gates:", out_circuit.gate_set)
print("  Gate Counts:", out_circuit.gate_counts)
print("  Connectivity:", out_circuit.coupling_graph)
# Save the output as qasm
#out_circuit.save('tfim_out.qasm')
