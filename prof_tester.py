"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.passes import *
from bqskit.compiler import Compiler
from bqskit import compile
import sys
import time
import argparse
from timeit import default_timer as timer

# Construct the unitary as an NumPy array
circ = Circuit.from_file("hubbard_4.qasm")
circ.remove_all_measurements()

print(circ.gate_counts)

parser = argparse.ArgumentParser(description="Process some integers.")
# required arguments
parser.add_argument('task', type=str, help='The task to perform')
parser.add_argument('num_workers', type=int, help='Number of workers')
# optional 
parser.add_argument('--block_size', type=int, default=3, help='Size of the block (default: 3)')

args = parser.parse_args()

task = args.task
num_workers = args.num_workers
block_size = args.block_size
# The compile function will perform synthesis

# Create a controlled workflow. 

# Model: Log to Timeline
start_time = timer()


if task == "default":
    compiler = Compiler(num_workers=num_workers, log_file=f"logs/{task}_{num_workers}.txt")
    synthesized_circuit = compile(circ, optimization_level=3, compiler=compiler)
else:
    compiler = Compiler(num_workers=num_workers, log_file=f"logs/{task}_{block_size}_{num_workers}.txt")

    if task == "leap":
        workflow = [
            ScanPartitioner(block_size),
            ForEachBlockPass([
                LEAPSynthesisPass(),  # LEAP performs native gate instantiation
            ]),
            UnfoldPass(),
        ]
    elif task == "scan":
        workflow = [
            ScanPartitioner(block_size),
            ForEachBlockPass([
                ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
            ]),
            UnfoldPass(),
        ]
    elif task == "qsearch":
        workflow = [
            ScanPartitioner(block_size),
            ForEachBlockPass([
                QSearchSynthesisPass(),  # QSearch Synthesis Pass
            ]),
            UnfoldPass(),
        ]
    elif task == "pas":
        workflow = [
            ScanPartitioner(block_size),
            ForEachBlockPass([
                PermutationAwareSynthesisPass(),
            ]),
            UnfoldPass(),
        ]

    synthesized_circuit = compiler.compile(circ, workflow=workflow)

print(synthesized_circuit.gate_counts)
print("Total Time:", timer() - start_time)