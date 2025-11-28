#!/usr/bin/env python3
"""Quick test of the Genetic Quantum Life Factory"""

import sys
sys.path.insert(0, '/home/user/Quantum')

from algorithm import *

print("Testing Genetic Quantum Life Factory...")
print("=" * 80)

# Create a simple MaxCut problem
problem = MaxCutProblem(
    num_vertices=4,
    edges=[(0,1), (1,2), (2,3), (3,0)],
    weights=[1.0, 1.0, 1.0, 1.0]
)

print("Creating Evolutionary QAOA...")
evolve_qaoa = EvolutionaryQAOA(
    problem=problem,
    p=2,
    population_size=15,
    use_evolution=True
)

print("Evolving for 20 generations...")
best_params, best_energy = evolve_qaoa.evolve_optimize(
    num_generations=20,
    elite_ratio=0.2,
    mutation_rate=0.15,
    crossover_rate=0.7,
    verbose=True
)

print("\nGetting best solution...")
best_bitstring, best_cut = evolve_qaoa.get_best_solution(num_shots=500)

print("\n" + "=" * 80)
print("RESULTS:")
print(f"  Best energy: {best_energy:.4f}")
print(f"  Best cut value: {best_cut}")
print(f"  Best bitstring: {format(best_bitstring, '04b')}")
print(f"  Total evaluations: {evolve_qaoa.population.total_evaluations}")
print("=" * 80)

print("\n✓ Genetic Quantum Life Factory test PASSED!")
