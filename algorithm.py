"""
GENETIC QUANTUM LIFE FACTORY
============================

An evolutionary quantum computing framework that creates, evolves, and optimizes
quantum circuits using genetic algorithms. This system treats quantum circuits as
living organisms that compete, mutate, reproduce, and evolve to solve complex
optimization problems.

Key Components:
1. Quantum State & Circuit Infrastructure
2. Multiple Optimization Problems (Max-Cut, TSP, Knapsack, Graph Coloring)
3. Quantum Organisms (genetic encoding of circuits)
4. Genetic Operators (mutation, crossover, selection)
5. Population-based Evolution
6. Adaptive QAOA with evolutionary strategies
7. Fitness Landscapes & Diversity Metrics
8. Multi-objective Optimization
9. Life Factory Orchestration
10. Advanced Evolutionary Visualization
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.optimize import minimize
import networkx as nx
from copy import deepcopy
from enum import Enum
import random
from collections import defaultdict


# ============================================================================
# QUANTUM STATE REPRESENTATION
# ============================================================================

class QuantumState:
    """
    Represents a quantum state vector in the computational basis.
    For n qubits, maintains a 2^n dimensional complex vector.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[0] = 1.0
    
    def set_state(self, state_vector: np.ndarray):
        """Set the quantum state vector directly."""
        if len(state_vector) != self.dim:
            raise ValueError(f"State vector must have dimension {self.dim}")
        # Normalize
        norm = np.linalg.norm(state_vector)
        self.state_vector = state_vector / norm
    
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution over computational basis states."""
        return np.abs(self.state_vector) ** 2
    
    def measure(self, num_shots: int = 1) -> List[int]:
        """
        Perform measurements on the quantum state.
        Returns list of measurement outcomes (integers representing bit strings).
        """
        probabilities = self.get_probabilities()
        measurements = np.random.choice(
            self.dim, 
            size=num_shots, 
            p=probabilities
        )
        return measurements.tolist()
    
    def expectation_value(self, operator: np.ndarray) -> float:
        """Calculate expectation value ⟨ψ|O|ψ⟩ for operator O."""
        return np.real(
            np.conj(self.state_vector) @ operator @ self.state_vector
        )


# ============================================================================
# QUANTUM GATES
# ============================================================================

class QuantumGates:
    """
    Collection of quantum gate operations represented as unitary matrices.
    """
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X gate (NOT gate)."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate - creates superposition."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """Rotation around X-axis by angle theta."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis by angle theta."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis by angle theta."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def cz() -> np.ndarray:
        """Controlled-Z gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)


# ============================================================================
# QUANTUM CIRCUIT
# ============================================================================

class QuantumCircuit:
    """
    Quantum circuit that applies gates to quantum states.
    Supports single-qubit and two-qubit gates.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = QuantumGates()
    
    def _single_qubit_operator(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """
        Construct the full operator for a single-qubit gate acting on specified qubit.
        Uses tensor product with identity matrices.
        """
        identity = np.eye(2, dtype=complex)
        
        # Build operator: I ⊗ I ⊗ ... ⊗ gate ⊗ ... ⊗ I
        operator = np.array([1.0], dtype=complex)
        for i in range(self.num_qubits):
            if i == qubit:
                operator = np.kron(operator, gate)
            else:
                operator = np.kron(operator, identity)
        
        return operator
    
    def _two_qubit_operator(self, gate: np.ndarray, control: int, target: int) -> np.ndarray:
        """
        Construct operator for two-qubit gate.
        More complex than single-qubit case due to non-adjacent qubits.
        """
        if control == target:
            raise ValueError("Control and target must be different qubits")
        
        # Ensure control < target for consistency
        if control > target:
            control, target = target, control
        
        # For CNOT and CZ gates, we need to carefully construct the operator
        identity = np.eye(2, dtype=complex)
        dim = 2 ** self.num_qubits
        operator = np.zeros((dim, dim), dtype=complex)
        
        # Iterate through all basis states
        for i in range(dim):
            # Get binary representation
            binary = format(i, f'0{self.num_qubits}b')
            bits = [int(b) for b in binary]
            
            # Apply gate logic
            control_bit = bits[control]
            target_bit = bits[target]
            
            if gate.shape == (4, 4):  # Two-qubit gate
                # Extract 2x2 block based on control bit
                if control_bit == 0:
                    # Don't flip target
                    operator[i, i] = 1.0
                else:
                    # Apply gate operation
                    new_bits = bits.copy()
                    if np.allclose(gate, self.gates.cnot()):
                        # CNOT: flip target
                        new_bits[target] = 1 - target_bit
                    elif np.allclose(gate, self.gates.cz()):
                        # CZ: phase flip if target is 1
                        if target_bit == 1:
                            operator[i, i] = -1.0
                        else:
                            operator[i, i] = 1.0
                        continue
                    
                    # Convert back to index
                    j = int(''.join(map(str, new_bits)), 2)
                    operator[j, i] = 1.0
        
        return operator
    
    def apply_hadamard(self, state: QuantumState, qubit: int):
        """Apply Hadamard gate to specified qubit."""
        h_op = self._single_qubit_operator(self.gates.hadamard(), qubit)
        state.state_vector = h_op @ state.state_vector
    
    def apply_rx(self, state: QuantumState, qubit: int, theta: float):
        """Apply RX rotation to specified qubit."""
        rx_op = self._single_qubit_operator(self.gates.rx(theta), qubit)
        state.state_vector = rx_op @ state.state_vector
    
    def apply_ry(self, state: QuantumState, qubit: int, theta: float):
        """Apply RY rotation to specified qubit."""
        ry_op = self._single_qubit_operator(self.gates.ry(theta), qubit)
        state.state_vector = ry_op @ state.state_vector
    
    def apply_rz(self, state: QuantumState, qubit: int, theta: float):
        """Apply RZ rotation to specified qubit."""
        rz_op = self._single_qubit_operator(self.gates.rz(theta), qubit)
        state.state_vector = rz_op @ state.state_vector
    
    def apply_cnot(self, state: QuantumState, control: int, target: int):
        """Apply CNOT gate with specified control and target qubits."""
        cnot_op = self._two_qubit_operator(self.gates.cnot(), control, target)
        state.state_vector = cnot_op @ state.state_vector
    
    def apply_cz(self, state: QuantumState, control: int, target: int):
        """Apply CZ gate with specified control and target qubits."""
        cz_op = self._two_qubit_operator(self.gates.cz(), control, target)
        state.state_vector = cz_op @ state.state_vector


# ============================================================================
# MAX-CUT PROBLEM DEFINITION
# ============================================================================

@dataclass
class MaxCutProblem:
    """
    Defines a Max-Cut problem instance on a graph.
    
    Max-Cut: Given a graph G=(V,E), partition vertices into two sets S and S'
    such that the number of edges between S and S' is maximized.
    """
    num_vertices: int
    edges: List[Tuple[int, int]]
    weights: List[float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.edges)
        
        if len(self.weights) != len(self.edges):
            raise ValueError("Number of weights must match number of edges")
    
    def evaluate_cut(self, bitstring: int) -> float:
        """
        Evaluate the cut value for a given partition.
        bitstring: integer whose binary representation indicates partition
                  (bit i = 0 means vertex i in S, bit i = 1 means vertex i in S')
        """
        # Convert integer to binary array
        bits = [(bitstring >> i) & 1 for i in range(self.num_vertices)]
        
        cut_value = 0.0
        for (i, j), weight in zip(self.edges, self.weights):
            # Edge contributes if vertices are in different partitions
            if bits[i] != bits[j]:
                cut_value += weight
        
        return cut_value
    
    def get_cost_hamiltonian(self) -> np.ndarray:
        """
        Construct the cost Hamiltonian H_C for the Max-Cut problem.
        
        H_C = -Σ_{(i,j) ∈ E} w_{ij} * (1 - Z_i Z_j) / 2
        
        Where Z_i is the Pauli-Z operator on qubit i.
        The ground state of H_C corresponds to the maximum cut.
        """
        dim = 2 ** self.num_vertices
        hamiltonian = np.zeros((dim, dim), dtype=complex)
        
        # For each edge, add contribution to Hamiltonian
        for (i, j), weight in zip(self.edges, self.weights):
            # Construct Z_i ⊗ Z_j operator
            zz_op = np.array([1.0], dtype=complex)
            for k in range(self.num_vertices):
                if k == i or k == j:
                    zz_op = np.kron(zz_op, np.array([[1, 0], [0, -1]], dtype=complex))
                else:
                    zz_op = np.kron(zz_op, np.eye(2, dtype=complex))
            
            # Add to Hamiltonian: -w * (I - Z_i Z_j) / 2
            hamiltonian += -weight * (np.eye(dim, dtype=complex) - zz_op) / 2
        
        return hamiltonian


# ============================================================================
# ADDITIONAL OPTIMIZATION PROBLEMS
# ============================================================================

@dataclass
class TSPProblem:
    """
    Traveling Salesman Problem: Find shortest route visiting all cities.
    """
    num_cities: int
    distance_matrix: np.ndarray

    def __post_init__(self):
        if self.distance_matrix.shape != (self.num_cities, self.num_cities):
            raise ValueError("Distance matrix must be square")

    def evaluate_route(self, route: List[int]) -> float:
        """Calculate total distance for a route."""
        total_distance = 0.0
        for i in range(len(route)):
            city_a = route[i]
            city_b = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[city_a, city_b]
        return total_distance

    def get_cost_hamiltonian(self, num_qubits: int) -> np.ndarray:
        """Construct TSP cost Hamiltonian."""
        dim = 2 ** num_qubits
        hamiltonian = np.zeros((dim, dim), dtype=complex)

        # Simplified TSP Hamiltonian encoding
        for i in range(dim):
            # Convert bitstring to potential route
            bits = [(i >> j) & 1 for j in range(num_qubits)]
            route = [j for j, bit in enumerate(bits[:self.num_cities]) if bit]

            if len(route) == self.num_cities and len(set(route)) == self.num_cities:
                cost = self.evaluate_route(route)
                hamiltonian[i, i] = cost
            else:
                hamiltonian[i, i] = 1e6  # Penalty for invalid routes

        return hamiltonian


@dataclass
class KnapsackProblem:
    """
    Knapsack Problem: Maximize value while staying within weight capacity.
    """
    num_items: int
    values: List[float]
    weights: List[float]
    capacity: float

    def __post_init__(self):
        if len(self.values) != self.num_items or len(self.weights) != self.num_items:
            raise ValueError("Values and weights must match number of items")

    def evaluate_selection(self, bitstring: int) -> Tuple[float, float]:
        """
        Evaluate a selection of items.
        Returns (total_value, total_weight).
        """
        bits = [(bitstring >> i) & 1 for i in range(self.num_items)]
        total_value = sum(v for i, v in enumerate(self.values) if bits[i])
        total_weight = sum(w for i, w in enumerate(self.weights) if bits[i])
        return total_value, total_weight

    def get_fitness(self, bitstring: int) -> float:
        """Get fitness (negative for minimization)."""
        value, weight = self.evaluate_selection(bitstring)
        if weight > self.capacity:
            return -1e6  # Heavy penalty for exceeding capacity
        return value  # Maximize value

    def get_cost_hamiltonian(self) -> np.ndarray:
        """Construct Knapsack cost Hamiltonian."""
        dim = 2 ** self.num_items
        hamiltonian = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            fitness = self.get_fitness(i)
            hamiltonian[i, i] = -fitness  # Negative because we minimize energy

        return hamiltonian


@dataclass
class GraphColoringProblem:
    """
    Graph Coloring Problem: Color vertices such that no adjacent vertices share a color.
    """
    num_vertices: int
    edges: List[Tuple[int, int]]
    num_colors: int

    def evaluate_coloring(self, coloring: List[int]) -> int:
        """
        Count number of edges with endpoints of the same color (conflicts).
        """
        conflicts = 0
        for i, j in self.edges:
            if i < len(coloring) and j < len(coloring):
                if coloring[i] == coloring[j]:
                    conflicts += 1
        return conflicts

    def bitstring_to_coloring(self, bitstring: int) -> List[int]:
        """Convert bitstring to color assignment."""
        # Use log2(num_colors) bits per vertex
        bits_per_vertex = max(1, int(np.ceil(np.log2(self.num_colors))))
        coloring = []

        for v in range(self.num_vertices):
            color_bits = 0
            for b in range(bits_per_vertex):
                bit_pos = v * bits_per_vertex + b
                if bit_pos < 32:  # Limit for int size
                    color_bits |= ((bitstring >> bit_pos) & 1) << b
            coloring.append(color_bits % self.num_colors)

        return coloring

    def get_cost_hamiltonian(self, num_qubits: int) -> np.ndarray:
        """Construct Graph Coloring cost Hamiltonian."""
        dim = 2 ** num_qubits
        hamiltonian = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            coloring = self.bitstring_to_coloring(i)
            conflicts = self.evaluate_coloring(coloring)
            hamiltonian[i, i] = conflicts

        return hamiltonian


# ============================================================================
# QUANTUM ORGANISM - GENETIC ENCODING
# ============================================================================

class QuantumOrganism:
    """
    A quantum organism represents an individual in the evolutionary population.
    Each organism has:
    - Genes: QAOA parameters (gammas, betas)
    - Phenotype: The quantum circuit/state produced
    - Fitness: Performance on optimization problem
    - Age: Number of generations survived
    - Lineage: Ancestry tracking
    """

    def __init__(self,
                 num_qubits: int,
                 p: int,
                 genes: Optional[np.ndarray] = None,
                 organism_id: Optional[str] = None):
        self.num_qubits = num_qubits
        self.p = p
        self.age = 0
        self.generation_born = 0
        self.organism_id = organism_id or self._generate_id()
        self.parent_ids: List[str] = []

        # Genetic encoding: [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
        if genes is None:
            self.genes = np.random.uniform(0, 2*np.pi, 2*p)
        else:
            self.genes = genes.copy()

        # Fitness tracking
        self.fitness = None
        self.fitness_history: List[float] = []
        self.evaluations = 0

        # Phenotype cache
        self._phenotype_cache = None

    @staticmethod
    def _generate_id() -> str:
        """Generate unique organism ID."""
        return f"QO-{random.randint(100000, 999999)}"

    def get_gammas(self) -> np.ndarray:
        """Extract gamma parameters."""
        return self.genes[:self.p]

    def get_betas(self) -> np.ndarray:
        """Extract beta parameters."""
        return self.genes[self.p:]

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3):
        """
        Mutate organism's genes.
        mutation_rate: Probability of mutating each gene
        mutation_strength: Standard deviation of mutation
        """
        mutated = False
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                # Add Gaussian noise
                mutation = np.random.normal(0, mutation_strength)
                self.genes[i] = (self.genes[i] + mutation) % (2 * np.pi)
                mutated = True

        if mutated:
            self.fitness = None  # Reset fitness after mutation
            self._phenotype_cache = None  # Invalidate cache

    def clone(self) -> 'QuantumOrganism':
        """Create a genetic clone."""
        clone = QuantumOrganism(
            num_qubits=self.num_qubits,
            p=self.p,
            genes=self.genes.copy()
        )
        clone.parent_ids = [self.organism_id]
        clone.fitness = self.fitness  # Preserve fitness
        clone.fitness_history = self.fitness_history.copy()
        clone.evaluations = self.evaluations
        return clone

    def age_one_generation(self):
        """Increment age counter."""
        self.age += 1

    def update_fitness(self, fitness: float):
        """Update fitness value."""
        self.fitness = fitness
        self.fitness_history.append(fitness)
        self.evaluations += 1

    def get_diversity_signature(self) -> np.ndarray:
        """Get signature for diversity calculation."""
        return self.genes.copy()

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"QuantumOrganism({self.organism_id}, age={self.age}, fitness={fitness_str})"


# ============================================================================
# GENETIC OPERATORS
# ============================================================================

class GeneticOperators:
    """
    Collection of genetic operators for quantum organism evolution.
    """

    @staticmethod
    def uniform_crossover(parent1: QuantumOrganism,
                         parent2: QuantumOrganism,
                         crossover_rate: float = 0.5) -> QuantumOrganism:
        """
        Uniform crossover: each gene randomly selected from either parent.
        """
        child_genes = np.zeros(len(parent1.genes))

        for i in range(len(child_genes)):
            if random.random() < crossover_rate:
                child_genes[i] = parent1.genes[i]
            else:
                child_genes[i] = parent2.genes[i]

        child = QuantumOrganism(
            num_qubits=parent1.num_qubits,
            p=parent1.p,
            genes=child_genes
        )
        child.parent_ids = [parent1.organism_id, parent2.organism_id]
        return child

    @staticmethod
    def single_point_crossover(parent1: QuantumOrganism,
                              parent2: QuantumOrganism) -> QuantumOrganism:
        """
        Single-point crossover: split at random point and combine.
        """
        crossover_point = random.randint(1, len(parent1.genes) - 1)

        child_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])

        child = QuantumOrganism(
            num_qubits=parent1.num_qubits,
            p=parent1.p,
            genes=child_genes
        )
        child.parent_ids = [parent1.organism_id, parent2.organism_id]
        return child

    @staticmethod
    def arithmetic_crossover(parent1: QuantumOrganism,
                           parent2: QuantumOrganism,
                           alpha: float = 0.5) -> QuantumOrganism:
        """
        Arithmetic crossover: weighted average of parent genes.
        """
        child_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes

        child = QuantumOrganism(
            num_qubits=parent1.num_qubits,
            p=parent1.p,
            genes=child_genes
        )
        child.parent_ids = [parent1.organism_id, parent2.organism_id]
        return child

    @staticmethod
    def tournament_selection(population: List[QuantumOrganism],
                           tournament_size: int = 3) -> QuantumOrganism:
        """
        Tournament selection: randomly select k individuals, return best.
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda org: org.fitness if org.fitness is not None else -float('inf'))

    @staticmethod
    def roulette_selection(population: List[QuantumOrganism]) -> QuantumOrganism:
        """
        Roulette wheel selection: probability proportional to fitness.
        """
        # Handle negative fitness values
        fitnesses = np.array([org.fitness if org.fitness is not None else -1e10
                             for org in population])
        min_fitness = fitnesses.min()
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6

        total_fitness = fitnesses.sum()
        if total_fitness <= 0:
            return random.choice(population)

        probabilities = fitnesses / total_fitness
        return np.random.choice(population, p=probabilities)

    @staticmethod
    def elitism_selection(population: List[QuantumOrganism],
                         elite_count: int) -> List[QuantumOrganism]:
        """
        Elitism: select top N individuals.
        """
        sorted_pop = sorted(population,
                          key=lambda org: org.fitness if org.fitness is not None else -float('inf'),
                          reverse=True)
        return sorted_pop[:elite_count]


# ============================================================================
# QUANTUM POPULATION - ECOSYSTEM MANAGEMENT
# ============================================================================

class QuantumPopulation:
    """
    Manages a population of quantum organisms with evolutionary dynamics.
    """

    def __init__(self,
                 population_size: int,
                 num_qubits: int,
                 p: int,
                 problem: Any):
        self.population_size = population_size
        self.num_qubits = num_qubits
        self.p = p
        self.problem = problem

        # Initialize population
        self.organisms: List[QuantumOrganism] = []
        self._initialize_population()

        # Evolution tracking
        self.generation = 0
        self.best_organism: Optional[QuantumOrganism] = None
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

        # Statistics
        self.total_evaluations = 0
        self.genetic_operators = GeneticOperators()

    def _initialize_population(self):
        """Create initial random population."""
        for _ in range(self.population_size):
            organism = QuantumOrganism(
                num_qubits=self.num_qubits,
                p=self.p
            )
            organism.generation_born = 0
            self.organisms.append(organism)

    def evaluate_fitness(self, organism: QuantumOrganism, qaoa: 'QAOA') -> float:
        """
        Evaluate organism fitness using QAOA.
        """
        # Use organism's genes as QAOA parameters
        state = qaoa.construct_qaoa_state(organism.genes)

        # Compute expected energy (negative for maximization)
        if hasattr(self.problem, 'get_cost_hamiltonian'):
            if isinstance(self.problem, MaxCutProblem):
                cost_hamiltonian = self.problem.get_cost_hamiltonian()
                energy = state.expectation_value(cost_hamiltonian)
                fitness = -energy  # Minimize energy = maximize fitness
            elif isinstance(self.problem, KnapsackProblem):
                # Sample and evaluate
                measurements = state.measure(num_shots=100)
                total_fitness = 0
                for m in measurements:
                    total_fitness += self.problem.get_fitness(m)
                fitness = total_fitness / len(measurements)
            else:
                cost_hamiltonian = self.problem.get_cost_hamiltonian(self.num_qubits)
                energy = state.expectation_value(cost_hamiltonian)
                fitness = -energy
        else:
            fitness = random.random()  # Fallback

        organism.update_fitness(fitness)
        self.total_evaluations += 1
        return fitness

    def evaluate_all(self, qaoa: 'QAOA'):
        """Evaluate fitness for all organisms."""
        for organism in self.organisms:
            if organism.fitness is None:
                self.evaluate_fitness(organism, qaoa)

        # Update best organism
        current_best = max(self.organisms, key=lambda org: org.fitness if org.fitness is not None else -float('inf'))
        if current_best.fitness is not None:
            if self.best_organism is None or (self.best_organism.fitness is None) or current_best.fitness > self.best_organism.fitness:
                self.best_organism = current_best.clone()

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity using pairwise genetic distances.
        """
        if len(self.organisms) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(self.organisms)):
            for j in range(i + 1, len(self.organisms)):
                dist = np.linalg.norm(
                    self.organisms[i].get_diversity_signature() -
                    self.organisms[j].get_diversity_signature()
                )
                total_distance += dist
                count += 1

        return total_distance / count if count > 0 else 0.0

    def evolve_generation(self,
                         qaoa: 'QAOA',
                         elite_ratio: float = 0.1,
                         mutation_rate: float = 0.1,
                         crossover_rate: float = 0.7):
        """
        Evolve population by one generation.
        """
        # Evaluate all organisms
        self.evaluate_all(qaoa)

        # Calculate statistics
        fitnesses = [org.fitness for org in self.organisms if org.fitness is not None]
        if fitnesses:
            self.best_fitness_history.append(max(fitnesses))
            self.average_fitness_history.append(np.mean(fitnesses))

        diversity = self.calculate_diversity()
        self.diversity_history.append(diversity)

        # Selection and reproduction
        elite_count = max(1, int(self.population_size * elite_ratio))
        elites = self.genetic_operators.elitism_selection(self.organisms, elite_count)

        # Create next generation
        next_generation = [org.clone() for org in elites]

        while len(next_generation) < self.population_size:
            if random.random() < crossover_rate:
                # Crossover
                parent1 = self.genetic_operators.tournament_selection(self.organisms)
                parent2 = self.genetic_operators.tournament_selection(self.organisms)
                child = self.genetic_operators.uniform_crossover(parent1, parent2)
            else:
                # Clone and mutate
                parent = self.genetic_operators.tournament_selection(self.organisms)
                child = parent.clone()

            # Mutation
            child.mutate(mutation_rate=mutation_rate)
            child.generation_born = self.generation + 1
            next_generation.append(child)

        # Age organisms and update population
        for org in self.organisms:
            org.age_one_generation()

        self.organisms = next_generation[:self.population_size]
        self.generation += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current population statistics."""
        fitnesses = [org.fitness for org in self.organisms if org.fitness is not None]

        return {
            'generation': self.generation,
            'population_size': len(self.organisms),
            'best_fitness': max(fitnesses) if fitnesses else None,
            'average_fitness': np.mean(fitnesses) if fitnesses else None,
            'worst_fitness': min(fitnesses) if fitnesses else None,
            'diversity': self.calculate_diversity(),
            'total_evaluations': self.total_evaluations,
            'best_organism': self.best_organism
        }


# ============================================================================
# QAOA ALGORITHM
# ============================================================================

class QAOA:
    """
    Quantum Approximate Optimization Algorithm implementation.
    
    QAOA alternates between:
    1. Cost Hamiltonian evolution: exp(-i γ H_C)
    2. Mixer Hamiltonian evolution: exp(-i β H_M)
    
    where H_M = Σ_i X_i (sum of Pauli-X operators)
    
    The parameters (γ, β) are optimized classically to minimize the
    expected energy ⟨ψ|H_C|ψ⟩.
    """
    
    def __init__(self, problem: MaxCutProblem, p: int = 1):
        """
        Initialize QAOA solver.
        
        Args:
            problem: MaxCutProblem instance
            p: Number of QAOA layers (depth)
        """
        self.problem = problem
        self.p = p
        self.num_qubits = problem.num_vertices
        self.circuit = QuantumCircuit(self.num_qubits)
        
        # Store optimization history
        self.history = {
            'params': [],
            'energies': [],
            'iterations': 0
        }
        
        # Best result found
        self.best_params = None
        self.best_energy = float('inf')
        self.best_bitstring = None
    
    def _apply_cost_unitary(self, state: QuantumState, gamma: float):
        """
        Apply cost unitary exp(-i γ H_C) to the state.
        
        For Max-Cut, this decomposes into individual ZZ rotations:
        exp(-i γ H_C) = Π_{(i,j)} exp(i γ w_{ij} Z_i Z_j / 2)
        """
        for (i, j), weight in zip(self.problem.edges, self.problem.weights):
            # Implement exp(i γ w Z_i Z_j / 2) using CNOT gates
            angle = gamma * weight
            
            # Decomposition: exp(i θ Z_i Z_j) = 
            # CNOT(j,i) * RZ_i(2θ) * CNOT(j,i)
            self.circuit.apply_cnot(state, j, i)
            self.circuit.apply_rz(state, i, 2 * angle)
            self.circuit.apply_cnot(state, j, i)
    
    def _apply_mixer_unitary(self, state: QuantumState, beta: float):
        """
        Apply mixer unitary exp(-i β H_M) to the state.
        
        H_M = Σ_i X_i, so exp(-i β H_M) = Π_i exp(-i β X_i) = Π_i RX(2β)
        """
        for i in range(self.num_qubits):
            self.circuit.apply_rx(state, i, 2 * beta)
    
    def construct_qaoa_state(self, params: np.ndarray) -> QuantumState:
        """
        Construct the QAOA state |γ, β⟩ for given parameters.
        
        1. Initialize to uniform superposition: H^⊗n |0⟩
        2. Apply p layers of (cost_unitary, mixer_unitary)
        """
        # Extract gamma and beta parameters
        gammas = params[:self.p]
        betas = params[self.p:]
        
        # Initialize state
        state = QuantumState(self.num_qubits)
        
        # Create uniform superposition
        for i in range(self.num_qubits):
            self.circuit.apply_hadamard(state, i)
        
        # Apply QAOA layers
        for layer in range(self.p):
            self._apply_cost_unitary(state, gammas[layer])
            self._apply_mixer_unitary(state, betas[layer])
        
        return state
    
    def compute_expectation(self, params: np.ndarray) -> float:
        """
        Compute the expected energy ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩.
        This is the objective function we want to minimize.
        """
        state = self.construct_qaoa_state(params)
        cost_hamiltonian = self.problem.get_cost_hamiltonian()
        energy = state.expectation_value(cost_hamiltonian)
        
        # Record history
        self.history['params'].append(params.copy())
        self.history['energies'].append(energy)
        self.history['iterations'] += 1
        
        return energy
    
    def optimize(self, 
                 initial_params: np.ndarray = None,
                 method: str = 'COBYLA',
                 maxiter: int = 200) -> Tuple[np.ndarray, float]:
        """
        Optimize QAOA parameters using classical optimization.
        
        Returns:
            best_params: Optimal parameters (γ, β)
            best_energy: Minimum energy achieved
        """
        # Initialize parameters if not provided
        if initial_params is None:
            # Random initialization in [0, 2π]
            initial_params = np.random.uniform(0, 2*np.pi, 2*self.p)
        
        print(f"Starting QAOA optimization with p={self.p} layers...")
        print(f"Using {method} optimizer with max {maxiter} iterations")
        
        # Reset history
        self.history = {'params': [], 'energies': [], 'iterations': 0}
        
        # Classical optimization
        result = minimize(
            self.compute_expectation,
            initial_params,
            method=method,
            options={'maxiter': maxiter}
        )
        
        self.best_params = result.x
        self.best_energy = result.fun
        
        print(f"\nOptimization completed!")
        print(f"Final energy: {self.best_energy:.6f}")
        print(f"Iterations: {self.history['iterations']}")
        
        return self.best_params, self.best_energy
    
    def sample_solution(self, params: np.ndarray, num_shots: int = 1000) -> Dict[int, int]:
        """
        Sample measurement outcomes from the QAOA state.
        
        Returns:
            Dictionary mapping bitstrings (as integers) to counts
        """
        state = self.construct_qaoa_state(params)
        measurements = state.measure(num_shots)
        
        # Count occurrences
        counts = {}
        for measurement in measurements:
            counts[measurement] = counts.get(measurement, 0) + 1
        
        return counts
    
    def get_best_solution(self, num_shots: int = 1000) -> Tuple[int, float]:
        """
        Get the best solution found by QAOA.
        
        Returns:
            bitstring: Best partition as integer
            cut_value: Value of the cut
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        # Sample from optimized state
        counts = self.sample_solution(self.best_params, num_shots)
        
        # Evaluate all sampled solutions
        best_bitstring = None
        best_cut = float('-inf')
        
        for bitstring, count in counts.items():
            cut_value = self.problem.evaluate_cut(bitstring)
            if cut_value > best_cut:
                best_cut = cut_value
                best_bitstring = bitstring
        
        self.best_bitstring = best_bitstring
        return best_bitstring, best_cut


# ============================================================================
# FITNESS LANDSCAPE ANALYSIS
# ============================================================================

class FitnessLandscape:
    """
    Analyze and visualize the fitness landscape.
    """

    @staticmethod
    def calculate_landscape_metrics(population: QuantumPopulation) -> Dict[str, float]:
        """Calculate various landscape metrics."""
        fitnesses = [org.fitness for org in population.organisms if org.fitness is not None]

        if not fitnesses:
            return {}

        return {
            'ruggedness': np.std(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'fitness_range': np.max(fitnesses) - np.min(fitnesses),
            'coefficient_variation': np.std(fitnesses) / np.mean(fitnesses) if np.mean(fitnesses) != 0 else 0
        }

    @staticmethod
    def plot_fitness_landscape_2d(population: QuantumPopulation,
                                  param_idx1: int = 0,
                                  param_idx2: int = 1):
        """Plot 2D projection of fitness landscape."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract parameters and fitnesses
        params1 = [org.genes[param_idx1] for org in population.organisms]
        params2 = [org.genes[param_idx2] for org in population.organisms]
        fitnesses = [org.fitness if org.fitness is not None else 0
                    for org in population.organisms]

        # Create scatter plot
        scatter = ax.scatter(params1, params2, c=fitnesses,
                           cmap='viridis', s=100, alpha=0.6, edgecolors='black')

        ax.set_xlabel(f'Parameter {param_idx1} (γ or β)', fontsize=12)
        ax.set_ylabel(f'Parameter {param_idx2} (γ or β)', fontsize=12)
        ax.set_title('Fitness Landscape Projection', fontsize=14, fontweight='bold')

        plt.colorbar(scatter, ax=ax, label='Fitness')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ============================================================================
# EVOLUTIONARY QAOA
# ============================================================================

class EvolutionaryQAOA(QAOA):
    """
    Enhanced QAOA with evolutionary strategies.
    """

    def __init__(self, problem: Any, p: int = 1,
                 population_size: int = 20,
                 use_evolution: bool = True):
        super().__init__(problem, p)
        self.use_evolution = use_evolution

        if use_evolution:
            self.population = QuantumPopulation(
                population_size=population_size,
                num_qubits=self.num_qubits,
                p=p,
                problem=problem
            )

    def evolve_optimize(self,
                       num_generations: int = 50,
                       elite_ratio: float = 0.1,
                       mutation_rate: float = 0.1,
                       crossover_rate: float = 0.7,
                       verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Optimize using evolutionary algorithm instead of gradient-based methods.
        """
        if not self.use_evolution:
            return self.optimize()

        if verbose:
            print(f"Starting Evolutionary QAOA with {self.population.population_size} organisms")
            print(f"Evolving for {num_generations} generations...")
            print()

        for gen in range(num_generations):
            self.population.evolve_generation(
                qaoa=self,
                elite_ratio=elite_ratio,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate
            )

            if verbose and (gen % 10 == 0 or gen == num_generations - 1):
                stats = self.population.get_statistics()
                best_fit = stats['best_fitness'] if stats['best_fitness'] is not None else 0.0
                avg_fit = stats['average_fitness'] if stats['average_fitness'] is not None else 0.0
                diversity = stats['diversity']
                print(f"Generation {gen:3d}: "
                      f"Best={best_fit:8.4f} "
                      f"Avg={avg_fit:8.4f} "
                      f"Diversity={diversity:6.3f}")

        if verbose:
            print()
            print("Evolution complete!")

        best_org = self.population.best_organism
        if best_org is not None and best_org.fitness is not None:
            self.best_params = best_org.genes
            self.best_energy = -best_org.fitness
        else:
            # Fallback to best in population
            best_in_pop = max(self.population.organisms,
                            key=lambda org: org.fitness if org.fitness is not None else -float('inf'))
            self.best_params = best_in_pop.genes
            self.best_energy = -best_in_pop.fitness if best_in_pop.fitness is not None else 0.0

        return self.best_params, self.best_energy


# ============================================================================
# LIFE FACTORY - MAIN ORCHESTRATOR
# ============================================================================

class QuantumLifeFactory:
    """
    The Life Factory: Creates, evolves, and manages quantum life.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.experiments: List[Dict[str, Any]] = []
        self.best_organisms_hall_of_fame: List[QuantumOrganism] = []

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration for the factory."""
        return {
            'population_size': 30,
            'num_generations': 50,
            'p': 2,
            'elite_ratio': 0.15,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'tournament_size': 3
        }

    def create_problem(self, problem_type: str, **kwargs) -> Any:
        """
        Factory method to create various problem types.
        """
        if problem_type == 'maxcut':
            return MaxCutProblem(**kwargs)
        elif problem_type == 'tsp':
            return TSPProblem(**kwargs)
        elif problem_type == 'knapsack':
            return KnapsackProblem(**kwargs)
        elif problem_type == 'graph_coloring':
            return GraphColoringProblem(**kwargs)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def evolve_solution(self,
                       problem: Any,
                       num_qubits: int,
                       experiment_name: str = "experiment",
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Main method: Evolve quantum organisms to solve a problem.
        """
        if verbose:
            print("=" * 80)
            print("QUANTUM LIFE FACTORY")
            print(f"Experiment: {experiment_name}")
            print("=" * 80)
            print()

        # Create evolutionary QAOA
        evolve_qaoa = EvolutionaryQAOA(
            problem=problem,
            p=self.config['p'],
            population_size=self.config['population_size'],
            use_evolution=True
        )

        # Evolve
        best_params, best_energy = evolve_qaoa.evolve_optimize(
            num_generations=self.config['num_generations'],
            elite_ratio=self.config['elite_ratio'],
            mutation_rate=self.config['mutation_rate'],
            crossover_rate=self.config['crossover_rate'],
            verbose=verbose
        )

        # Get best organism
        best_organism = evolve_qaoa.population.best_organism
        self.best_organisms_hall_of_fame.append(best_organism)

        # Sample solution
        best_bitstring, best_value = evolve_qaoa.get_best_solution(num_shots=1000)

        # Record experiment
        experiment_result = {
            'name': experiment_name,
            'problem': problem,
            'num_qubits': num_qubits,
            'best_organism': best_organism,
            'best_params': best_params,
            'best_energy': best_energy,
            'best_bitstring': best_bitstring,
            'best_value': best_value,
            'population': evolve_qaoa.population,
            'qaoa': evolve_qaoa
        }

        self.experiments.append(experiment_result)

        if verbose:
            print()
            print("=" * 80)
            print(f"EXPERIMENT COMPLETE: {experiment_name}")
            print(f"Best fitness: {best_organism.fitness:.4f}")
            print(f"Best organism: {best_organism}")
            print("=" * 80)
            print()

        return experiment_result

    def run_multi_problem_evolution(self, verbose: bool = True):
        """
        Run evolution on multiple problem types simultaneously.
        """
        if verbose:
            print("\n" + "=" * 80)
            print("MULTI-PROBLEM QUANTUM LIFE FACTORY")
            print("Creating and evolving quantum organisms across problem domains")
            print("=" * 80 + "\n")

        results = []

        # Max-Cut Problem
        maxcut_problem = self.create_problem(
            'maxcut',
            num_vertices=5,
            edges=[(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3), (2,4)],
            weights=[1.0]*8
        )
        result1 = self.evolve_solution(
            maxcut_problem,
            num_qubits=5,
            experiment_name="MaxCut Pentagon",
            verbose=verbose
        )
        results.append(result1)

        # Knapsack Problem
        knapsack_problem = self.create_problem(
            'knapsack',
            num_items=5,
            values=[10.0, 15.0, 20.0, 25.0, 30.0],
            weights=[5.0, 7.0, 10.0, 12.0, 15.0],
            capacity=30.0
        )
        result2 = self.evolve_solution(
            knapsack_problem,
            num_qubits=5,
            experiment_name="Knapsack Optimization",
            verbose=verbose
        )
        results.append(result2)

        # Graph Coloring Problem
        coloring_problem = self.create_problem(
            'graph_coloring',
            num_vertices=4,
            edges=[(0,1), (1,2), (2,3), (3,0), (0,2)],
            num_colors=3
        )
        result3 = self.evolve_solution(
            coloring_problem,
            num_qubits=4,
            experiment_name="Graph 3-Coloring",
            verbose=verbose
        )
        results.append(result3)

        return results

    def visualize_all_experiments(self, save_path: str = '/mnt/user-data/outputs/'):
        """
        Create comprehensive visualizations for all experiments.
        """
        print("\nGenerating comprehensive visualizations...")

        for i, exp in enumerate(self.experiments):
            prefix = save_path + f"exp{i}_{exp['name'].replace(' ', '_')}_"

            # Evolution progress
            fig1 = EvolutionaryVisualizer.plot_evolution_progress(exp['population'])
            fig1.savefig(prefix + 'evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig1)

            # Diversity
            fig2 = EvolutionaryVisualizer.plot_diversity_over_time(exp['population'])
            fig2.savefig(prefix + 'diversity.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)

            # Fitness landscape (if applicable)
            if exp['num_qubits'] <= 6:
                fig3 = FitnessLandscape.plot_fitness_landscape_2d(exp['population'])
                fig3.savefig(prefix + 'landscape.png', dpi=150, bbox_inches='tight')
                plt.close(fig3)

        print(f"✓ Saved {len(self.experiments) * 3} visualization files")


# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

class EvolutionaryVisualizer:
    """Visualization tools for evolutionary dynamics."""

    @staticmethod
    def plot_evolution_progress(population: QuantumPopulation):
        """Plot fitness evolution over generations."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        generations = range(len(population.best_fitness_history))

        # Best and average fitness
        ax1.plot(generations, population.best_fitness_history,
                'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
        ax1.plot(generations, population.average_fitness_history,
                'r--', linewidth=2, label='Average Fitness', marker='s', markersize=3)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Evolutionary Progress', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Fitness improvement rate
        if len(population.best_fitness_history) > 1:
            improvements = np.diff(population.best_fitness_history)
            ax2.plot(range(1, len(improvements) + 1), improvements,
                    'g-', linewidth=2, marker='d', markersize=4)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Generation', fontsize=12)
            ax2.set_ylabel('Fitness Improvement', fontsize=12)
            ax2.set_title('Generation-to-Generation Improvement', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_diversity_over_time(population: QuantumPopulation):
        """Plot population diversity over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        generations = range(len(population.diversity_history))
        ax.plot(generations, population.diversity_history,
               'purple', linewidth=2, marker='o', markersize=5)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Population Diversity', fontsize=12)
        ax.set_title('Genetic Diversity Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_organism_lineage(organism: QuantumOrganism):
        """Visualize organism lineage tree."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.text(0.5, 0.9, f"Organism: {organism.organism_id}",
               ha='center', va='top', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.8, f"Age: {organism.age} generations",
               ha='center', va='top', fontsize=12)
        ax.text(0.5, 0.7, f"Fitness: {organism.fitness:.4f}",
               ha='center', va='top', fontsize=12)
        ax.text(0.5, 0.6, f"Parents: {organism.parent_ids}",
               ha='center', va='top', fontsize=10)

        ax.axis('off')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_population_snapshot(population: QuantumPopulation):
        """Create comprehensive population snapshot."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Fitness distribution
        ax1 = fig.add_subplot(gs[0, 0])
        fitnesses = [org.fitness for org in population.organisms if org.fitness is not None]
        ax1.hist(fitnesses, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Fitness', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Current Fitness Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Age distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ages = [org.age for org in population.organisms]
        ax2.hist(ages, bins=range(max(ages) + 2), color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Age (generations)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Age Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Evolution progress
        ax3 = fig.add_subplot(gs[1, :])
        gens = range(len(population.best_fitness_history))
        ax3.plot(gens, population.best_fitness_history, 'b-', linewidth=2, label='Best')
        ax3.plot(gens, population.average_fitness_history, 'r--', linewidth=2, label='Average')
        ax3.set_xlabel('Generation', fontsize=11)
        ax3.set_ylabel('Fitness', fontsize=11)
        ax3.set_title('Evolution Progress', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Diversity
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(range(len(population.diversity_history)), population.diversity_history,
                'green', linewidth=2)
        ax4.set_xlabel('Generation', fontsize=11)
        ax4.set_ylabel('Diversity', fontsize=11)
        ax4.set_title('Population Diversity', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Population Snapshot - Generation {population.generation}',
                    fontsize=16, fontweight='bold', y=0.995)

        return fig


class QAOAVisualizer:
    """Tools for visualizing QAOA results."""
    
    @staticmethod
    def plot_optimization_history(qaoa: QAOA):
        """Plot the optimization trajectory."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(qaoa.history['energies']) + 1)
        energies = qaoa.history['energies']
        
        ax.plot(iterations, energies, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Expected Energy', fontsize=12)
        ax.set_title('QAOA Optimization Progress', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best point
        best_idx = np.argmin(energies)
        ax.plot(best_idx + 1, energies[best_idx], 'r*', markersize=15, 
                label=f'Best: {energies[best_idx]:.4f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_solution(problem: MaxCutProblem, bitstring: int):
        """Visualize the graph and the cut solution."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(problem.num_vertices))
        for (i, j), weight in zip(problem.edges, problem.weights):
            G.add_edge(i, j, weight=weight)
        
        # Partition nodes based on bitstring
        bits = [(bitstring >> i) & 1 for i in range(problem.num_vertices)]
        partition_0 = [i for i in range(problem.num_vertices) if bits[i] == 0]
        partition_1 = [i for i in range(problem.num_vertices) if bits[i] == 1]
        
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=partition_0, 
                              node_color='lightblue', node_size=700, 
                              label='Partition 0', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=partition_1, 
                              node_color='lightcoral', node_size=700,
                              label='Partition 1', ax=ax)
        
        # Draw edges
        cut_edges = [(i, j) for (i, j) in problem.edges if bits[i] != bits[j]]
        non_cut_edges = [(i, j) for (i, j) in problem.edges if bits[i] == bits[j]]
        
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, 
                              edge_color='red', width=3, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=non_cut_edges, 
                              edge_color='gray', width=1, ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        ax.set_title(f'Max-Cut Solution\nCut Value: {problem.evaluate_cut(bitstring):.1f}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_measurement_distribution(counts: Dict[int, int], num_qubits: int):
        """Plot the measurement outcome distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by bitstring
        sorted_items = sorted(counts.items())
        bitstrings = [format(bs, f'0{num_qubits}b') for bs, _ in sorted_items]
        values = [count for _, count in sorted_items]
        
        ax.bar(range(len(bitstrings)), values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Bitstring', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('QAOA Measurement Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bitstrings)))
        ax.set_xticklabels(bitstrings, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def create_example_graph(graph_type: str = 'triangle') -> MaxCutProblem:
    """Create example graphs for testing."""
    
    if graph_type == 'triangle':
        # Simple triangle graph
        return MaxCutProblem(
            num_vertices=3,
            edges=[(0, 1), (1, 2), (2, 0)],
            weights=[1.0, 1.0, 1.0]
        )
    
    elif graph_type == 'square':
        # Square graph
        return MaxCutProblem(
            num_vertices=4,
            edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
            weights=[1.0, 1.0, 1.0, 1.0]
        )
    
    elif graph_type == 'complete4':
        # Complete graph on 4 vertices
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                edges.append((i, j))
        return MaxCutProblem(
            num_vertices=4,
            edges=edges,
            weights=[1.0] * len(edges)
        )
    
    elif graph_type == 'weighted':
        # Weighted graph
        return MaxCutProblem(
            num_vertices=4,
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            weights=[2.0, 1.5, 3.0, 1.0, 2.5]
        )
    
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def run_qaoa_example():
    """Complete example demonstrating QAOA algorithm."""
    
    print("="*70)
    print("QUANTUM APPROXIMATE OPTIMIZATION ALGORITHM (QAOA)")
    print("Max-Cut Problem Solver")
    print("="*70)
    print()
    
    # Create problem instance
    print("Creating Max-Cut problem instance...")
    problem = create_example_graph('complete4')
    print(f"Graph: {problem.num_vertices} vertices, {len(problem.edges)} edges")
    print(f"Edges: {problem.edges}")
    print()
    
    # Brute force solution for comparison
    print("Computing optimal solution via brute force...")
    best_cut = 0
    best_partition = 0
    for i in range(2 ** problem.num_vertices):
        cut_value = problem.evaluate_cut(i)
        if cut_value > best_cut:
            best_cut = cut_value
            best_partition = i
    
    print(f"Optimal cut value: {best_cut}")
    print(f"Optimal partition: {format(best_partition, f'0{problem.num_vertices}b')}")
    print()
    
    # Initialize QAOA
    p = 2  # Number of QAOA layers
    qaoa = QAOA(problem, p=p)
    
    # Optimize
    print("-"*70)
    best_params, best_energy = qaoa.optimize(maxiter=100)
    print("-"*70)
    print()
    
    print(f"Optimized parameters:")
    print(f"  γ = {best_params[:p]}")
    print(f"  β = {best_params[p:]}")
    print()
    
    # Sample solution
    print("Sampling solutions from optimized state...")
    num_shots = 1000
    counts = qaoa.sample_solution(best_params, num_shots)
    
    print(f"\nTop 5 most frequent outcomes:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for bitstring, count in sorted_counts:
        binary = format(bitstring, f'0{problem.num_vertices}b')
        cut_value = problem.evaluate_cut(bitstring)
        print(f"  {binary}: {count:4d} counts, cut = {cut_value}")
    
    # Get best solution
    best_bitstring, best_cut_found = qaoa.get_best_solution(num_shots)
    print(f"\nBest solution found: {format(best_bitstring, f'0{problem.num_vertices}b')}")
    print(f"Cut value: {best_cut_found}")
    print(f"Approximation ratio: {best_cut_found / best_cut:.4f}")
    print()
    
    # Visualization
    print("Generating visualizations...")
    
    # Plot optimization history
    fig1 = QAOAVisualizer.plot_optimization_history(qaoa)
    fig1.savefig('/mnt/user-data/outputs/qaoa_optimization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved optimization history plot")
    
    # Plot solution
    fig2 = QAOAVisualizer.plot_solution(problem, best_bitstring)
    fig2.savefig('/mnt/user-data/outputs/qaoa_solution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved solution visualization")
    
    # Plot measurement distribution
    fig3 = QAOAVisualizer.plot_measurement_distribution(counts, problem.num_vertices)
    fig3.savefig('/mnt/user-data/outputs/qaoa_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved measurement distribution")
    
    print()
    print("="*70)
    print("QAOA execution completed successfully!")
    print("="*70)
    
    return qaoa, problem


def run_genetic_quantum_life_factory_demo():
    """
    ULTIMATE GENETIC QUANTUM LIFE FACTORY DEMONSTRATION
    """
    print("\n" + "=" * 100)
    print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                      GENETIC QUANTUM LIFE FACTORY - FULL SYSTEM DEMO                        ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝")
    print("=" * 100 + "\n")

    # Initialize the Life Factory
    factory = QuantumLifeFactory(config={
        'population_size': 40,
        'num_generations': 60,
        'p': 2,
        'elite_ratio': 0.15,
        'mutation_rate': 0.12,
        'crossover_rate': 0.75,
        'tournament_size': 4
    })

    print("Factory initialized with:")
    print(f"  Population Size: {factory.config['population_size']}")
    print(f"  Generations: {factory.config['num_generations']}")
    print(f"  QAOA Depth (p): {factory.config['p']}")
    print(f"  Elite Ratio: {factory.config['elite_ratio']}")
    print(f"  Mutation Rate: {factory.config['mutation_rate']}")
    print(f"  Crossover Rate: {factory.config['crossover_rate']}")
    print("\n" + "=" * 100 + "\n")

    # Run multi-problem evolution
    results = factory.run_multi_problem_evolution(verbose=True)

    # Generate visualizations
    factory.visualize_all_experiments()

    # Summary statistics
    print("\n" + "=" * 100)
    print("FINAL SUMMARY - HALL OF FAME")
    print("=" * 100)

    for i, exp in enumerate(factory.experiments):
        print(f"\nExperiment {i+1}: {exp['name']}")
        print(f"  Best Organism: {exp['best_organism'].organism_id}")
        print(f"  Fitness: {exp['best_organism'].fitness:.6f}")
        print(f"  Age: {exp['best_organism'].age} generations")
        print(f"  Solution Value: {exp['best_value']:.2f}")
        print(f"  Total Evaluations: {exp['population'].total_evaluations}")
        print(f"  Final Diversity: {exp['population'].diversity_history[-1]:.4f}")

    print("\n" + "=" * 100)
    print("GENETIC QUANTUM LIFE FACTORY DEMONSTRATION COMPLETE!")
    print("All quantum organisms have evolved, competed, and produced optimal solutions.")
    print("=" * 100 + "\n")

    return factory


def run_single_evolutionary_experiment():
    """
    Run a focused single-problem evolutionary experiment.
    """
    print("\n" + "=" * 80)
    print("SINGLE EVOLUTIONARY EXPERIMENT - MaxCut on Hexagon Graph")
    print("=" * 80 + "\n")

    # Create hexagon graph problem
    problem = MaxCutProblem(
        num_vertices=6,
        edges=[(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (1,4), (2,5)],
        weights=[1.0] * 9
    )

    # Create evolutionary QAOA
    evolve_qaoa = EvolutionaryQAOA(
        problem=problem,
        p=3,
        population_size=50,
        use_evolution=True
    )

    # Evolve
    best_params, best_energy = evolve_qaoa.evolve_optimize(
        num_generations=80,
        elite_ratio=0.2,
        mutation_rate=0.1,
        crossover_rate=0.8,
        verbose=True
    )

    # Get solution
    best_bitstring, best_cut = evolve_qaoa.get_best_solution(num_shots=2000)

    print(f"\nBest solution found: {format(best_bitstring, '06b')}")
    print(f"Cut value: {best_cut}")

    # Visualizations
    print("\nGenerating visualizations...")

    fig1 = EvolutionaryVisualizer.plot_evolution_progress(evolve_qaoa.population)
    fig1.savefig('/mnt/user-data/outputs/single_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2 = EvolutionaryVisualizer.plot_population_snapshot(evolve_qaoa.population)
    fig2.savefig('/mnt/user-data/outputs/single_snapshot.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3 = QAOAVisualizer.plot_solution(problem, best_bitstring)
    fig3.savefig('/mnt/user-data/outputs/single_solution.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print("✓ Saved all visualizations")

    return evolve_qaoa


def compare_traditional_vs_evolutionary():
    """
    Compare traditional QAOA optimization vs. evolutionary approach.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Traditional QAOA vs. Evolutionary QAOA")
    print("=" * 80 + "\n")

    # Create problem
    problem = create_example_graph('complete4')

    # Traditional QAOA
    print("Running Traditional QAOA...")
    traditional_qaoa = QAOA(problem, p=2)
    trad_params, trad_energy = traditional_qaoa.optimize(maxiter=150)
    trad_bitstring, trad_cut = traditional_qaoa.get_best_solution(1000)

    print(f"\nTraditional QAOA Results:")
    print(f"  Energy: {trad_energy:.6f}")
    print(f"  Cut Value: {trad_cut:.2f}")
    print(f"  Iterations: {traditional_qaoa.history['iterations']}")

    # Evolutionary QAOA
    print("\n" + "-" * 80)
    print("Running Evolutionary QAOA...")
    evolutionary_qaoa = EvolutionaryQAOA(problem, p=2, population_size=30)
    evol_params, evol_energy = evolutionary_qaoa.evolve_optimize(
        num_generations=40,
        verbose=False
    )
    evol_bitstring, evol_cut = evolutionary_qaoa.get_best_solution(1000)

    print(f"\nEvolutionary QAOA Results:")
    print(f"  Energy: {evol_energy:.6f}")
    print(f"  Cut Value: {evol_cut:.2f}")
    print(f"  Total Evaluations: {evolutionary_qaoa.population.total_evaluations}")
    print(f"  Final Diversity: {evolutionary_qaoa.population.diversity_history[-1]:.4f}")

    # Comparison
    print("\n" + "-" * 80)
    print("COMPARISON SUMMARY:")
    print(f"  Winner (by cut value): {'Evolutionary' if evol_cut > trad_cut else 'Traditional' if trad_cut > evol_cut else 'Tie'}")
    print(f"  Difference: {abs(evol_cut - trad_cut):.4f}")
    print("=" * 80 + "\n")

    return traditional_qaoa, evolutionary_qaoa


if __name__ == "__main__":
    # Run the FULL genetic quantum life factory demonstration
    factory = run_genetic_quantum_life_factory_demo()

    # Additional experiments (uncomment to run):
    # single_exp = run_single_evolutionary_experiment()
    # trad, evol = compare_traditional_vs_evolutionary()
