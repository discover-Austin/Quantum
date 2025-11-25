"""
Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut Problem
=====================================================================

This implementation provides a complete quantum algorithm using QAOA to solve
the Maximum Cut problem on graphs. QAOA is a hybrid quantum-classical algorithm
that uses variational principles to find approximate solutions to combinatorial
optimization problems.

Key Components:
1. Quantum circuit construction with parameterized gates
2. Cost Hamiltonian for Max-Cut problem
3. Mixer Hamiltonian for quantum superposition
4. Classical optimization loop
5. Measurement and result analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize
import networkx as nx


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
# VISUALIZATION AND ANALYSIS
# ============================================================================

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


if __name__ == "__main__":
    # Run the complete example
    qaoa, problem = run_qaoa_example()
