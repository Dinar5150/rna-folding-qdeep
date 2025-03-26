# Copyright 2021 D-Wave Systems
# Based on the paper 'RNA folding using quantum computers’
# Fox DM, MacDermaid CM, Schreij AM, Zwierzyna M, Walker RC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import dirname, join
from collections import defaultdict
from itertools import product, combinations

import click
import matplotlib
import numpy as np
import networkx as nx

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

# Import the new hybrid QUBO solver API from qdeepsdk
from qdeepsdk import QDeepHybridSolver


def text_to_matrix(file_name, min_loop):
    """Reads properly formatted RNA text file and returns a matrix of possible hydrogen bonding pairs.

    Args:
        file_name (str): Path to text file.
        min_loop (int): Minimum number of nucleotides separating two sides of a stem.

    Returns:
        np.ndarray: Matrix of 0's and 1's, where 1 represents a possible bonding pair.
    """
    with open(file_name) as f:
        rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()

    index_dict = defaultdict(list)
    for i, nucleotide in enumerate(rna):
        index_dict[nucleotide].append(i)

    hydrogen_bonds = [('a', 't'), ('a', 'u'), ('c', 'g'), ('g', 't'), ('g', 'u')]
    bond_matrix = np.zeros((len(rna), len(rna)), dtype=bool)
    for pair in hydrogen_bonds:
        for bond in product(index_dict[pair[0]], index_dict[pair[1]]):
            if abs(bond[0] - bond[1]) > min_loop:
                bond_matrix[min(bond), max(bond)] = 1

    return bond_matrix


def make_stem_dict(bond_matrix, min_stem, min_loop):
    """Takes a matrix of potential hydrogen bonding pairs and returns a dictionary of possible stems.

    Args:
        bond_matrix (np.ndarray): Matrix with 0's and 1's.
        min_stem (int): Minimum number of consecutive bonds to be considered a stem.
        min_loop (int): Minimum number of nucleotides separating the two sides of a stem.

    Returns:
        dict: Dictionary with maximal stems as keys mapping to lists of weakly contained substems.
    """
    stem_dict = {}
    n = bond_matrix.shape[0]
    for i in range(n + 1 - (2 * min_stem + min_loop)):
        for j in range(i + 2 * min_stem + min_loop - 1, n):
            if bond_matrix[i, j]:
                k = 1
                while bond_matrix[i + k, j - k]:
                    bond_matrix[i + k, j - k] = False
                    k += 1
                if k >= min_stem:
                    stem_dict[(i, i + k - 1, j - k + 1, j)] = []

    for stem in stem_dict.keys():
        stem_dict[stem].extend([(stem[0] + i, stem[0] + k, stem[3] - k, stem[3] - i)
                                for i in range(stem[1] - stem[0] - min_stem + 2)
                                for k in range(i + min_stem - 1, stem[1] - stem[0] + 1)])
    return stem_dict


def check_overlap(stem1, stem2):
    """Checks if two stems use any of the same nucleotides.

    Args:
        stem1 (tuple): 4-tuple with stem information.
        stem2 (tuple): 4-tuple with stem information.

    Returns:
        bool: True if the stems overlap; False otherwise.
    """
    if isinstance(stem1, str) or isinstance(stem2, str):
        return False

    for val in stem2:
        if stem1[0] <= val <= stem1[1] or stem1[2] <= val <= stem1[3]:
            return True
    for val in stem1[1:3]:
        if stem2[0] <= val <= stem2[1] or stem2[2] <= val <= stem2[3]:
            return True
    return False


def pseudoknot_terms(stem_dict, min_stem=3, c=0.3):
    """Creates a dictionary with all possible pseudoknots as keys and appropriate penalties as values.

    Args:
        stem_dict (dict): Dictionary with maximal stems as keys mapping to lists of substems.
        min_stem (int): Minimum consecutive bonds for a stem.
        c (float): Penalty multiplier.

    Returns:
        dict: Dictionary mapping pseudoknot stem pairs to penalty values.
    """
    pseudos = {}
    for stem1, stem2 in product(stem_dict.keys(), stem_dict.keys()):
        if stem1[0] + 2 * min_stem < stem2[1] and stem1[2] + 2 * min_stem < stem2[3]:
            pseudos.update({(substem1, substem2): c * (1 + substem1[1] - substem1[0]) * (1 + substem2[1] - substem2[0])
                            for substem1, substem2
                            in product(stem_dict[stem1], stem_dict[stem2])
                            if substem1[1] < substem2[0] and substem2[1] < substem1[2] and substem1[3] < substem2[2]})
    return pseudos


def make_plot(file, stems, fig_name='RNA_plot'):
    """Produces a graph plot of the RNA folding and saves it as a PNG file.

    Args:
        file (str): Path to the RNA text file.
        stems (list): List of stems (4-tuples) in the solution.
        fig_name (str): Base name for the saved figure file.
    """
    with open(file) as f:
        rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()

    G = nx.Graph()
    rna_edges = [(i, i + 1) for i in range(len(rna) - 1)]
    stem_edges = [(stem[0] + i, stem[3] - i) for stem in stems for i in range(stem[1] - stem[0] + 1)]
    G.add_edges_from(rna_edges + stem_edges)

    color_map = []
    for node in rna:
        if node == 'g':
            color_map.append('tab:red')
        elif node == 'c':
            color_map.append('tab:green')
        elif node == 'a':
            color_map.append('y')
        else:
            color_map.append('tab:blue')

    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.8}
    pos = nx.spring_layout(G, iterations=5000)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, **options)

    labels = {i: rna[i].upper() for i in range(len(rna))}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="whitesmoke")
    nx.draw_networkx_edges(G, pos, edgelist=rna_edges, width=3.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=stem_edges, width=4.5, alpha=0.7, edge_color='tab:pink')

    plt.savefig(fig_name + '.png')
    print('\nPlot of solution saved as {}.png'.format(fig_name))


def build_qubo(stem_dict, min_stem, c):
    """Builds a QUBO matrix from the stem dictionary.

    This function creates linear coefficients that favor longer stems (using a -length² cost)
    and quadratic penalties from pseudoknot interactions and overlaps.

    Args:
        stem_dict (dict): Dictionary of maximal stems mapping to substems.
        min_stem (int): Minimum stem length.
        c (float): Pseudoknot penalty multiplier.

    Returns:
        tuple: A QUBO matrix (numpy.ndarray) and a dictionary mapping each variable (stem tuple)
               to its index in the QUBO.
    """
    # Collect all substems (variables) from the dictionary values.
    variables = set()
    for substems in stem_dict.values():
        variables.update(substems)
    variables = list(variables)
    var_index = {var: i for i, var in enumerate(variables)}
    n = len(variables)
    Q = np.zeros((n, n))

    # Linear coefficients: favor inclusion of long stems.
    for var in variables:
        length = var[1] - var[0] + 1
        Q[var_index[var], var_index[var]] += - (length ** 2)

    # Quadratic terms from pseudoknot penalties.
    quad = pseudoknot_terms(stem_dict, min_stem, c)
    for (var1, var2), coeff in quad.items():
        if var1 in var_index and var2 in var_index:
            i, j = var_index[var1], var_index[var2]
            Q[i, j] += coeff
            Q[j, i] += coeff  # Ensure symmetry

    # Add penalty for overlapping substems within the same maximal stem.
    penalty = 100  # Chosen penalty weight (tunable)
    for max_stem, substems in stem_dict.items():
        for i, var1 in enumerate(substems):
            for var2 in substems[i+1:]:
                if check_overlap(var1, var2):
                    idx1, idx2 = var_index[var1], var_index[var2]
                    Q[idx1, idx2] += penalty
                    Q[idx2, idx1] += penalty

    # Add penalty for overlaps between different maximal stems.
    for stem1, stem2 in combinations(stem_dict.keys(), 2):
        if check_overlap(stem1, stem2):
            for var1 in stem_dict[stem1]:
                for var2 in stem_dict[stem2]:
                    if check_overlap(var1, var2) and var1 in var_index and var2 in var_index:
                        idx1, idx2 = var_index[var1], var_index[var2]
                        Q[idx1, idx2] += penalty
                        Q[idx2, idx1] += penalty

    return Q, var_index


def process_solution(configuration, var_index, verbose=True):
    """Processes the binary configuration solution from the solver.

    Args:
        configuration (list): Binary list representing the selected variables.
        var_index (dict): Mapping from stem tuple to index in the QUBO matrix.
        verbose (bool): Whether to print additional details.

    Returns:
        list: List of stems (4-tuples) selected in the solution.
    """
    # Reverse mapping from index to stem tuple.
    index_var = {i: var for var, i in var_index.items()}
    selected = [index_var[i] for i, val in enumerate(configuration) if val == 1]
    print('Stems in best solution:', selected)
    if verbose:
        print('Number of variables (original):', len(configuration))
    return selected


@click.command(help='Solve RNA folding problem using QDeep Hybrid Solver.')
@click.option('--path', type=click.Path(), default='RNA_text_files/TMGMV_UPD-PK1.txt',
              help='Path to the RNA text file.')
@click.option('--verbose/--no-verbose', default=True,
              help='Print additional model information.')
@click.option('--min-stem', type=click.IntRange(1,), default=3,
              help='Minimum length for a stem.')
@click.option('--min-loop', type=click.IntRange(0,), default=2,
              help='Minimum nucleotides between two sides of a stem.')
@click.option('-c', type=click.FloatRange(0,), default=0.3,
              help='Pseudoknot penalty multiplier.')
def main_qdeep(path, verbose, min_stem, min_loop, c):
    if verbose:
        print('\nPreprocessing data from:', path)
    matrix = text_to_matrix(path, min_loop)
    stem_dict = make_stem_dict(matrix, min_stem, min_loop)
    if not stem_dict:
        print('\nNo possible stems found. Check your parameters.')
        return

    # Build QUBO matrix and variable mapping
    Q, var_index = build_qubo(stem_dict, min_stem, c)
    if verbose:
        print('QUBO matrix built. Solving with QDeep Hybrid Solver...')

    # Initialize and configure the QDeep hybrid solver
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token-here"  # Replace with your valid authentication token
    solver.m_budget = 50000  # Measurement budget (can be tuned)
    solver.num_reads = 10000  # Number of reads (can be tuned)

    try:
        response = solver.solve(Q)
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    if verbose:
        print('Solver response:', response)

    configuration = response['QdeepHybridSolver']['configuration']
    stems = process_solution(configuration, var_index, verbose)
    make_plot(path, stems)


if __name__ == "__main__":
    main_qdeep()
