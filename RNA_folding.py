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
import dimod

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

from qdeepsdk import QDeepHybridSolver


def text_to_matrix(file_name, min_loop):
    """ Reads properly formatted RNA text file and returns a matrix of possible hydrogen bonding pairs.

    Args:
        file_name (str):
            Path to text file.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.

    Returns:
        :class:`numpy.ndarray`:
            Numpy matrix of 0's and 1's, where 1 represents a possible bonding pair.
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
    """ Takes a matrix of potential hydrogen binding pairs and returns a dictionary of possible stems.
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
    """ Checks if 2 stems use any of the same nucleotides.
    """
    if type(stem1) == str or type(stem2) == str:
        return False
    for val in stem2:
        if stem1[0] <= val <= stem1[1] or stem1[2] <= val <= stem1[3]:
            return True
    for val in stem1[1:3]:
        if stem2[0] <= val <= stem2[1] or stem2[2] <= val <= stem2[3]:
            return True
    return False


def pseudoknot_terms(stem_dict, min_stem=3, c=0.3):
    """ Creates a dictionary with all possible pseudoknots as keys and appropriate penalties as values.
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
    """ Produces graph plot and saves as .png file.
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


def build_cqm(stem_dict, min_stem, c):
    """ Creates a constrained quadratic model to optimize most likely stems from a dictionary of possible stems.
    """
    linear_coeffs = {stem: -1 * (stem[1] - stem[0] + 1) ** 2 for sublist in stem_dict.values() for stem in sublist}
    quadratic_coeffs = pseudoknot_terms(stem_dict, min_stem=min_stem, c=c)
    bqm = dimod.BinaryQuadraticModel(linear_coeffs, quadratic_coeffs, 'BINARY')
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(bqm)
    for stem, substems in stem_dict.items():
        if len(substems) > 1:
            zeros = 'Null:' + str(stem)
            cqm.add_variable('BINARY', zeros)
            cqm.add_discrete(substems + [zeros], stem)
    for stem1, stem2 in combinations(stem_dict.keys(), 2):
        if check_overlap(stem1, stem2):
            for stem_pair in product(stem_dict[stem1], stem_dict[stem2]):
                if check_overlap(stem_pair[0], stem_pair[1]):
                    cqm.add_constraint(dimod.quicksum([dimod.Binary(stem) for stem in stem_pair]) <= 1)
    return cqm


def process_bqm_solution(sample, verbose=True):
    bonded_stems = [stem for stem, val in sample.items() if val == 1 and isinstance(stem, tuple)]
    print('Stems in best solution:', bonded_stems)
    if verbose:
        print('Number of variables (original):', len(sample))
    return bonded_stems


@click.command(help='Solve RNA folding problem using a BQM solved by QDeepHybridSolver.')
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
def main_bqm_sa(path, verbose, min_stem, min_loop, c):
    if verbose:
        print('\nPreprocessing data from:', path)
    matrix = text_to_matrix(path, min_loop)
    stem_dict = make_stem_dict(matrix, min_stem, min_loop)
    
    if not stem_dict:
        print('\nNo possible stems found. Check your parameters.')
        return None

    # Build the constrained quadratic model (CQM)
    cqm = build_cqm(stem_dict, min_stem, c)
    
    if verbose:
        print('CQM built. Now converting CQM to BQM...')
    
    # Convert CQM to BQM using a penalty (lagrange multiplier chosen empirically)
    bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
    
    if verbose:
        print('BQM has been created. Now solving using QDeepHybridSolver...')
    
    # Convert the BQM to a QUBO matrix with a fixed ordering.
    ordering = list(bqm.variables)
    n = len(ordering)
    Q_dict, offset = bqm.to_qubo()
    Q = np.zeros((n, n))
    for (var_i, var_j), val in Q_dict.items():
        idx_i = ordering.index(var_i)
        idx_j = ordering.index(var_j)
        Q[idx_i, idx_j] = val

    # Initialize the QDeepHybridSolver and set your authentication token.
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token-here"  # Replace with your valid token

    # Solve the QUBO problem.
    result = solver.solve(Q)
    configuration = result["configuration"]
    # Map solution back to BQM variables.
    bqm_solution = {ordering[i]: configuration[i] for i in range(n)}
    # Invert the solution to recover the original CQM variable space.
    sample = inverter(bqm_solution)
    
    if verbose:
        print('Best BQM sample (after inversion):', sample)
    
    # Process the sample to extract the stems in the solution.
    stems = process_bqm_solution(sample, verbose)
    # Plot the result.
    make_plot(path, stems)

if __name__ == "__main__":
    main_bqm_sa()
