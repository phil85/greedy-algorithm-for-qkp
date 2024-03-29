import time
import numpy as np
import pandas as pd


def compute_ofv(items, nodes, edges):

    # Compute a dense utility matrix
    utility_matrix = np.zeros((len(nodes), len(nodes)))
    rows, cols = np.array(list(edges.keys())).T
    values = np.array(list(edges.values()))
    utility_matrix[rows, cols] = values
    utility_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(utility_matrix).copy()
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    # Add linear utilities
    items = np.array(items, dtype=int)
    ofv = linear_utilities[items].sum()

    # Add quadratic utilities
    ofv += utility_matrix[items][:, items].sum() / 2

    return ofv


def run_greedy_algorithm(nodes, edges, weights, budgets, params):

    # Initialize results
    results = pd.DataFrame()

    # Convert lists to numpy arrays
    nodes = np.array(nodes)
    weights = np.array(weights)

    # Compute a dense utility matrix
    utility_matrix = np.zeros((len(nodes), len(nodes)))
    rows, cols = np.array(list(edges.keys())).T
    values = np.array(list(edges.values()))
    utility_matrix[rows, cols] = values
    utility_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(utility_matrix).copy()
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    # Set time limit
    if 'time_limit' in params:
        time_limit = params['time_limit']
    else:
        time_limit = 1e10

    for budget in budgets:

        # Start stopwatch
        tic = time.perf_counter()

        # Update candidate nodes
        contributions = np.zeros(len(nodes))

        best_ofv = 0
        best_idx = np.zeros(len(nodes), dtype=bool)

        for i in nodes[weights <= budget]:
            idx = np.zeros(len(nodes), dtype=bool)
            idx[i] = True
            ofv = linear_utilities[i]
            total_weight = weights[i]

            contributions[~idx] = utility_matrix[~idx, i]
            contributions[~idx] += linear_utilities[~idx]
            contributions[~idx] /= weights[~idx]

            # Start adding nodes
            for j in range(len(nodes) - 1):

                budget_idx = total_weight + weights <= budget
                ind = ~idx & budget_idx

                if not ind.any():
                    break

                # Identify item to add
                item_to_add = nodes[ind][contributions[ind].argmax()]

                weight = weights[item_to_add]
                ofv += contributions[item_to_add] * weight

                # Update current budget
                total_weight += weight

                contributions[~idx] += utility_matrix[~idx, item_to_add] / weights[~idx]

                # Get best node
                idx[item_to_add] = True

            if ofv > best_ofv:
                best_ofv = ofv
                best_idx = idx.copy()

            elapsed_time = time.perf_counter() - tic
            if elapsed_time > time_limit:
                break

        # Stop stopwatch
        cpu = time.perf_counter() - tic

        # Get results
        result = pd.Series(dtype=object)
        result['items'] = nodes[best_idx]
        result['ofv'] = compute_ofv(result['items'], nodes, edges)
        result['cpu'] = cpu
        result['budget'] = budget
        result['budget_fraction'] = '{:.4f}'.format(budget / sum(weights))
        result['total_weight'] = sum([weights[i] for i in result['items']])
        result['approach'] = 'greedy'

        # Convert to dataframe
        result = result.to_frame().transpose()

        # Append results to result
        results = pd.concat((results, result))

    return results
