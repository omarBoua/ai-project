import pickle
from itertools import permutations
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import hamming
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from graph import Graph
from models import MyPredictionModel, GraphPredictionModel, EdgePredictionModel
from part import Part


def load_model(file_path: str) -> MyPredictionModel:
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    if 'edge_predictor' in file_path:
        loaded_model = EdgePredictionModel(part_vocab_size=2271, family_vocab_size=96)

        # Load model to CPU to avoid CUDA-related issues
        loaded_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode before using it for inference
        loaded_model.eval()
        return loaded_model

    else:
        loaded_model = GraphPredictionModel(part_vocab_size=2271, family_vocab_size=96, embed_dim=1, gnn_hidden_dim=32)

        # Load model to CPU to avoid CUDA-related issues
        loaded_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode before using it for inference
        loaded_model.eval()
        return loaded_model


def evaluate(model: MyPredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> float:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0

    for input_parts, target_graph in data_set:
        predicted_graph = model.predict_graph(input_parts)

        # We prepared a simple evaluation metric `edge_accuracy()`for you
        # Think of other suitable metrics for this task and evaluate your model on them!
        # FYI: maybe some more evaluation metrics will be used in final evaluation
        edges_counter += len(input_parts) * len(input_parts)
        sum_correct_edges += edge_accuracy(predicted_graph, target_graph)

    # return value in percent
    return sum_correct_edges / edges_counter * 100


def edge_hamming_distance(predicted_graph: Graph, target_graph: Graph) -> int:
    perms = __generate_part_list_permutations(predicted_graph.get_parts())
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order).flatten()

    min_distance = float('inf')

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm).flatten()
        distance = hamming(target_adj_matrix, predicted_adj_matrix) * len(target_adj_matrix)
        min_distance = min(min_distance, distance)

    # Divide by 2 since the adjacency matrix is symmetric
    return min_distance / 2


def edge_jaccard_similarity(predicted_graph: Graph, target_graph: Graph) -> float:
    perms = __generate_part_list_permutations(predicted_graph.get_parts())
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order).flatten()

    best_jaccard = 0

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm).flatten()
        jaccard = jaccard_score(target_adj_matrix, predicted_adj_matrix)
        best_jaccard = max(best_jaccard, jaccard)

    return best_jaccard


def graph_edit_distance(predicted_graph: Graph, target_graph: Graph) -> float:
    perms = __generate_part_list_permutations(predicted_graph.get_parts())
    target_parts_order = perms[0]
    target_nx_graph = nx.Graph(target_graph.get_adjacency_matrix(target_parts_order))

    best_edit = float('inf')
    for perm in perms:
        predicted_nx_graph = nx.Graph(predicted_graph.get_adjacency_matrix(perm))
        edit = nx.graph_edit_distance(predicted_nx_graph, target_nx_graph)
        best_edit = min(best_edit, edit)

    return best_edit


def evaluate_all_metrics(model: MyPredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> None:
    """
    Evaluates a given prediction model on a given data set using multiple evaluation metrics.

    :param model: prediction model
    :param data_set: data set containing input parts and target graphs
    :return: None (prints mean metrics)
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    hamming_distances = []
    jaccard_similarities = []
    edit_distances = []
    edge_accuracies = []
    failed_graphs = 0

    progress_bar = tqdm(data_set, desc="Processing graphs", unit="graph", dynamic_ncols=True)

    for input_parts, target_graph in progress_bar:
        predicted_graph = model.predict_graph(input_parts)
        if len(predicted_graph.get_nodes()) != len(target_graph.get_nodes()):
            failed_graphs += 1
            continue

        # Compute precision, recall, F1-score
        precision, recall, f1 = edge_precision_recall_f1(predicted_graph, target_graph)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Compute Hamming distance
        hamming_distances.append(edge_hamming_distance(predicted_graph, target_graph))

        # Compute Jaccard similarity
        jaccard_similarities.append(edge_jaccard_similarity(predicted_graph, target_graph))

        # Compute Graph edit distance
        edit_distances.append(graph_edit_distance(predicted_graph, target_graph))

        # Edge accuracy
        edge_accuracies.append(edge_accuracy(predicted_graph, target_graph) / (len(input_parts) * len(input_parts)))

        # Update tqdm progress bar with current mean values
        progress_bar.set_postfix({
            "failed": f"{failed_graphs}",
            "P": f"{np.mean(precision_scores):.4f}",
            "R": f"{np.mean(recall_scores):.4f}",
            "F1": f"{np.mean(f1_scores):.4f}",
            "Hamming": f"{np.mean(hamming_distances):.4f}",
            "Jaccard": f"{np.mean(jaccard_similarities):.4f}",
            "Edit Dist": f"{np.mean(edit_distances):.4f}",
            "Acc": f"{100 * np.mean(edge_accuracies):.2f}%",
        })

    # Calculate final mean values
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_hamming = np.mean(hamming_distances)
    mean_jaccard = np.mean(jaccard_similarities)
    mean_edit_distance = np.mean(edit_distances)
    edge_accuracy_value = 100 * np.mean(edge_accuracies)

    # Print the evaluation results
    print("\nEvaluation Results:")
    print(f"  Number of invalid graphs due to mismatch in number of nodes: {failed_graphs}")
    print(f"  Precision: {mean_precision:.4f}")
    print(f"  Recall: {mean_recall:.4f}")
    print(f"  F1-score: {mean_f1:.4f}")
    print(f"  Hamming Distance: {mean_hamming:.4f}")
    print(f"  Jaccard Similarity: {mean_jaccard:.4f}")
    print(f"  Graph Edit Distance: {mean_edit_distance:.4f}")
    print(f"  Edge Accuracy: {edge_accuracy_value:.4f}%")


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    A simple evaluation metric: Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(predicted_graph.get_parts())

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score


def edge_precision_recall_f1(predicted_graph: Graph, target_graph: Graph) -> Tuple[float, float, float]:
    perms = __generate_part_list_permutations(predicted_graph.get_parts())
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order).flatten()

    best_precision, best_recall, best_f1 = 0, 0, 0

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm).flatten()

        precision = precision_score(target_adj_matrix, predicted_adj_matrix)
        recall = recall_score(target_adj_matrix, predicted_adj_matrix)
        f1 = f1_score(target_adj_matrix, predicted_adj_matrix)

        best_precision = max(best_precision, precision)
        best_recall = max(best_recall, recall)
        best_f1 = max(best_f1, f1)

    return best_precision, best_recall, best_f1


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all([len(perm) == len(parts) for perm in full_perms]), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Example code for evaluation

if __name__ == '__main__':
    # Load train data
    with open('./data/graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)
        train_graphs_list, test_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)

    # Load the final model

    model_file_path = 'edge_predictor_model.pth'
    prediction_model: MyPredictionModel = load_model(model_file_path)

    # For illustration, we compute the eval score on a portion of the training data
    instances = [(graph.get_parts(), graph) for graph in test_graphs[:500]]
    eval_score = evaluate(prediction_model, instances)
    print(eval_score)
