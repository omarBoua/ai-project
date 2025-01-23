from abc import ABC, abstractmethod
from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from graph import Graph
from node import Node
from part import Part


class MyPredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @abstractmethod
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate()`.
        :param parts: set of parts to form up an assembly (i.e. a graph)
        :return: graph
        """
        # TODO: implement this method
        ...

class OmarPredictionModel(MyPredictionModel, nn.Module):
    def __init__(self, part_vocab_size, family_vocab_size, embed_dim=1, gnn_hidden_dim=32):
        super().__init__()
        self.part_embedding = nn.Embedding(part_vocab_size, embed_dim)
        self.family_embedding = nn.Embedding(family_vocab_size, embed_dim)
        self.gnn_hidden_dim = gnn_hidden_dim

        # MLP for initial node features
        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, gnn_hidden_dim),
            nn.ReLU()
        )

        # Suppose we do 1 GNN layer or skip directly for brevity:
        self.gnn1 = nn.Linear(gnn_hidden_dim, gnn_hidden_dim)

        # Bilinear scoring matrix
        self.bilinear = nn.Parameter(torch.randn(gnn_hidden_dim, gnn_hidden_dim))
        # Or define it as nn.Linear(gnn_hidden_dim, gnn_hidden_dim, bias=False)

    def forward(self, part_ids, family_ids):
        # Node feature initialization
        part_emb = self.part_embedding(part_ids)          # (N, embed_dim)
        family_emb = self.family_embedding(family_ids)    # (N, embed_dim)
        node_features = torch.cat([part_emb, family_emb], dim=1)  # (N, 2*embed_dim)
        node_features = self.node_mlp(node_features)      # (N, gnn_hidden_dim)

        # Example GNN step (if you want)
        node_features = F.relu(self.gnn1(node_features))

        # Now do a bilinear form: (N,D) x (D,D) x (D,N) => (N,N)
        # 1) transform node_features: (N, D) -> (N, D) with W
        transformed = node_features @ self.bilinear       # (N, D)

        # 2) multiply by node_features^T => (N, N)
        # final scores = (N, D) @ (D, N) = (N, N)
        edge_logits = transformed @ node_features.transpose(0,1)

        return edge_logits  # raw logits, shape (N, N)

    @torch.no_grad()
    def __createGraph(self, model, parts, threshold=0.005) -> Graph:
        """
        Evaluate the model on a single graph.

        Args:
            model: Trained model.
            graph: Graph object to evaluate.
            threshold: Threshold for classifying edges (default: 0.5).

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1-score.
        """
        # Get sorted part IDs and family IDs
        parts = sorted(
            list(parts),
            key=lambda part: (part.get_part_id(), part.get_family_id())
        )

        # Move tensors to the same device as the model
        part_ids = torch.tensor([int(part.get_part_id()) for part in parts], dtype=torch.long)
        family_ids = torch.tensor([int(part.get_family_id()) for part in parts], dtype=torch.long)

        # Get sorted part order and true adjacency matrix
        part_order = tuple(part for part in parts)

        # Predict adjacency matrix
        logits = model(part_ids, family_ids)
        probabilities = torch.sigmoid(logits)
        predicted_adjacency = (probabilities > threshold).float()

        res = Graph()

        print(probabilities)

        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                if predicted_adjacency[i][j]:  # If an edge exists
                    res.add_undirected_edge(parts[i], parts[j])

        return res

    def predict_graph(self, parts: Set[Part]) -> Graph:
        return self.__createGraph(self, parts)

class EdgePredictor(MyPredictionModel, nn.Module):
    def __init__(self, part_vocab_size, family_vocab_size,
                 embed_dim=16, hidden_dim=32):
        super().__init__()
        self.part_embedding = nn.Embedding(part_vocab_size, embed_dim)
        self.family_embedding = nn.Embedding(family_vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, part_i, fam_i, part_j, fam_j):
        """
        part_i, fam_i, part_j, fam_j are integer tensors
        of shape (batch_size,).
        """
        pi = self.part_embedding(part_i)  # (B, embed_dim)
        fi = self.family_embedding(fam_i) # (B, embed_dim)
        pj = self.part_embedding(part_j)  # (B, embed_dim)
        fj = self.family_embedding(fam_j) # (B, embed_dim)

        x = torch.cat([pi, fi, pj, fj], dim=1)  # (B, 4*embed_dim)
        x = self.relu(self.fc1(x))             # (B, hidden_dim)
        x = self.fc2(x)                        # (B, 1)
        return x.squeeze(1)  # (B,)

    @torch.no_grad()
    def __predict_edges(self, model, parts, threshold=0.001):
        """
        part_family_list: list of (part_id, family_id)
        We will return a list of edges (i, j) for i < j if predicted prob > threshold.
        """
        parts = sorted(
            list(parts),
            key=lambda part: (part.get_part_id(), part.get_family_id())
        )
        model.eval()
        n = len(parts)
        graph = Graph()

        for i in range(n):
            for j in range(i + 1, n):
                part_i, fam_i = int(parts[i].get_part_id()), int(parts[i].get_family_id())
                part_j, fam_j = int(parts[j].get_part_id()), int(parts[j].get_family_id())

                pi = torch.tensor([part_i], dtype=torch.long)
                fi = torch.tensor([fam_i], dtype=torch.long)
                pj = torch.tensor([part_j], dtype=torch.long)
                fj = torch.tensor([fam_j], dtype=torch.long)

                logit = model(pi, fi, pj, fj)
                prob = torch.sigmoid(logit)
                if prob.item() > threshold:
                    graph.add_edge(parts[i], parts[j])
                    graph.add_edge(parts[j], parts[i])

        return graph

    def predict_graph(self, parts: Set[Part]) -> Graph:
        return self.__predict_edges(self, parts)


def load_model(file_path: str) -> MyPredictionModel:
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    if 'EdgePredictor' in file_path:
        loaded_model = EdgePredictor(part_vocab_size=2271, family_vocab_size=96)

        # Load model to CPU to avoid CUDA-related issues
        loaded_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode before using it for inference
        loaded_model.eval()
        return loaded_model

    else:
        loaded_model = OmarPredictionModel(part_vocab_size=2271, family_vocab_size=96, embed_dim=1, gnn_hidden_dim=32)

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


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    A simple evaluation metric: Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    print(len(predicted_graph.get_nodes()))
    print(len(target_graph.get_nodes()))
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

    model_file_path = 'model_EdgePredictor.pth'
    prediction_model: MyPredictionModel = load_model(model_file_path)

    # For illustration, we compute the eval score on a portion of the training data
    instances = [(graph.get_parts(), graph) for graph in test_graphs[:500]]
    eval_score = evaluate(prediction_model, instances)
    print(eval_score)
