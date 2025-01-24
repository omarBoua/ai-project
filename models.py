from abc import ABC, abstractmethod
from typing import Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph
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
        ...


class GraphPredictionModel(MyPredictionModel, nn.Module):
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
        part_emb = self.part_embedding(part_ids)  # (N, embed_dim)
        family_emb = self.family_embedding(family_ids)  # (N, embed_dim)
        node_features = torch.cat([part_emb, family_emb], dim=1)  # (N, 2*embed_dim)
        node_features = self.node_mlp(node_features)  # (N, gnn_hidden_dim)

        # Example GNN step (if you want)
        node_features = F.relu(self.gnn1(node_features))

        # Now do a bilinear form: (N,D) x (D,D) x (D,N) => (N,N)
        # 1) transform node_features: (N, D) -> (N, D) with W
        transformed = node_features @ self.bilinear  # (N, D)

        # 2) multiply by node_features^T => (N, N)
        # final scores = (N, D) @ (D, N) = (N, N)
        edge_logits = transformed @ node_features.transpose(0, 1)

        return edge_logits  # raw logits, shape (N, N)

    @torch.no_grad()
    def __createGraph(self, model, parts, threshold=0.1) -> Graph:
        """
        Evaluate the model on a single graph.

        Args:
            model: Trained model.
            graph: Graph object to evaluate.
            threshold: Threshold for classifying edges.

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

        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                if predicted_adjacency[i][j]:  # If an edge exists
                    res.add_undirected_edge(parts[i], parts[j])

        return res

    def predict_graph(self, parts: Set[Part]) -> Graph:
        return self.__createGraph(self, parts)


class EdgePredictionModel(MyPredictionModel, nn.Module):
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
        fi = self.family_embedding(fam_i)  # (B, embed_dim)
        pj = self.part_embedding(part_j)  # (B, embed_dim)
        fj = self.family_embedding(fam_j)  # (B, embed_dim)

        x = torch.cat([pi, fi, pj, fj], dim=1)  # (B, 4*embed_dim)
        x = self.relu(self.fc1(x))  # (B, hidden_dim)
        x = self.fc2(x)  # (B, 1)
        return x.squeeze(1)  # (B,)

    @torch.no_grad()
    def __predict_edges(self, model, parts, threshold=0.1):
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
