from torch.utils.data import Dataset

from graph import *
from node import *


class LinkPredictionDataset(Dataset):
    """
    Creates positive/negative samples from each Graph.
    For each Graph:
      - Collect all nodes
      - For every pair (i, j), check if it's an edge (label=1) or not (label=0)
    """

    def __init__(self, graphs: List[Graph]):
        super().__init__()
        self.samples = []

        for g in graphs:
            # Get the nodes and edges
            node_list = g.get_nodes()  # List[Node]
            edge_list = g.get_edges()  # List of (Node, Node)
            edge_set = self.get_edge_list(edge_list)  # for quick membership checks

            # Map node ID -> (part_id, family_id)
            node_id_to_features = {}
            for node in node_list:
                node_id_to_features[node.get_id()] = (
                    node.get_part().get_part_id(),
                    node.get_part().get_family_id()
                )

            # We'll gather all node IDs from the node list
            node_ids = [n.get_id() for n in node_list]
            id_to_node = {n.get_id(): n for n in node_list}

            # Create all (i, j) pairs
            for i in node_ids:
                for j in node_ids:
                    if i == j:
                        continue
                    part_i, fam_i = node_id_to_features[i]
                    part_j, fam_j = node_id_to_features[j]

                    # Sort the pair for an undirected edge check
                    pair = tuple(sorted([id_to_node[i], id_to_node[j]],
                                        key=lambda x: x.get_id()))
                    label = 1 if pair in edge_set else 0

                    self.samples.append((int(part_i), int(fam_i), int(part_j), int(fam_j), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (part_i, fam_i, part_j, fam_j, label)

    def get_edge_list(self, __edges: Dict[Node, List[Node]]):
        edge_pairs = set()  # use a set to avoid duplicates

        for src, neighbors in __edges.items():
            for dst in neighbors:
                # Sort the pair so that (NodeA, NodeB) == (NodeB, NodeA)
                sorted_pair = tuple(sorted([src, dst], key=lambda n: n.get_id()))
                edge_pairs.add(sorted_pair)

        return edge_pairs  # Now we have a list of (Node, Node) pairs
