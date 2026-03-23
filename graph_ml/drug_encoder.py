"""Drug SMILES to molecular graph encoder."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DrugEncoder:
    """Encode drug SMILES into molecular graphs."""

    ATOM_FEATURES = ["atomic_num", "degree", "formal_charge", "is_aromatic", "num_hs"]

    def smiles_to_graph(self, smiles: str) -> Optional[Dict]:
        """Convert SMILES to graph with node/edge features."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            atoms = mol.GetAtoms()
            n_atoms = len(atoms)

            # Node features
            node_features = []
            for atom in atoms:
                node_features.append([
                    atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                    int(atom.GetIsAromatic()), atom.GetTotalNumHs(),
                ])
            node_features = np.array(node_features, dtype=float)

            # Edge features and indices
            edge_src, edge_dst, edge_features = [], [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                feat = [
                    float(bond.GetBondTypeAsDouble()),
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                    int(bond.GetStereo()),
                ]
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_features.extend([feat, feat])

            return {
                "node_features": node_features,
                "edge_index": np.array([edge_src, edge_dst]) if edge_src else np.zeros((2, 0), dtype=int),
                "edge_features": np.array(edge_features) if edge_features else np.zeros((0, 4)),
                "n_atoms": n_atoms,
            }
        except ImportError:
            logger.warning("RDKit not available, returning dummy graph")
            return {"node_features": np.random.randn(10, 5), "edge_index": np.array([[0,1],[1,0]]),
                    "edge_features": np.random.randn(2, 4), "n_atoms": 10}
