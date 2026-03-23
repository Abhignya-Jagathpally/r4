"""Tests for graph_ml module."""
import numpy as np
import torch
import pytest


def test_tensor_fusion_shape():
    from graph_ml.fusion_model import TensorFusion
    model = TensorFusion(drug_dim=64, prot_dim=32, hidden_dim=16, out_dim=1)
    drug = torch.randn(8, 64)
    prot = torch.randn(8, 32)
    out = model(drug, prot)
    assert out.shape == (8, 1)


def test_drug_aware_split():
    import pandas as pd
    from graph_ml.run_graph_ml import drug_aware_split
    df = pd.DataFrame({"drug_id": ["A"]*10 + ["B"]*10 + ["C"]*10, "y": range(30)})
    train_idx, test_idx = drug_aware_split(df, "drug_id", test_size=0.33)
    train_drugs = set(df.loc[train_idx, "drug_id"])
    test_drugs = set(df.loc[test_idx, "drug_id"])
    assert train_drugs.isdisjoint(test_drugs)


def test_smiles_hash():
    from graph_ml.run_graph_ml import hash_smiles
    h1 = hash_smiles(["CCO", "CC"])
    h2 = hash_smiles(["CC", "CCO"])
    assert h1 == h2  # Order independent


def test_pathway_encoder():
    from graph_ml.proteomics_encoder import PathwayAwareEncoder
    model = PathwayAwareEncoder(input_dim=100, hidden_dim=32, output_dim=64)
    x = torch.randn(4, 100)
    out = model(x)
    assert out.shape == (4, 64)  # M39: output dim matches
