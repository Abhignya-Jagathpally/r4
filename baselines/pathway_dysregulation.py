"""Pathway dysregulation scoring (M23: test-set AUC, M27: BH correction)."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class PathwayDysregulation:
    """Pathway dysregulation with PLS-DA and enrichment testing."""

    def pls_da(
        self, X: np.ndarray, y: np.ndarray, n_components: int = 5, cv_folds: int = 5,
    ) -> Dict:
        """PLS-DA with TEST-set AUC reporting (M23)."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        test_aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_comp = min(n_components, X_train.shape[1], X_train.shape[0])
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train, y_train)

            # TEST-set prediction (M23: not training AUC)
            y_pred = pls.predict(X_test).flatten()
            try:
                auc = roc_auc_score(y_test, y_pred)
                test_aucs.append(auc)
            except ValueError:
                pass

        return {
            "mean_test_auc": float(np.mean(test_aucs)) if test_aucs else float("nan"),
            "std_test_auc": float(np.std(test_aucs)) if test_aucs else float("nan"),
            "n_folds": len(test_aucs),
        }

    def enrichment_test(
        self, gene_list: List[str], pathway_db: Dict[str, List[str]],
        background: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Hypergeometric enrichment with BH correction (M27)."""
        from statsmodels.stats.multitest import multipletests

        if background is None:
            background = list(set(g for genes in pathway_db.values() for g in genes))

        gene_set = set(gene_list)
        bg_set = set(background)
        N = len(bg_set)

        results = []
        for pathway, genes in pathway_db.items():
            pw_set = set(genes) & bg_set
            if not pw_set:
                continue
            overlap = gene_set & pw_set
            k = len(overlap)
            K = len(pw_set)
            n = len(gene_set & bg_set)
            p_value = stats.hypergeom.sf(k - 1, N, K, n)
            results.append({
                "pathway": pathway, "n_overlap": k, "n_pathway": K,
                "p_value": p_value, "overlap_genes": ",".join(sorted(overlap)),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            # BH correction (M27)
            _, adj_p, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
            df["p_adjusted"] = adj_p
            df["significant"] = df["p_adjusted"] < 0.05
            df = df.sort_values("p_adjusted")

        return df.set_index("pathway") if len(df) > 0 else df
