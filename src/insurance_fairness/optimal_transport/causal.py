"""Causal graph specification and path decomposition for insurance fairness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np
import polars as pl


_NODE_TYPES = frozenset({"protected", "proxy", "justified_mediator", "outcome", "covariate"})


class CausalGraph:
    """Directed acyclic graph for specifying the insurance pricing fairness structure.

    Nodes are typed as protected (S), proxy (V), justified mediator (R),
    outcome (Y), or covariate (Z). The graph drives path decomposition and
    determines which variables participate in discrimination-free correction.

    Builder pattern — every mutating method returns self for chaining:

        graph = (CausalGraph()
            .add_protected("gender")
            .add_justified_mediator("claims_history", parents=["gender"])
            .add_proxy("annual_mileage", parents=["gender"])
            .add_outcome("claim_freq")
            .add_edge("claims_history", "claim_freq")
            .add_edge("annual_mileage", "claim_freq"))
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._types: dict[str, str] = {}

    def _add_node(self, name: str, node_type: str) -> "CausalGraph":
        if node_type not in _NODE_TYPES:
            raise ValueError(f"Unknown node type '{node_type}'. Must be one of {_NODE_TYPES}")
        self._g.add_node(name, node_type=node_type)
        self._types[name] = node_type
        return self

    def add_protected(self, name: str) -> "CausalGraph":
        """Register a protected attribute (S node). Chainable."""
        return self._add_node(name, "protected")

    def add_justified_mediator(self, name: str, parents: list[str]) -> "CausalGraph":
        """Register a justified risk variable (R node) with its causal parents."""
        self._add_node(name, "justified_mediator")
        for p in parents:
            if p not in self._g:
                raise ValueError(f"Parent '{p}' not yet added to graph")
            self._g.add_edge(p, name)
        return self

    def add_proxy(self, name: str, parents: list[str]) -> "CausalGraph":
        """Register a proxy variable for a protected attribute (V node)."""
        self._add_node(name, "proxy")
        for p in parents:
            if p not in self._g:
                raise ValueError(f"Parent '{p}' not yet added to graph")
            self._g.add_edge(p, name)
        return self

    def add_outcome(self, name: str) -> "CausalGraph":
        """Register the claims/loss outcome node (Y node)."""
        return self._add_node(name, "outcome")

    def add_covariate(self, name: str) -> "CausalGraph":
        """Register a non-protected, non-mediator covariate (Z node)."""
        return self._add_node(name, "covariate")

    def add_edge(self, source: str, target: str) -> "CausalGraph":
        """Add a causal edge between existing nodes."""
        if source not in self._g:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target not in self._g:
            raise ValueError(f"Target node '{target}' not found in graph")
        self._g.add_edge(source, target)
        return self

    def validate(self) -> None:
        """Raise ValueError if the graph is invalid.

        Checks: acyclicity, at least one protected node, exactly one outcome node,
        and that each protected node has a path to the outcome.
        """
        if not nx.is_directed_acyclic_graph(self._g):
            cycles = list(nx.find_cycle(self._g))
            raise ValueError(f"Causal graph contains a cycle: {cycles}")

        protected = self.get_protected_nodes()
        if not protected:
            raise ValueError("Graph has no protected nodes. Add at least one with add_protected().")

        outcomes = self.get_outcome_nodes()
        if not outcomes:
            raise ValueError("Graph has no outcome node. Add one with add_outcome().")
        if len(outcomes) > 1:
            raise ValueError(f"Graph has multiple outcome nodes: {outcomes}. Only one is supported.")

        outcome = outcomes[0]
        for p in protected:
            if not nx.has_path(self._g, p, outcome):
                raise ValueError(
                    f"Protected node '{p}' has no path to outcome '{outcome}'. "
                    "Ensure causal edges connect the protected attribute to the outcome."
                )

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the underlying networkx DiGraph."""
        return self._g.copy()

    def get_protected_nodes(self) -> list[str]:
        """Return all nodes classified as protected (S)."""
        return [n for n, t in self._types.items() if t == "protected"]

    def get_proxy_nodes(self) -> list[str]:
        """Return all nodes classified as proxy (V)."""
        return [n for n, t in self._types.items() if t == "proxy"]

    def get_justified_nodes(self) -> list[str]:
        """Return all nodes classified as justified mediators (R)."""
        return [n for n, t in self._types.items() if t == "justified_mediator"]

    def get_outcome_nodes(self) -> list[str]:
        """Return all nodes classified as outcome (Y)."""
        return [n for n, t in self._types.items() if t == "outcome"]

    def paths_from_protected_to_outcome(self) -> dict[str, list[list[str]]]:
        """Return all simple paths from each protected node to the outcome, grouped by protected node."""
        outcomes = self.get_outcome_nodes()
        if not outcomes:
            return {}
        outcome = outcomes[0]
        result = {}
        for p in self.get_protected_nodes():
            paths = list(nx.all_simple_paths(self._g, p, outcome))
            result[p] = paths
        return result

    def classify_path(self, path: list[str]) -> str:
        """Classify a single causal path as 'direct', 'proxy', or 'justified'.

        'direct': path S -> Y with no intermediary
        'proxy': path passes through at least one proxy node
        'justified': path passes through only justified mediators
        """
        intermediaries = path[1:-1]
        if not intermediaries:
            return "direct"
        node_types = [self._types.get(n, "covariate") for n in intermediaries]
        if any(t == "proxy" for t in node_types):
            return "proxy"
        return "justified"

    def __repr__(self) -> str:
        n = len(self._g.nodes)
        e = len(self._g.edges)
        protected = self.get_protected_nodes()
        return f"CausalGraph(nodes={n}, edges={e}, protected={protected})"


@dataclass
class PathDecomposition:
    """Results of causal path decomposition for a protected attribute.

    All arrays have shape (n,), matching the input data rows.

    direct_effect: premium variance from the direct S -> Y path
    proxy_effect: premium variance via proxy (V) intermediaries
    justified_effect: premium variance via justified mediator (R) intermediaries
    total_premium: best-estimate prediction from the model
    protected_attr: name of the protected attribute being decomposed
    """

    direct_effect: np.ndarray
    proxy_effect: np.ndarray
    justified_effect: np.ndarray
    total_premium: np.ndarray
    protected_attr: str
    policy_ids: np.ndarray | None = None

    def as_polars(self) -> pl.DataFrame:
        """Return decomposition as a Polars DataFrame."""
        data = {
            "total_premium": self.total_premium,
            "direct_effect": self.direct_effect,
            "proxy_effect": self.proxy_effect,
            "justified_effect": self.justified_effect,
        }
        if self.policy_ids is not None:
            data["policy_id"] = self.policy_ids
        return pl.DataFrame(data)


class PathDecomposer:
    """Decomposes premium into direct, proxy, and justified causal path effects.

    Uses do-calculus: for each path type, constructs counterfactual predictions
    by setting the protected attribute to a reference value and computing the
    premium shift. The decomposition is additive in log-space.
    """

    def __init__(
        self,
        graph: CausalGraph,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
    ) -> None:
        graph.validate()
        self.graph = graph
        self.model_fn = model_fn

    def decompose(
        self,
        X: pl.DataFrame,
        D_values: dict[str, list],
        exposure: np.ndarray | None = None,
    ) -> PathDecomposition:
        """Compute causal path attribution for each observation.

        D_values must supply all observed values for each protected attribute,
        e.g. {"gender": ["M", "F"]}.

        The decomposition method:
        - Computes best-estimate predictions under actual D values
        - Computes predictions under the reference (first) D value
        - Separates the total shift into proxy-path and justified-path components
          by comparing predictions when proxy columns are neutralised vs when
          justified mediator columns are neutralised
        """
        protected_attrs = self.graph.get_protected_nodes()
        if not protected_attrs:
            raise ValueError("Graph has no protected nodes")
        if len(protected_attrs) > 1:
            raise ValueError(
                "PathDecomposer currently supports one protected attribute at a time. "
                f"Graph has: {protected_attrs}"
            )

        attr = protected_attrs[0]
        if attr not in D_values:
            raise ValueError(f"D_values missing key '{attr}'")

        n = X.shape[0]
        proxy_nodes = self.graph.get_proxy_nodes()
        justified_nodes = self.graph.get_justified_nodes()

        # Best-estimate predictions
        best_est = self.model_fn(X)

        # Reference value: first in D_values list
        ref_val = D_values[attr][0]

        # Build reference DataFrame: set protected attribute to ref value
        X_ref = X.clone()
        if attr in X_ref.columns:
            X_ref = X_ref.with_columns(pl.lit(ref_val).alias(attr))
        ref_pred = self.model_fn(X_ref)

        # Total shift = log(best_est) - log(ref_pred)
        total_shift = np.log(best_est) - np.log(ref_pred)

        # Helper: compute a scalar "neutral" value for a column in the reference group.
        # For numeric columns: exposure-weighted mean within gender=ref_val.
        # For string/categorical columns: mode (most common value) within gender=ref_val.
        # This is used to neutralise a mediator column's variation while keeping
        # gender fixed at ref_val, so that the path S -> mediator -> Y is blocked.
        def _neutral_value(col: str) -> object:
            if attr in X.columns:
                ref_mask = (X[attr] == ref_val).to_numpy()
                ref_rows = X.filter(pl.Series(ref_mask))
            else:
                ref_rows = X
            if len(ref_rows) == 0:
                ref_rows = X
            series = ref_rows[col]
            if series.dtype in (pl.String, pl.String, pl.Categorical, pl.Boolean):
                # Mode: most frequent value
                counts = series.value_counts(sort=True)
                return counts[col][0]
            else:
                # Exposure-weighted mean (use uniform weights for simplicity)
                return float(series.mean())

        # Proxy effect: neutralise justified mediators (set to ref-group average/mode),
        # keep proxy columns at their original values, gender at ref_val.
        # This isolates the variation that flows through proxy paths only.
        # proxy_shift = pred(gender=ref, justified=neutral, proxy=original) - pred(X_ref)
        if justified_nodes:
            X_no_justified = X_ref.clone()
            for col in justified_nodes:
                if col in X.columns and col in X_no_justified.columns:
                    neutral = _neutral_value(col)
                    X_no_justified = X_no_justified.with_columns(
                        pl.lit(neutral).alias(col)
                    )
            proxy_pred = self.model_fn(X_no_justified)
            proxy_shift = np.log(proxy_pred) - np.log(ref_pred)
        else:
            proxy_shift = np.zeros(n)

        # Justified effect: residual after removing proxy shift from total.
        justified_shift = total_shift - proxy_shift

        # Direct effect: neutralise ALL mediating variables (both proxy and justified),
        # keeping gender at ref_val. This leaves only the direct S -> Y pathway.
        # direct_shift = pred(gender=ref, all_mediators=neutral) - pred(X_ref)
        intermediary_nodes = set(proxy_nodes) | set(justified_nodes)
        if intermediary_nodes:
            X_no_med = X_ref.clone()
            for col in intermediary_nodes:
                if col in X.columns and col in X_no_med.columns:
                    neutral = _neutral_value(col)
                    X_no_med = X_no_med.with_columns(pl.lit(neutral).alias(col))
            direct_pred = self.model_fn(X_no_med)
            direct_shift = np.log(direct_pred) - np.log(ref_pred)
        else:
            direct_shift = total_shift

        # Re-scale so effects are on original premium scale (multiplicative factors)
        best_log = np.log(best_est)
        return PathDecomposition(
            direct_effect=direct_shift * best_est,
            proxy_effect=proxy_shift * best_est,
            justified_effect=justified_shift * best_est,
            total_premium=best_est,
            protected_attr=attr,
        )
