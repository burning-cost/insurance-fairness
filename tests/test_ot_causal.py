"""Tests for CausalGraph and PathDecomposer."""
import numpy as np
import polars as pl
import pytest

from insurance_fairness.optimal_transport.causal import CausalGraph, PathDecomposer, PathDecomposition


def simple_graph() -> CausalGraph:
    return (
        CausalGraph()
        .add_protected("gender")
        .add_justified_mediator("claims_history", parents=["gender"])
        .add_proxy("annual_mileage", parents=["gender"])
        .add_outcome("claim_freq")
        .add_edge("claims_history", "claim_freq")
        .add_edge("annual_mileage", "claim_freq")
    )


class TestCausalGraphBuilder:
    def test_chaining_returns_self(self):
        g = CausalGraph()
        result = g.add_protected("gender")
        assert result is g

    def test_get_protected_nodes(self):
        g = simple_graph()
        assert g.get_protected_nodes() == ["gender"]

    def test_get_proxy_nodes(self):
        g = simple_graph()
        assert g.get_proxy_nodes() == ["annual_mileage"]

    def test_get_justified_nodes(self):
        g = simple_graph()
        assert g.get_justified_nodes() == ["claims_history"]

    def test_get_outcome_nodes(self):
        g = simple_graph()
        assert g.get_outcome_nodes() == ["claim_freq"]

    def test_to_networkx(self):
        import networkx as nx
        g = simple_graph()
        nxg = g.to_networkx()
        assert isinstance(nxg, nx.DiGraph)
        assert "gender" in nxg.nodes
        assert "claim_freq" in nxg.nodes

    def test_add_edge_creates_edge(self):
        g = CausalGraph().add_protected("s").add_outcome("y")
        g.add_edge("s", "y")
        assert ("s", "y") in g.to_networkx().edges

    def test_add_proxy_auto_adds_parent_edges(self):
        g = CausalGraph().add_protected("s").add_proxy("v", parents=["s"]).add_outcome("y")
        assert ("s", "v") in g.to_networkx().edges

    def test_add_justified_auto_adds_parent_edges(self):
        g = CausalGraph().add_protected("s").add_justified_mediator("r", parents=["s"]).add_outcome("y")
        assert ("s", "r") in g.to_networkx().edges

    def test_unknown_parent_raises(self):
        g = CausalGraph()
        with pytest.raises(ValueError, match="not yet added"):
            g.add_proxy("v", parents=["nonexistent"])

    def test_add_edge_unknown_source_raises(self):
        g = CausalGraph().add_outcome("y")
        with pytest.raises(ValueError, match="Source node"):
            g.add_edge("nonexistent", "y")

    def test_add_edge_unknown_target_raises(self):
        g = CausalGraph().add_protected("s")
        with pytest.raises(ValueError, match="Target node"):
            g.add_edge("s", "nonexistent")

    def test_repr(self):
        g = simple_graph()
        r = repr(g)
        assert "CausalGraph" in r
        assert "gender" in r

    def test_covariate_node(self):
        g = CausalGraph().add_covariate("age").add_protected("s").add_outcome("y")
        nxg = g.to_networkx()
        assert "age" in nxg.nodes


class TestCausalGraphValidation:
    def test_valid_graph_passes(self):
        g = simple_graph()
        g.validate()  # no exception

    def test_no_protected_node_raises(self):
        g = CausalGraph().add_outcome("y")
        with pytest.raises(ValueError, match="no protected nodes"):
            g.validate()

    def test_no_outcome_raises(self):
        g = CausalGraph().add_protected("s")
        with pytest.raises(ValueError, match="no outcome node"):
            g.validate()

    def test_cycle_raises(self):
        g = CausalGraph().add_protected("s").add_outcome("y")
        g._g.add_edge("s", "y")
        g._g.add_edge("y", "s")  # cycle
        with pytest.raises(ValueError, match="cycle"):
            g.validate()

    def test_disconnected_protected_raises(self):
        g = CausalGraph().add_protected("s").add_outcome("y")
        # No edge from s to y
        with pytest.raises(ValueError, match="no path"):
            g.validate()

    def test_multiple_outcomes_raises(self):
        g = CausalGraph().add_protected("s").add_outcome("y1").add_outcome("y2")
        g.add_edge("s", "y1")
        g.add_edge("s", "y2")
        with pytest.raises(ValueError, match="multiple outcome"):
            g.validate()


class TestPathClassification:
    def test_direct_path(self):
        g = simple_graph()
        path = ["gender", "claim_freq"]
        assert g.classify_path(path) == "direct"

    def test_proxy_path(self):
        g = simple_graph()
        path = ["gender", "annual_mileage", "claim_freq"]
        assert g.classify_path(path) == "proxy"

    def test_justified_path(self):
        g = simple_graph()
        path = ["gender", "claims_history", "claim_freq"]
        assert g.classify_path(path) == "justified"

    def test_paths_from_protected_to_outcome(self):
        g = simple_graph()
        paths = g.paths_from_protected_to_outcome()
        assert "gender" in paths
        # Should include direct and mediated paths
        assert len(paths["gender"]) >= 1


class TestPathDecomposer:
    def _make_data(self, n=100):
        rng = np.random.default_rng(42)
        X = pl.DataFrame({
            "claims_history": rng.choice(["yes", "no"], n).tolist(),
            "annual_mileage": rng.integers(5000, 30000, n).tolist(),
            "gender": rng.choice(["M", "F"], n).tolist(),
            "claim_freq": np.zeros(n).tolist(),  # outcome placeholder
        })
        return X

    def _simple_model(self, df: pl.DataFrame) -> np.ndarray:
        """Toy model: base + 0.1 * (gender==M) + 0.05 * (claims_history==yes)."""
        n = df.shape[0]
        pred = np.ones(n) * 0.1
        if "gender" in df.columns:
            pred += (df["gender"] == "M").to_numpy().astype(float) * 0.05
        if "claims_history" in df.columns:
            pred += (df["claims_history"] == "yes").to_numpy().astype(float) * 0.03
        return pred

    def test_decompose_returns_dataclass(self):
        g = simple_graph()
        decomposer = PathDecomposer(g, self._simple_model)
        X = self._make_data(50)
        result = decomposer.decompose(X, {"gender": ["M", "F"]})
        assert isinstance(result, PathDecomposition)

    def test_decompose_shapes(self):
        g = simple_graph()
        decomposer = PathDecomposer(g, self._simple_model)
        X = self._make_data(50)
        result = decomposer.decompose(X, {"gender": ["M", "F"]})
        assert result.total_premium.shape == (50,)
        assert result.direct_effect.shape == (50,)
        assert result.proxy_effect.shape == (50,)
        assert result.justified_effect.shape == (50,)

    def test_decompose_protected_attr(self):
        g = simple_graph()
        decomposer = PathDecomposer(g, self._simple_model)
        X = self._make_data(50)
        result = decomposer.decompose(X, {"gender": ["M", "F"]})
        assert result.protected_attr == "gender"

    def test_as_polars_returns_dataframe(self):
        g = simple_graph()
        decomposer = PathDecomposer(g, self._simple_model)
        X = self._make_data(20)
        result = decomposer.decompose(X, {"gender": ["M", "F"]})
        df = result.as_polars()
        assert isinstance(df, pl.DataFrame)
        assert "total_premium" in df.columns

    def test_missing_d_values_raises(self):
        g = simple_graph()
        decomposer = PathDecomposer(g, self._simple_model)
        X = self._make_data(20)
        with pytest.raises(ValueError, match="missing key"):
            decomposer.decompose(X, {"nonexistent_attr": ["A", "B"]})
