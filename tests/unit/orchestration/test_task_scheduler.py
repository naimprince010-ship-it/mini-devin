"""Unit tests for DAG task scheduler (orchestration)."""

from mini_devin.orchestration.task_scheduler import (
    SchedulableUnit,
    auto_skip_when_dep_skipped,
    topological_layers,
    transitive_descendants,
)


def test_topological_layers_parallel_roots():
    a = SchedulableUnit(id="a", goal="ga", acceptance_criteria=[], depends_on=())
    b = SchedulableUnit(id="b", goal="gb", acceptance_criteria=[], depends_on=())
    c = SchedulableUnit(id="c", goal="gc", acceptance_criteria=[], depends_on=("a", "b"))
    layers = topological_layers([a, b, c])
    assert len(layers) == 2
    assert {x.id for x in layers[0]} == {"a", "b"}
    assert [x.id for x in layers[1]] == ["c"]


def test_topological_layers_chain():
    a = SchedulableUnit(id="a", goal="ga", acceptance_criteria=[], depends_on=())
    b = SchedulableUnit(id="b", goal="gb", acceptance_criteria=[], depends_on=("a",))
    c = SchedulableUnit(id="c", goal="gc", acceptance_criteria=[], depends_on=("b",))
    layers = topological_layers([a, b, c])
    assert len(layers) == 3
    assert [layer[0].id for layer in layers] == ["a", "b", "c"]


def test_transitive_descendants():
    a = SchedulableUnit(id="a", goal="ga", acceptance_criteria=[], depends_on=())
    b = SchedulableUnit(id="b", goal="gb", acceptance_criteria=[], depends_on=("a",))
    c = SchedulableUnit(id="c", goal="gc", acceptance_criteria=[], depends_on=("b",))
    d = SchedulableUnit(id="d", goal="gd", acceptance_criteria=[], depends_on=("a",))
    nodes = [a, b, c, d]
    assert transitive_descendants(nodes, "a") == {"b", "c", "d"}


def test_auto_skip_when_dep_skipped():
    a = SchedulableUnit(id="a", goal="ga", acceptance_criteria=[], depends_on=())
    b = SchedulableUnit(id="b", goal="gb", acceptance_criteria=[], depends_on=("a",))
    c = SchedulableUnit(id="c", goal="gc", acceptance_criteria=[], depends_on=("b",))
    skipped = {"a"}
    units = [a, b, c]
    auto_skip_when_dep_skipped(units, skipped)
    assert skipped == {"a", "b", "c"}
