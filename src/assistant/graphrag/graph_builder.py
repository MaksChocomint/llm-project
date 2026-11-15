"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from collections import defaultdict

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def _add_node(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    node = entity.node_id
    if node not in graph:
        graph.add_node(node, type=entity.type, name=entity.name, sources=[])
    if chunk_id not in graph.nodes[node]["sources"]:
        graph.nodes[node]["sources"] = sorted([*graph.nodes[node]["sources"], chunk_id])


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    edges: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for result in results:
        for entity in result.entities:
            _add_node(graph, entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            _add_node(graph, relation.subject, result.chunk_id)
            _add_node(graph, relation.object, result.chunk_id)
            edges[(relation.subject.node_id, relation.type, relation.object.node_id)].add(result.chunk_id)

    # Для одной пары узлов схема допускает только один тип связи, но на случай расхождений
    # оставляем тип, найденный в большем числе чанков.
    best: dict[tuple[str, str], tuple[str, set[str]]] = {}
    for (source, kind, target), chunks in sorted(edges.items()):
        current = best.get((source, target))
        if current is None or len(chunks) > len(current[1]):
            best[(source, target)] = (kind, chunks)

    conflicts = 0
    removed: set[tuple[str, str]] = set()
    for (source, target), (kind, chunks) in sorted(best.items()):
        back = best.get((target, source))
        if kind != "REPLACED_BY" or back is None or back[0] != "REPLACED_BY" or (source, target) in removed:
            continue
        if source > target:
            continue  # каждую встречную пару разбираем один раз
        conflicts += 1
        forward_weight, backward_weight = len(chunks), len(back[1])
        if forward_weight >= backward_weight:
            removed.add((target, source))
        if backward_weight >= forward_weight:
            removed.add((source, target))

    for (source, target), (kind, chunks) in best.items():
        if (source, target) in removed:
            continue
        graph.add_edge(source, target, type=kind, weight=len(chunks), sources=sorted(chunks))
    graph.graph["conflicts"] = conflicts
    return graph
