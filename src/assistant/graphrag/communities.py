"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 42
MAX_WORDS = 150


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, attrs in sorted(graph.edges(data=True)):
        weight = attrs.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    if undirected.number_of_nodes() == 0:
        return []
    groups = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: (-len(nodes), nodes[0]))
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def _name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("name", node.split(":", 1)[-1])


def _type(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("type", node.split(":", 1)[0])


def _out(graph: nx.DiGraph, node: str, kind: str) -> list[tuple[str, int]]:
    return [
        (target, attrs.get("weight", 1))
        for _, target, attrs in graph.out_edges(node, data=True)
        if attrs.get("type") == kind
    ]


def _replacement_chain(graph: nx.DiGraph, part: str) -> list[str]:
    chain = [part]
    while True:
        nxt = [target for target, _ in _out(graph, chain[-1], "REPLACED_BY") if target not in chain]
        if not nxt:
            return chain
        chain.append(nxt[0])


def community_facts(community: Community, graph: nx.DiGraph) -> list[str]:
    """Подсчёты по сообществу, сделанные кодом. Самые важные идут первыми."""
    members = [node for node in community.nodes if node in graph]
    lines: list[str] = []

    # Отказы по линейкам: линейка входит в сообщество, модели могут лежать в других.
    for line in (node for node in members if _type(graph, node) == "ProductLine"):
        models = sorted(
            source for source, _, attrs in graph.in_edges(line, data=True) if attrs.get("type") == "BELONGS_TO"
        )
        by_component: Counter = Counter()
        by_model: Counter = Counter()
        for model in models:
            for component, weight in _out(graph, model, "FAILED"):
                by_component[_name(graph, component)] += weight
                by_model[_name(graph, model)] += weight
        lines.append(f"Линейка «{_name(graph, line)}»: модели {', '.join(_name(graph, m) for m in models) or 'нет'}.")
        if by_component:
            top = ", ".join(f"{name} – {count}" for name, count in by_component.most_common())
            lines.append(f"Отказы узлов в линейке «{_name(graph, line)}» по отчётам: {top}.")
            models_top = ", ".join(f"{name} – {count}" for name, count in by_model.most_common())
            lines.append(f"Отказы по моделям линейки «{_name(graph, line)}»: {models_top}.")
        replaced = sorted(
            {
                _name(graph, part)
                for model in models
                for part, _ in _out(graph, model, "HAS_PART")
                if _out(graph, part, "REPLACED_BY")
            }
        )
        if replaced:
            lines.append(
                f"Сняты с производства детали из руководств линейки «{_name(graph, line)}»: {', '.join(replaced)}."
            )

    # Замены деталей.
    for part in (node for node in members if _type(graph, node) == "Part"):
        has_parent = any(attrs.get("type") == "REPLACED_BY" for _, _, attrs in graph.in_edges(part, data=True))
        chain = _replacement_chain(graph, part)
        if len(chain) > 1 and not has_parent:
            lines.append(
                f"Цепочка замен: {' -> '.join(_name(graph, p) for p in chain)}, актуальна {_name(graph, chain[-1])}."
            )

    # Отказы по моделям и узлам.
    for model in (node for node in members if _type(graph, node) == "Model"):
        failed = sorted(_out(graph, model, "FAILED"), key=lambda item: (-item[1], item[0]))
        if failed:
            total = sum(weight for _, weight in failed)
            detail = ", ".join(f"{_name(graph, c)} – {w}" for c, w in failed)
            lines.append(f"Модель {_name(graph, model)}: всего отказов {total} ({detail}).")
    for component in (node for node in members if _type(graph, node) == "Component"):
        failures = sum(
            attrs.get("weight", 1)
            for _, _, attrs in graph.in_edges(component, data=True)
            if attrs.get("type") == "FAILED"
        )
        codes = sorted(
            _name(graph, source)
            for source, _, attrs in graph.in_edges(component, data=True)
            if attrs.get("type") == "INDICATES"
        )
        if failures or codes:
            listed = ", ".join(codes) or "нет"
            lines.append(f"Узел «{_name(graph, component)}»: отказов в отчётах {failures}, коды ошибок {listed}.")
    for supplier in (node for node in members if _type(graph, node) == "Supplier"):
        parts = sorted(
            _name(graph, s) for s, _, a in graph.in_edges(supplier, data=True) if a.get("type") == "SUPPLIED_BY"
        )
        lines.append(f"Поставщик {_name(graph, supplier)}: деталей {len(parts)}.")

    kinds = Counter(_type(graph, node) for node in members)
    lines.append("Состав сообщества: " + ", ".join(f"{kind} – {count}" for kind, count in sorted(kinds.items())) + ".")
    return lines


def _truncate(text: str, limit: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:limit])


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Сводку пишет модель одним вызовом. Все числа считает код по весам рёбер и передаёт модели готовыми:
    модель их не вычисляет, а только переносит в текст. Ответ обрезается до 150 слов.
    """
    facts = community_facts(community, graph)
    prompt = (
        "Ниже подсчёты по фрагменту графа знаний сервисной службы «ПолярТех», сделанные кодом. Напиши по ним "
        f"сводку не длиннее {MAX_WORDS} слов: что входит во фрагмент, как связано и что встречается чаще "
        "всего. Пиши только о том, что есть в подсчётах: если линеек или моделей в подсчётах нет, не "
        "упоминай их и не придумывай примеров. Числа и коды переноси точно, вместе с линейкой, моделью или "
        "узлом, к которым они относятся; фразы о линейке пиши с её названием, например «детали из "
        "руководств линейки «…» сняты с производства: …». Ничего не пересчитывай.\n\n"
        "Подсчёты:\n" + "\n".join(f"- {line}" for line in facts)
    )
    reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=400)
    return _truncate(reply.content.strip())
