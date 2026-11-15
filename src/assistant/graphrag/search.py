"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

NOT_FOUND = "в базе знаний не найдено"
MAX_FACT_CHARS = 8000
PATH_CUTOFF = 3
# Порядок важности связей, когда фактов больше, чем помещается в запрос.
PRIORITY = ["REPLACED_BY", "FIXED_BY", "INDICATES", "SUPPLIED_BY", "IS_A", "BELONGS_TO", "HAS_PART", "FAILED"]

LOCAL_SYSTEM = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Отвечай кратко и только по пронумерованным фактам "
    "из графа знаний. Факт «A -REL-> B» означает связь от A к B: HAS_PART – в модели установлена деталь, "
    "IS_A – деталь является узлом, INDICATES – ошибка указывает на отказ узла, FIXED_BY – ошибка устраняется "
    "по регламенту, REPLACED_BY – деталь снята с производства и заменена, SUPPLIED_BY – поставщик детали, "
    "BELONGS_TO – модель входит в линейку, FAILED – отказ узла в отчётах. "
    "Коды пиши точно как в фактах. Если деталь заменена, называй последнюю деталь цепочки замен как ту, "
    "которую нужно заказывать сейчас, и её поставщика. "
    f"Если в фактах нет ответа, напиши: «{NOT_FOUND}»."
)


def _doc(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def _name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("name", node.split(":", 1)[-1])


def find_anchors(question: str, kb: KnowledgeBase) -> tuple[list[str], list[str]]:
    """Опорные узлы вопроса и коды, которых в графе нет."""
    anchors: list[str] = []
    missing: list[str] = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node in kb.graph:
            anchors.append(node)
        else:
            missing.append(code)
    names = find_names(question, kb.vocabulary)
    # «реле» внутри «пусковое реле» не должно давать лишних якорей: короткое название, вложенное в найденное
    # длинное, отбрасываем.
    for kind, name in names:
        if any(name != other and name.lower() in other.lower() for _, other in names):
            continue
        node = f"{kind}:{name}"
        if node in kb.graph and node not in anchors:
            anchors.append(node)
    return anchors, missing


def _oriented(graph: nx.DiGraph, a: str, b: str) -> tuple[str, str]:
    return (a, b) if graph.has_edge(a, b) else (b, a)


def _replacement_edges(graph: nx.DiGraph, part: str) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    seen = {part}
    frontier = [part]
    while frontier:
        node = frontier.pop()
        for _, target, attrs in graph.out_edges(node, data=True):
            if attrs.get("type") == "REPLACED_BY" and target not in seen:
                edges.append((node, target))
                seen.add(target)
                frontier.append(target)
    return edges


def _part_details(graph: nx.DiGraph, part: str) -> list[tuple[str, str]]:
    """Цепочка замен детали и поставщики всех деталей цепочки."""
    edges = _replacement_edges(graph, part)
    chain = [part] + [target for _, target in edges]
    for node in chain:
        edges += [
            (node, target)
            for _, target, attrs in graph.out_edges(node, data=True)
            if attrs.get("type") in ("SUPPLIED_BY", "IS_A")
        ]
    return edges


def _path_edges(graph: nx.DiGraph, anchors: list[str]) -> list[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    edges: list[tuple[str, str]] = []
    for a, b in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, a, b, cutoff=PATH_CUTOFF):
            edges += [_oriented(graph, u, v) for u, v in zip(path, path[1:], strict=False)]
    return edges


def _neighbourhood_edges(graph: nx.DiGraph, anchor: str, depth: int) -> list[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    distance = nx.single_source_shortest_path_length(undirected, anchor, cutoff=depth)
    edges = []
    for node, dist in sorted(distance.items(), key=lambda item: item[1]):
        if dist >= depth:
            continue
        for neighbour in undirected.neighbors(node):
            edges.append(_oriented(graph, node, neighbour))
    unique = list(dict.fromkeys(edges))
    near = [edge for edge in unique if anchor in edge]
    far = [edge for edge in unique if anchor not in edge]
    return near + far


def _rank(graph: nx.DiGraph, edges: list[tuple[str, str]], keep_first: int) -> list[tuple[str, str]]:
    """Убирает повторы. Первые `keep_first` рёбер сохраняют порядок, остальные сортируются по важности."""
    unique = list(dict.fromkeys(edges))
    head, tail = unique[:keep_first], unique[keep_first:]

    def key(edge: tuple[str, str]):
        attrs = graph.edges[edge]
        kind = attrs.get("type", "")
        return (PRIORITY.index(kind) if kind in PRIORITY else len(PRIORITY), -attrs.get("weight", 1), edge)

    return head + sorted(tail, key=key)


def _fact(graph: nx.DiGraph, edge: tuple[str, str]) -> str:
    attrs = graph.edges[edge]
    text = f"{_name(graph, edge[0])} -{attrs.get('type')}-> {_name(graph, edge[1])}"
    if attrs.get("type") == "FAILED":
        text += f" (отчётов: {attrs.get('weight', 1)})"
    return text


def collect_facts(question: str, kb: KnowledgeBase) -> tuple[list[str], list[str], list[str]] | None:
    """Факты для вопроса, источники и примечания о заменах. None – надо отказать."""
    graph = kb.graph
    anchors, missing = find_anchors(question, kb)
    if missing or not anchors:
        return None
    if len(anchors) >= 2:
        core = _path_edges(graph, anchors)
        if not core:  # пути нет: берём ближайших соседей каждой опорной сущности
            core = [edge for anchor in anchors for edge in _neighbourhood_edges(graph, anchor, 1)]
    else:
        core = _neighbourhood_edges(graph, anchors[0], 2)
    if not core:
        return None
    parts = [node for edge in core for node in edge if graph.nodes[node].get("type") == "Part"]
    extra = [edge for part in dict.fromkeys(parts) for edge in _part_details(graph, part)]
    if len(anchors) >= 2:
        edges = _rank(graph, extra + core, keep_first=len(dict.fromkeys(extra)))
    else:
        near = sum(1 for edge in dict.fromkeys(core) if anchors[0] in edge)
        edges = _rank(graph, core[:near] + extra + core[near:], keep_first=near + len(dict.fromkeys(extra)))

    facts: list[str] = []
    sources: list[str] = []
    used = 0
    for edge in edges:
        line = f"{len(facts) + 1}. {_fact(graph, edge)}"
        if used + len(line) + 1 > MAX_FACT_CHARS:
            break
        facts.append(line)
        used += len(line) + 1
        for chunk_id in graph.edges[edge].get("sources", []):
            if _doc(chunk_id) not in sources:
                sources.append(_doc(chunk_id))

    notes = []
    for part in dict.fromkeys(parts):
        chain = [part] + [target for _, target in _replacement_edges(graph, part)]
        if len(chain) > 1:
            names = " -> ".join(_name(graph, p) for p in chain)
            notes.append(
                f"Деталь {_name(graph, part)} снята с производства, цепочка замен {names}: "
                f"сейчас заказывать {_name(graph, chain[-1])}."
            )
    return facts, sources, notes


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    collected = collect_facts(question, kb)
    if collected is None:
        return Answer(f"{NOT_FOUND.capitalize()}.", refused=True)
    facts, sources, notes = collected
    context = "Факты:\n" + "\n".join(facts)
    if notes:
        context += "\n\nЗамены деталей:\n" + "\n".join(f"- {note}" for note in notes)
    messages = [
        {"role": "system", "content": LOCAL_SYSTEM},
        {"role": "user", "content": f"{context}\n\nВопрос: {question}"},
    ]
    text = llm.chat(messages).content.strip()
    refused = NOT_FOUND in text.lower() or not text
    return Answer(text or f"{NOT_FOUND.capitalize()}.", sources=sources, facts=facts, refused=refused)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    raise NotImplementedError("Реализуйте global_search")
