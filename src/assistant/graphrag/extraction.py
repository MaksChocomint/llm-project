"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

_RELATION_HELP = {
    "BELONGS_TO": "модель входит в линейку",
    "HAS_PART": "в модели установлена деталь (из раздела «Состав узлов» руководства)",
    "IS_A": "деталь является узлом этого вида",
    "INDICATES": "код ошибки указывает на отказ узла",
    "FIXED_BY": "ошибка устраняется по регламенту",
    "REPLACED_BY": "старая деталь снята с производства и заменена новой (только если так сказано прямо)",
    "SUPPLIED_BY": "деталь поставляет поставщик",
    "FAILED": "в отчёте о выезде у модели зафиксирован отказ узла",
}

_EXAMPLE_TEXT = (
    "# Руководство по эксплуатации: витрина AR-999\n\n"
    "Витрина AR-999 входит в линейку «Арктика».\n\n"
    "- компрессор – деталь P-9910, поставщик «КомпрессорМаш»\n\n"
    "- E-11 – компрессор не запускается, отказ узла «компрессор»"
)
_EXAMPLE_ANSWER = {
    "entities": [
        {"type": "Model", "name": "AR-999"},
        {"type": "ProductLine", "name": "Арктика"},
        {"type": "Part", "name": "P-9910"},
        {"type": "Component", "name": "компрессор"},
        {"type": "Supplier", "name": "КомпрессорМаш"},
        {"type": "ErrorCode", "name": "E-11"},
    ],
    "relations": [
        {
            "subject_type": "Model",
            "subject": "AR-999",
            "type": "BELONGS_TO",
            "object_type": "ProductLine",
            "object": "Арктика",
        },
        {"subject_type": "Model", "subject": "AR-999", "type": "HAS_PART", "object_type": "Part", "object": "P-9910"},
        {
            "subject_type": "Part",
            "subject": "P-9910",
            "type": "IS_A",
            "object_type": "Component",
            "object": "компрессор",
        },
        {
            "subject_type": "Part",
            "subject": "P-9910",
            "type": "SUPPLIED_BY",
            "object_type": "Supplier",
            "object": "КомпрессорМаш",
        },
        {
            "subject_type": "ErrorCode",
            "subject": "E-11",
            "type": "INDICATES",
            "object_type": "Component",
            "object": "компрессор",
        },
    ],
}


def _system_prompt(vocab: dict[str, list[str]]) -> str:
    relations = "\n".join(
        f"- {name}: {start} -> {end} – {_RELATION_HELP[name]}" for name, (start, end) in RELATION_SIGNATURES.items()
    )
    names = "\n".join(f"- {kind}: {', '.join(items)}" for kind, items in vocab.items())
    return (
        "Ты извлекаешь граф знаний из документов сервисной службы «ПолярТех». Верни JSON с полями "
        "entities и relations.\n\n"
        f"Типы сущностей: {', '.join(ENTITY_TYPES)}.\n"
        "Коды пиши латиницей прописными буквами с дефисом: модель AR-210, деталь P-1042B, ошибка E-17, "
        "регламент SP-104. Номера бюллетеней и отчётов (SB-2025-03, R-2025-017) сущностями не являются.\n"
        "Названия линеек, узлов и поставщиков пиши только из словаря, узлы строчными буквами в именительном "
        f"падеже:\n{names}\n\n"
        "Связи (направление строго от первого типа ко второму):\n"
        f"{relations}\n\n"
        "Извлекай только то, что прямо сказано в тексте. Не придумывай коды. Если деталь «не снята с "
        "производства и не заменяется», связи REPLACED_BY нет. В отчёте о выезде модель связана с отказавшим "
        "узлом связью FAILED. Плановые работы без ошибок и деталей дают пустые списки.\n\n"
        f"Пример.\nТекст:\n{_EXAMPLE_TEXT}\nОтвет:\n{json.dumps(_EXAMPLE_ANSWER, ensure_ascii=False)}"
    )


def _load_truncated(content: str) -> object:
    """Достаёт целые объекты из JSON, обрезанного по лимиту токенов.

    Длинные руководства дают сотни связей, и ответ может оборваться посреди списка. Тогда текст
    обрезается по последней закрывающей фигурной скобке и дописываются недостающие скобки.
    """
    end = len(content)
    for _ in range(50):
        end = content.rfind("}", 0, end)
        if end < 0:
            return None
        head = content[: end + 1]
        tail = "]}" if '"relations"' in head else '], "relations": []}'
        try:
            return json.loads(head + tail)
        except json.JSONDecodeError:
            continue
    return None


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Разбирает ответ модели. None, если ответ не соответствует схеме."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _load_truncated(content)
    except TypeError:
        return None
    if not isinstance(data, dict):
        return None
    entities, relations = data.get("entities"), data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None
    entity_keys = ("type", "name")
    relation_keys = ("subject_type", "subject", "type", "object_type", "object")
    for items, keys in ((entities, entity_keys), (relations, relation_keys)):
        for item in items:
            if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in keys):
                return None
    return entities, relations


def _ground(entity_type: str, raw: str, in_text: set[str], vocab: dict[str, list[str]]) -> Entity | None:
    """Каноническая сущность, подтверждённая текстом чанка или словарём, иначе None.

    in_text – коды, которые встречаются в тексте чанка.
    """
    if entity_type not in ENTITY_TYPES:
        return None
    name = canonical_name(entity_type, raw, vocab)
    if name is None:
        return None
    if entity_type in CODE_TYPES and name not in in_text:
        return None
    return Entity(entity_type, name)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    vocab = vocabulary()
    messages = [
        {"role": "system", "content": _system_prompt(vocab)},
        {"role": "user", "content": f"Текст:\n{chunk.text}\nОтвет:"},
    ]
    reply = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
    parsed = _parse(reply.content)
    if parsed is None:
        # Повтор с пояснением: тот же запрос при temperature=0 вернул бы тот же ответ из кеша.
        retry = messages + [
            {"role": "assistant", "content": reply.content[:2000]},
            {"role": "user", "content": "Ответ не разобрался. Верни один JSON-объект с полями entities и relations."},
        ]
        parsed = _parse(llm.chat(retry, json_schema=EXTRACTION_SCHEMA, max_tokens=4096).content)
    if parsed is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    raw_entities, raw_relations = parsed
    in_text = set(find_codes(chunk.text))
    dropped = 0
    entities: dict[str, Entity] = {}
    for item in raw_entities:
        entity = _ground(item["type"], item["name"], in_text, vocab)
        if entity is None:
            dropped += 1
        else:
            entities.setdefault(entity.node_id, entity)

    relations: dict[tuple, Relation] = {}
    for item in raw_relations:
        subject = _ground(item["subject_type"], item["subject"], in_text, vocab)
        obj = _ground(item["object_type"], item["object"], in_text, vocab)
        relation = Relation(subject, item["type"], obj) if subject and obj else None
        if relation is None or not relation.matches_schema() or subject == obj:
            dropped += 1
            continue
        relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)
        for end in (subject, obj):
            entities.setdefault(end.node_id, end)

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), list(relations.values()), dropped)
