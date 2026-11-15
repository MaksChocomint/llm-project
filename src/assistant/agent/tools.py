"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.agent.validation import WAREHOUSES, check_order
from assistant.agent.validation import part_id as normalize_part
from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_QUESTION = {"type": "string", "description": "вопрос на русском языке с кодами модели, ошибки или детали"}
_PART = {"type": "string", "description": "номер детали вида P-1042 или P-1042B, взятый из базы знаний"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES, "description": "склад"}

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
TOOL_SCHEMAS: list[dict] = [
    _function(
        "search_knowledge",
        "Поиск по графу знаний документов: какой узел отказал по коду ошибки, какая деталь стоит в модели, "
        "чем она заменена, кто поставщик, какой регламент. Вызывай первым, чтобы узнать номер детали.",
        {"question": _QUESTION},
        ["question"],
    ),
    _function(
        "summarize_topic",
        "Обзор по всему корпусу документов: какой узел отказывает чаще всего, у какой модели больше отказов.",
        {"question": _QUESTION},
        ["question"],
    ),
    _function(
        "check_stock",
        "Остаток детали на складе по данным ERP и признак discontinued (снята с производства).",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _function(
        "get_price",
        "Цена детали в рублях по данным ERP и признак discontinued (снята с производства).",
        {"part_id": _PART},
        ["part_id"],
    ),
    _function(
        "create_service_order",
        "Создаёт заказ детали на складе для модели. Выполняется только после подтверждения человеком.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "количество, названное пользователем"},
            "model": {"type": "string", "description": "код модели оборудования вида AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def _erp_error(exc: Exception, part: str) -> dict:
    if isinstance(exc, ErpNotFound):
        return {"error": f"данных нет: деталь {part} не заведена в ERP ({exc})", "outcome": "not_found"}
    return {"error": f"ERP отклонила запрос: {exc}", "outcome": "rejected"}


def _discontinued_note(result: dict) -> dict:
    if result.get("discontinued"):
        return {
            **result,
            "note": "деталь снята с производства, заказ невозможен: найдите замену через search_knowledge",
        }
    return result


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question=None, **_) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "нужен непустой вопрос question"}
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question=None, **_) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "нужен непустой вопрос question"}
        answer = global_search(question, kb, llm)
        if answer.refused:
            # Модель иногда отправляет сюда точный вопрос («по какому регламенту…»). Сводки на него не
            # отвечают, а граф отвечает, поэтому пробуем точный поиск, прежде чем отказать.
            exact = local_search(question, kb, llm)
            if not exact.refused:
                return {"answer": exact.text, "facts": exact.facts, "sources": exact.sources, "refused": False}
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id=None, warehouse=None, **_) -> dict:
        part = normalize_part(part_id)
        if part is None:
            return {"error": f"неверный номер детали {part_id!r}: нужен вид P-1042 или P-1042B"}
        if warehouse not in WAREHOUSES:
            return {"error": f"неизвестный склад {warehouse!r}: допустимы {', '.join(WAREHOUSES)}"}
        try:
            return _discontinued_note(erp.get_stock(part, warehouse))
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc, part)

    def get_price(part_id=None, **_) -> dict:
        part = normalize_part(part_id)
        if part is None:
            return {"error": f"неверный номер детали {part_id!r}: нужен вид P-1042 или P-1042B"}
        try:
            return _discontinued_note(erp.get_price(part))
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc, part)

    def create_service_order(**arguments) -> dict:
        order, error = check_order(arguments)
        if error:
            return {"error": error}
        try:
            return erp.create_order(order["part_id"], order["warehouse"], order["qty"], order["model"])
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc, order["part_id"])

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
