"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.agent.validation import WAREHOUSES
from assistant.infra.erp_client import ErpClient
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


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    raise NotImplementedError("Реализуйте build_tools")
