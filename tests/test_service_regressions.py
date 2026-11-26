"""Пограничные сценарии агента и извлечения сверх контрактов курса."""

import pytest

from assistant.agent.agent import named_quantities, run_agent
from assistant.graphrag.extraction import extract_chunk
from assistant.infra.testing import FakeLLM
from tests.contract.test_extraction_and_graph import BULLETIN, GOOD
from tests.contract.test_tools_and_agent import ORDER, Tools, call


@pytest.mark.parametrize("bad_price", ["1042", "3", "2630,50"])
def test_price_must_come_from_price_field(bad_price):
    llm = FakeLLM(
        [
            call("check_stock", part_id="P-1042B", warehouse="Казань"),
            call("get_price", part_id="P-1042B"),
            f"Цена {bad_price} руб.",
            "Цена 2 630 руб.",
        ]
    )
    result = run_agent("Сколько стоит P-1042B?", llm, Tools().as_dict())
    assert result.answer == "Цена 2 630 руб." and not result.refused


def test_budget_is_not_an_order_quantity_even_with_confirmation():
    tools = Tools()
    llm = FakeLLM([call("create_service_order", **{**ORDER, "qty": 500})])
    result = run_agent(
        "Оформи заказ на P-1042B для AR-210 в Казани. Бюджет: 500 рублей.", llm, tools.as_dict(), confirm=True
    )
    assert "Сколько штук" in result.answer
    assert result.pending_action is None and tools.count("create_service_order") == 0


@pytest.mark.parametrize("text", ["Закажи две детали", "Закажите два контроллера", "Количество: 2", "2 шт. P-1042B"])
def test_explicit_quantities(text):
    assert named_quantities(text) == {2}


def test_price_only_reply_identifies_the_part_found_by_search():
    llm = FakeLLM(
        [call("search_knowledge", question="Деталь для AR-210?"), call("get_price", part_id="P-1042B"), "2630 руб."]
    )
    answer = run_agent("Сколько стоит деталь для AR-210?", llm, Tools().as_dict()).answer
    assert answer == "P-1042B: 2630 руб."


@pytest.mark.parametrize("retry, expected_relations", [(GOOD, 3), (GOOD[:-4], 0)])
def test_truncation_after_complete_objects_still_requires_retry(retry, expected_relations):
    llm = FakeLLM([GOOD[:-4], retry])
    result = extract_chunk(BULLETIN, llm)
    assert len(llm.requests) == 2 and len(result.relations) == expected_relations
    if not expected_relations:
        assert not result.entities
