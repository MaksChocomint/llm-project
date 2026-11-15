"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.agent.validation import check_order, part_id
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

ERP_ATTEMPTS = 3
NOT_FOUND = "в базе знаний не найдено"
ERP_FAILED = "Данные ERP получить не удалось."
SYSTEM = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Отвечай кратко по-русски. "
    "Номер детали узнавай только инструментом search_knowledge: передай ему вопрос с моделью и кодом ошибки "
    "или названием узла. Если деталь заменена, работай с последней деталью цепочки замен. "
    "Остаток детали узнавай инструментом check_stock, цену – get_price. Обзорные вопросы о частоте отказов "
    "задавай summarize_topic. Заказ оформляй create_service_order только когда пользователь просит заказ "
    "и сам назвал количество. Никогда не придумывай номера деталей, цены и остатки: пиши только то, что "
    "вернули инструменты. Если инструмент вернул ошибку «данных нет», так и скажи. "
    f"Если данных нет, ответь: «{NOT_FOUND}»."
)

_NUMBER_WORDS = {
    "один": 1,
    "одну": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "пару": 2,
    "пара": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}
_CODE_LIKE = re.compile(r"[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?(?![\dA-Za-zА-Яа-яЁё])")
_AMOUNT = r"(?<![\w-])(\d{1,3}(?:[   ]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(_AMOUNT + r"\s*(?:руб|₽|шт)", re.IGNORECASE)
_AFTER_WORD = re.compile(r"(?:цен[аеуы]|стоимост\w*|остат\w*|налич\w*)\D{0,20}?" + _AMOUNT, re.IGNORECASE)


def named_quantities(question: str) -> set[int]:
    """Числа, которые пользователь назвал в вопросе, без цифр внутри кодов."""
    text = _CODE_LIKE.sub(" ", question)
    numbers = {int(raw) for raw in re.findall(r"(?<!\d)\d+(?!\d)", text)}
    numbers |= {value for word, value in _NUMBER_WORDS.items() if re.search(rf"\b{word}\b", text.lower())}
    return numbers


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def ungrounded(answer: str, question: str, tool_text: str) -> list[str]:
    """Коды, цены и остатки ответа, которых нет ни в вопросе, ни в результатах инструментов."""
    allowed_codes = set(find_codes(question)) | set(find_codes(tool_text))
    allowed_numbers = set(re.findall(r"\d+", tool_text)) | set(re.findall(r"\d+", question))
    problems = [code for code in find_codes(answer) if code not in allowed_codes]
    for raw in _WITH_UNIT.findall(answer) + _AFTER_WORD.findall(answer):
        if _digits(raw) not in allowed_numbers:
            problems.append(_digits(raw))
    return list(dict.fromkeys(problems))


def _call_key(name: str, args: dict) -> str:
    return json.dumps([name, args], ensure_ascii=False, sort_keys=True)


def _summary(steps: list[ToolStep]) -> str:
    known = [
        f"{step.tool}({json.dumps(step.args, ensure_ascii=False)}) -> {json.dumps(step.result, ensure_ascii=False)}"
        for step in steps
        if step.error is None and step.result is not None
    ]
    return "; ".join(known) if known else "ничего"


class _Run:
    """Состояние одного диалога агента."""

    def __init__(self, question: str, tools: dict[str, Callable[..., dict]], confirm: bool) -> None:
        self.question = question
        self.tools = tools
        self.confirm = confirm
        self.steps: list[ToolStep] = []
        self.sources: list[str] = []
        self.cache: dict[str, tuple[dict | None, str | None]] = {}
        self.tool_texts: list[str] = []
        self.known_parts = {code for code in find_codes(question) if code_type(code) == "Part"}
        self.discontinued: set[str] = set()
        self.erp_failed = False
        self.kb_refused = False
        self.not_found = False  # ERP ответила «данных нет»
        self.order_id: str | None = None
        self.stop: AgentResult | None = None  # досрочный результат: подтверждение или уточнение

    @property
    def tool_text(self) -> str:
        return "\n".join(self.tool_texts)

    def result(self, answer: str, **kwargs) -> AgentResult:
        return AgentResult(answer, steps=self.steps, sources=self.sources, **kwargs)

    def _record(self, name: str, args: dict, result: dict | None, error: str | None) -> dict:
        self.steps.append(ToolStep(name, args, result, error))
        return result if result is not None else {"error": error}

    def execute(self, call: ToolCall) -> dict:
        name, args = call.name, call.arguments
        if name not in self.tools:
            return self._record(name, args, None, f"неизвестный инструмент {name!r}, доступны: {', '.join(self.tools)}")
        if "_raw" in args:
            return self._record(name, args, None, "аргументы не разобраны как JSON-объект, повторите вызов")

        if "part_id" in args:
            part = part_id(args.get("part_id"))
            if part is None or part not in self.known_parts:
                return self._record(
                    name,
                    args,
                    None,
                    f"номер детали {args.get('part_id')!r} не встречался ни в вопросе, ни в результатах "
                    "инструментов. Сначала вызовите search_knowledge, чтобы узнать номер детали",
                )

        key = _call_key(name, args)
        if key in self.cache:
            result, error = self.cache[key]
            return self._record(name, args, result, error)

        if name == "create_service_order":
            order, error = check_order(args)
            if error:
                return self._record(name, args, None, error)
            if order["qty"] not in named_quantities(self.question):
                self._record(name, args, None, "количество не названо пользователем")
                self.stop = self.result(
                    f"Сколько штук детали {order['part_id']} заказать для {order['model']} на склад "
                    f"{order['warehouse']}? Укажите количество."
                )
                return {"error": "количество не названо пользователем"}
            if order["part_id"] in self.discontinued:
                return self._record(name, args, None, f"деталь {order['part_id']} снята с производства: найдите замену")
            if not self.confirm:
                self._record(name, args, None, "ожидает подтверждения человека, заказ не создан")
                self.stop = self.result(
                    f"Подготовлен заказ: деталь {order['part_id']}, {order['qty']} шт., склад {order['warehouse']}, "
                    f"модель {order['model']}. Заказ будет создан после подтверждения.",
                    needs_confirmation=True,
                    pending_action={"tool": name, "args": order},
                )
                return {"status": "pending_confirmation"}

        result, error = self._run_tool(name, args)
        self.cache[key] = (result, error)
        return self._record(name, args, result, error)

    def _run_tool(self, name: str, args: dict) -> tuple[dict | None, str | None]:
        last = ""
        for _ in range(ERP_ATTEMPTS):
            try:
                result = self.tools[name](**args)
                break
            except ErpUnavailable as exc:
                last = str(exc)
            except TypeError as exc:
                return None, f"неверные аргументы: {exc}"
        else:
            self.erp_failed = True
            return None, f"ERP недоступна после {ERP_ATTEMPTS} попыток: {last}"

        if not isinstance(result, dict):
            result = {"result": result}
        self.tool_texts.append(json.dumps(result, ensure_ascii=False))
        self.known_parts |= {code for code in find_codes(self.tool_text) if code_type(code) == "Part"}
        for source in result.get("sources", []) or []:
            if source not in self.sources:
                self.sources.append(source)
        if result.get("discontinued") and isinstance(result.get("part_id"), str):
            self.discontinued.add(result["part_id"])
        if result.get("outcome") == "not_found":
            self.not_found = True
        if name == "search_knowledge" and result.get("refused"):
            self.kb_refused = True
        if name == "create_service_order" and result.get("order_id"):
            self.order_id = str(result["order_id"])
        return result, result.get("error")


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    run = _Run(question, tools, confirm)
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    while True:
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not reply.tool_calls:
            return _finish(run, llm, messages, reply.content.strip())
        messages.append(reply.as_message())
        for call in reply.tool_calls:
            if len(run.steps) >= max_steps:
                return run.result(
                    f"Работа прервана: исчерпан лимит в {max_steps} вызовов инструментов. "
                    f"Что удалось узнать: {_summary(run.steps)}."
                )
            observation = run.execute(call)
            if run.stop is not None:
                return run.stop
            messages.append(tool_message(call, observation))


def _finish(run: _Run, llm: LLMClient, messages: list[dict], answer: str) -> AgentResult:
    for attempt in range(2):
        problems = ungrounded(answer, run.question, run.tool_text)
        if not problems and answer:
            break
        if attempt == 1:
            return run.result(
                f"{NOT_FOUND.capitalize()}: ответ не удалось подтвердить данными инструментов.", refused=True
            )
        messages.append({"role": "assistant", "content": answer})
        issue = f"В ответе есть данные, которых нет в вопросе и результатах инструментов: {', '.join(problems)}. "
        if not answer:
            issue = "Ответ пустой. "
        messages.append(
            {
                "role": "user",
                "content": issue + "Исправь ответ: используй только данные из результатов инструментов.",
            }
        )
        answer = llm.chat(messages).content.strip()

    if run.erp_failed and "не удалось" not in answer.lower():
        answer = f"{answer} {ERP_FAILED}".strip()
    if run.order_id and run.order_id not in answer:
        answer = f"{answer} Номер заказа: {run.order_id}."
    lowered = answer.lower()
    no_data = any(marker in lowered for marker in ("не найден", "данных нет", "неизвестн", "не заведен"))
    refused = NOT_FOUND in lowered or ((run.kb_refused or run.not_found) and no_data)
    if refused and NOT_FOUND not in lowered:
        answer = f"{NOT_FOUND.capitalize()}. {answer}"
    return run.result(answer, refused=refused)
