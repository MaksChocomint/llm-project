"""Проверки аргументов инструментов. Общие для tools.py и agent.py."""

import re

from assistant.infra.grounding import code_type, normalize_code

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART = re.compile(r"P-\d{4}[A-Z]?")


def part_id(raw) -> str | None:
    """Канонический номер детали вида P-1042 или P-1042B, иначе None."""
    if not isinstance(raw, str):
        return None
    code = normalize_code(raw)
    return code if code and code_type(code) == "Part" and _PART.fullmatch(code) else None


def model_code(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    code = normalize_code(raw)
    return code if code and code_type(code) == "Model" else None


def quantity(raw) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str) and raw.strip().isdigit():
        raw = int(raw.strip())
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    return raw if isinstance(raw, int) and raw > 0 else None


def check_order(args: dict) -> tuple[dict | None, str | None]:
    """Нормализованные аргументы заказа или текст ошибки."""
    part = part_id(args.get("part_id"))
    if part is None:
        return None, f"неверный номер детали {args.get('part_id')!r}: нужен вид P-1042 или P-1042B"
    if args.get("warehouse") not in WAREHOUSES:
        return None, f"неизвестный склад {args.get('warehouse')!r}: допустимы {', '.join(WAREHOUSES)}"
    qty = quantity(args.get("qty"))
    if qty is None:
        return None, f"неверное количество {args.get('qty')!r}: нужно целое число больше нуля"
    model = model_code(args.get("model"))
    if model is None:
        return None, f"неверный код модели {args.get('model')!r}: нужен вид AR-210"
    return {"part_id": part, "warehouse": args["warehouse"], "qty": qty, "model": model}, None
