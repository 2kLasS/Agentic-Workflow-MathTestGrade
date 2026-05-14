from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sympy import Eq, N, simplify, solveset
from sympy.core.symbol import Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class MathToolEngine:
    def run(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if mode == "parse_expr":
                expr = self._parse(payload["expression"])
                return self._ok(mode, {"expression": str(expr)})

            if mode == "simplify":
                expr = self._parse(payload["expression"])
                return self._ok(mode, {"simplified": str(simplify(expr))})

            if mode == "check_equivalence":
                expr1 = self._parse(payload["expr1"])
                expr2 = self._parse(payload["expr2"])
                is_equivalent = simplify(expr1 - expr2) == 0
                return self._ok(
                    mode,
                    {
                        "is_equivalent": bool(is_equivalent),
                        "expr1": str(expr1),
                        "expr2": str(expr2),
                    },
                )

            if mode == "substitute":
                expr = self._parse(payload["expression"])
                substitutions = {
                    Symbol(key): self._parse(str(value))
                    for key, value in payload.get("substitutions", {}).items()
                }
                return self._ok(mode, {"result": str(expr.subs(substitutions))})

            if mode == "solve":
                left, right = self._split_equation(str(payload["equation"]))
                variable = Symbol(str(payload["variable"]))
                result = list(solveset(Eq(left, right), variable))
                return self._ok(mode, {"solutions": [str(item) for item in result]})

            if mode == "numeric_check":
                expr = self._parse(payload["expression"])
                substitutions = {
                    Symbol(key): self._parse(str(value))
                    for key, value in payload.get("substitutions", {}).items()
                }
                return self._ok(mode, {"value": str(N(expr.subs(substitutions)))})

            return self._error(mode, f"Unsupported mode: {mode}")
        except Exception as exc:  # noqa: BLE001
            return self._error(mode, str(exc))

    def _parse(self, expression: str):
        normalized = expression.replace("＝", "=").replace("×", "*").replace("÷", "/")
        if "=" in normalized:
            left, _ = self._split_equation(normalized)
            return left
        return parse_expr(normalized, transformations=TRANSFORMATIONS, evaluate=True)

    def _split_equation(self, equation: str):
        normalized = equation.replace("＝", "=")
        left_raw, right_raw = normalized.split("=", maxsplit=1)
        left = parse_expr(left_raw, transformations=TRANSFORMATIONS, evaluate=True)
        right = parse_expr(right_raw, transformations=TRANSFORMATIONS, evaluate=True)
        return left, right

    def _ok(self, mode: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"success": True, "mode": mode, "result": result, "error": ""}

    def _error(self, mode: str, error: str) -> dict[str, Any]:
        return {"success": False, "mode": mode, "result": {}, "error": error}


def math_tool(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    return MathToolEngine().run(mode, payload)
