from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from grade_system.config import Settings, load_settings

StructuredSchema = TypeVar("StructuredSchema", bound=BaseModel)


class QwenWorkflowLLM:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        if not self.settings.qwen_api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY，请先在系统环境变量中配置它。")
        if not self.settings.qwen_base_url:
            raise ValueError("未检测到 QWEN_BASE_URL，请先在系统环境变量中配置它。")

        self.client = ChatOpenAI(
            api_key=self.settings.qwen_api_key,
            base_url=self.settings.qwen_base_url,
            model=self.settings.qwen_model,
            temperature=self.settings.qwen_temperature,
            request_timeout=self.settings.qwen_request_timeout_seconds,
            max_retries=self.settings.qwen_max_retries,
        )
        self._usage_totals = self._empty_usage_totals()

    def reset_usage_totals(self) -> None:
        self._usage_totals = self._empty_usage_totals()

    def get_usage_totals(self) -> dict[str, int]:
        return dict(self._usage_totals)

    def invoke_structured(
        self,
        schema: type[StructuredSchema],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredSchema:
        structured_llm = self.client.with_structured_output(schema, include_raw=True)
        result = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_message = result.get("raw")
        self._accumulate_usage(raw_message)

        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            raise parsing_error

        parsed = result.get("parsed")
        if parsed is None:
            response_metadata = getattr(raw_message, "response_metadata", None)
            raise ValueError(
                "Structured Output response is missing parsed content. "
                f"response_metadata={response_metadata!r}"
            )
        return parsed

    def _accumulate_usage(self, raw_message: object | None) -> None:
        if raw_message is None:
            return

        usage_metadata = getattr(raw_message, "usage_metadata", None) or {}
        response_metadata = getattr(raw_message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

        input_tokens = int(
            usage_metadata.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or 0
        )
        output_tokens = int(
            usage_metadata.get("output_tokens")
            or token_usage.get("completion_tokens")
            or 0
        )
        total_tokens = int(
            usage_metadata.get("total_tokens")
            or token_usage.get("total_tokens")
            or input_tokens + output_tokens
        )

        reasoning_tokens = 0
        output_token_details = usage_metadata.get("output_token_details") or {}
        if isinstance(output_token_details, dict):
            reasoning_tokens = int(output_token_details.get("reasoning") or 0)
        if not reasoning_tokens and isinstance(token_usage, dict):
            completion_details = token_usage.get("completion_tokens_details") or {}
            if isinstance(completion_details, dict):
                reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

        self._usage_totals["llm_call_count"] += 1
        self._usage_totals["input_tokens"] += input_tokens
        self._usage_totals["output_tokens"] += output_tokens
        self._usage_totals["total_tokens"] += total_tokens
        self._usage_totals["reasoning_tokens"] += reasoning_tokens

    def _empty_usage_totals(self) -> dict[str, int]:
        return {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
