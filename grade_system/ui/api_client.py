from __future__ import annotations

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    pass


class GradeSystemApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def register(self, username: str, password: str, display_name: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/auth/register",
            payload={
                "username": username,
                "password": password,
                "display_name": display_name,
            },
        )

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/auth/login",
            payload={"username": username, "password": password},
        )

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/auth/refresh",
            payload={"refresh_token": refresh_token},
        )

    def logout(self, refresh_token: str, access_token: str) -> None:
        self._request_json(
            "POST",
            "/api/v1/auth/logout",
            payload={"refresh_token": refresh_token},
            access_token=access_token,
        )

    def get_me(self, access_token: str) -> dict[str, Any]:
        return self._request_json("GET", "/api/v1/auth/me", access_token=access_token)

    def create_grading_task(
        self,
        access_token: str,
        question_text: str,
        student_answer_text: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/grading-tasks",
            payload={
                "question_text": question_text,
                "student_answer_text": student_answer_text,
            },
            access_token=access_token,
        )

    def list_grading_tasks(
        self,
        access_token: str,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            "/api/v1/grading-tasks",
            access_token=access_token,
            params={"page": page, "page_size": page_size},
        )

    def get_grading_task_detail(self, access_token: str, task_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"/api/v1/grading-tasks/{task_id}",
            access_token=access_token,
        )

    def _headers(self, access_token: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        access_token: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    json=payload,
                    params=params,
                    headers=self._headers(access_token),
                )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise ApiClientError(detail) from exc
        except httpx.HTTPError as exc:
            raise ApiClientError(f"无法连接后端服务：{exc}") from exc

    def _extract_error_detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return f"后端请求失败（HTTP {response.status_code}）。"

        detail = data.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return f"后端请求失败（HTTP {response.status_code}）。"
