from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

import click
import requests
from requests import Response

from .config import APISettings


class APIClient:
    """Thin wrapper over requests to talk to the Portfolio API."""

    def __init__(self, settings: APISettings, timeout: int = 30) -> None:
        self.settings = settings
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.settings.token:
            headers["Authorization"] = f"Bearer {self.settings.token}"
        return headers

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        expect_json: bool = True,
    ) -> Any:
        url = f"{self.settings.base_url}{path}"
        try:
            response = self.session.get(
                url, headers=self.headers, params=params, timeout=self.timeout, stream=stream
            )
        except requests.RequestException as exc:
            raise click.ClickException(f"GET {url} failed: {exc}") from exc
        return self._handle_response(response, expect_json=expect_json)

    def post(
        self,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        files: Optional[Iterable] = None,
        params: Optional[Dict[str, Any]] = None,
        expect_json: bool = True,
    ) -> Any:
        url = f"{self.settings.base_url}{path}"
        try:
            response = self.session.post(
                url,
                headers=self.headers,
                params=params,
                json=json_body,
                files=files,
                timeout=self.timeout,
                stream=not expect_json,
            )
        except requests.RequestException as exc:
            raise click.ClickException(f"POST {url} failed: {exc}") from exc
        return self._handle_response(response, expect_json=expect_json)

    def _handle_response(self, response: Response, *, expect_json: bool) -> Any:
        if not response.ok:
            self._raise_for_status(response)

        if expect_json:
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise click.ClickException("Response was not valid JSON") from exc

        return response

    def _raise_for_status(self, response: Response) -> None:
        try:
            payload = response.json()
            message = payload.get("detail") or payload
        except Exception:
            message = response.text
        raise click.ClickException(
            f"Request failed with status {response.status_code}: {message}"
        )
