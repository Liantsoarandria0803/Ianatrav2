#!/usr/bin/env python3
"""MAÏA backend — smoke test.

Ce script teste les fonctionnalités principales du backend en enchaînant :
- healthcheck
- register + login + /auth/me
- diagnostic start + submit (LLM)
- session start + message en streaming SSE (LLM)
- session history
- profile/competences

Usage:
  python backend/scripts/smoke_test.py --base-url https://<backend>/api/v1

Variables d'env (optionnelles):
  MAIA_API_BASE_URL: base URL de l'API (ex: https://xxx.onrender.com/api/v1)

Notes:
- Les étapes Diagnostic + Session consomment des appels LLM (Groq). Utilise --skip-llm si besoin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date

import httpx


def _normalize_api_base_url(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://localhost:8000/api/v1"
    return base if base.endswith("/api/v1") else f"{base}/api/v1"


def _root_url_from_api_base_url(api_base_url: str) -> str:
    base = (api_base_url or "").rstrip("/")
    if base.endswith("/api/v1"):
        return base[: -len("/api/v1")]
    return base


@dataclass(frozen=True)
class Cfg:
    base_url: str
    timeout_s: float
    skip_llm: bool


def _fail(step: str, message: str) -> None:
    print(f"❌ {step}: {message}")
    raise SystemExit(1)


def _ok(step: str, message: str = "OK") -> None:
    print(f"✅ {step}: {message}")


def _request_json(client: httpx.Client, method: str, url: str, *, headers: dict[str, str] | None = None, json_body=None, step: str) -> dict:
    try:
        res = client.request(method, url, headers=headers, json=json_body)
    except httpx.RequestError as e:
        _fail(step, f"RequestError: {e}")

    content_type = (res.headers.get("content-type") or "").lower()
    data = {}
    if "application/json" in content_type:
        try:
            data = res.json()
        except Exception:
            data = {}

    if res.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        extra = f" — detail={detail}" if detail else ""
        _fail(step, f"HTTP {res.status_code}{extra} — body={res.text[:500]}")

    if not isinstance(data, dict):
        _fail(step, f"Réponse JSON inattendue: {type(data)}")

    return data


def _stream_sse_tokens(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    step: str,
) -> str:
    full_text: list[str] = []
    try:
        with client.stream("POST", url, headers=headers, json=payload) as res:
            if res.status_code >= 400:
                _fail(step, f"HTTP {res.status_code} — body={res.text[:500]}")

            for line in res.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                raw = line[len("data:") :].strip()
                if not raw:
                    continue

                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if isinstance(evt, dict) and evt.get("error"):
                    _fail(step, f"Erreur SSE: {evt.get('error')}")

                if isinstance(evt, dict) and evt.get("done"):
                    return "".join(full_text)

                token = evt.get("token") if isinstance(evt, dict) else None
                if isinstance(token, str):
                    full_text.append(token)

    except httpx.RequestError as e:
        _fail(step, f"RequestError: {e}")

    _fail(step, "Flux SSE terminé sans évènement done")
    return ""  # unreachable


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test du backend MAÏA")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MAIA_API_BASE_URL", "http://localhost:8000/api/v1"),
        help="Base URL de l'API (ex: https://xxx.onrender.com/api/v1)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout HTTP global (secondes)")
    parser.add_argument("--skip-llm", action="store_true", help="Ignore diagnostic + session SSE (pas d'appels LLM)")
    args = parser.parse_args()

    cfg = Cfg(base_url=_normalize_api_base_url(args.base_url), timeout_s=args.timeout, skip_llm=bool(args.skip_llm))
    root_url = _root_url_from_api_base_url(cfg.base_url)

    print("\nMAÏA backend smoke test")
    print(f"- API: {cfg.base_url}")
    print(f"- root: {root_url}")
    print(f"- timeout: {cfg.timeout_s}s")
    print(f"- skip_llm: {cfg.skip_llm}")

    # User data
    email = f"smoke+{uuid.uuid4().hex[:10]}@example.com"
    password = "SmokeTest123!"
    name = "Smoke Test"
    vertical = "concours"
    exam_date = str(date.today().replace(year=date.today().year + 1))

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    with httpx.Client(timeout=cfg.timeout_s, limits=limits, follow_redirects=True) as client:
        # Health
        _request_json(client, "GET", f"{root_url}/health", step="health")
        _ok("health")

        # Register
        register_payload = {
            "email": email,
            "password": password,
            "name": name,
            "vertical": vertical,
            "exam_date": exam_date,
        }
        reg = _request_json(client, "POST", f"{cfg.base_url}/auth/register", json_body=register_payload, step="auth.register")
        access_token = reg.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            _fail("auth.register", "access_token manquant")
        _ok("auth.register", f"user={email}")

        headers_auth = {"Authorization": f"Bearer {access_token}"}

        # /me
        me = _request_json(client, "GET", f"{cfg.base_url}/auth/me", headers=headers_auth, step="auth.me")
        if me.get("email") != email:
            _fail("auth.me", f"email inattendu: {me.get('email')} (attendu {email})")
        _ok("auth.me")

        # Login
        login_payload = {"email": email, "password": password}
        login = _request_json(client, "POST", f"{cfg.base_url}/auth/login", json_body=login_payload, step="auth.login")
        if not isinstance(login.get("access_token"), str):
            _fail("auth.login", "access_token manquant")
        _ok("auth.login")

        if cfg.skip_llm:
            _ok("diagnostic", "skipped")
        else:
            # Diagnostic start
            diag = _request_json(client, "POST", f"{cfg.base_url}/diagnostic/start", headers=headers_auth, json_body={}, step="diagnostic.start")
            diagnostic_id = diag.get("diagnostic_id")
            questions = diag.get("questions")
            if not isinstance(diagnostic_id, str) or not diagnostic_id:
                _fail("diagnostic.start", "diagnostic_id manquant")
            if not isinstance(questions, list) or not questions:
                _fail("diagnostic.start", "questions manquant")
            _ok("diagnostic.start", f"questions={len(questions)}")

            # Diagnostic submit
            answers = []
            for q in questions:
                qid = q.get("id") if isinstance(q, dict) else None
                if isinstance(qid, int):
                    answers.append({"question_id": qid, "answer": "Réponse de test (smoke)"})
            submit_payload = {"diagnostic_id": diagnostic_id, "answers": answers}
            submit = _request_json(client, "POST", f"{cfg.base_url}/diagnostic/submit", headers=headers_auth, json_body=submit_payload, step="diagnostic.submit")
            scores = submit.get("scores")
            if not isinstance(scores, list):
                _fail("diagnostic.submit", "scores manquant")
            _ok("diagnostic.submit", f"scores={len(scores)}")

        # Session start (DB only)
        sess_payload = {"mode": "cours", "topic": None}
        sess = _request_json(client, "POST", f"{cfg.base_url}/session/start", headers=headers_auth, json_body=sess_payload, step="session.start")
        session_id = sess.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            _fail("session.start", "session_id manquant")
        _ok("session.start", f"id={session_id}")

        if cfg.skip_llm:
            _ok("session.message(SSE)", "skipped")
        else:
            # SSE message (LLM)
            sse_url = f"{cfg.base_url}/session/{session_id}/message"
            reply = _stream_sse_tokens(
                client,
                sse_url,
                headers=headers_auth,
                payload={"content": "Bonjour MAÏA, réponse courte stp."},
                step="session.message(SSE)",
            )
            _ok("session.message(SSE)", f"chars={len(reply)}")

        # History
        hist = _request_json(client, "GET", f"{cfg.base_url}/session/{session_id}/history", headers=headers_auth, step="session.history")
        msgs = hist.get("messages")
        if not isinstance(msgs, list):
            _fail("session.history", "messages manquant")
        if not cfg.skip_llm and len(msgs) < 2:
            _fail("session.history", f"messages insuffisants: {len(msgs)}")
        _ok("session.history", f"messages={len(msgs)}")

        # Profile
        prof = _request_json(client, "GET", f"{cfg.base_url}/profile/competences", headers=headers_auth, step="profile.competences")
        comps = prof.get("competences")
        if not isinstance(comps, list):
            _fail("profile.competences", "competences manquant")
        _ok("profile.competences", f"competences={len(comps)}")

    print("\n🎉 Smoke test terminé avec succès.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
        sys.exit(130)
