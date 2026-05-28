"""Delta Exchange request authentication.

Delta uses HMAC-SHA256 over a deterministic concatenation of request parts:

    signature_data = method + timestamp + request_path + query_string + body
    signature      = hex( HMAC_SHA256(api_secret, signature_data) )

Headers required on every authenticated REST call::

    api-key:     <API_KEY>
    timestamp:   <UNIX_TIMESTAMP_SECONDS>
    signature:   <HEX_SIGNATURE>
    User-Agent:  <ANY_NON_EMPTY_STRING>
    Content-Type: application/json   (when there's a JSON body)

The same signature scheme is used to authenticate the websocket subscription
to private channels (orders, positions, trading_notifications).

This module is dependency-light on purpose — it only needs ``hmac`` and
``hashlib`` from the stdlib, so it's trivially unit-testable and the rest of
the bot can stay venue-agnostic.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode


def sign_payload(api_secret: str, payload: str) -> str:
    """HMAC-SHA256(api_secret, payload), hex-encoded."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    return hmac.new(
        api_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def canonical_query_string(params: Optional[Mapping[str, Any]]) -> str:
    """Delta expects ``?a=1&b=2`` (leading '?') in the signed string.

    Empty params -> empty string (NOT ``"?"``).
    """
    if not params:
        return ""
    # Keep deterministic ordering so the signature is reproducible.
    items = sorted((str(k), str(v)) for k, v in params.items() if v is not None)
    if not items:
        return ""
    return "?" + urlencode(items)


def canonical_body(body: Any) -> str:
    """Serialise a JSON body the same way it's sent over the wire."""
    if body is None or body == "":
        return ""
    if isinstance(body, (dict, list)):
        # Delta accepts any JSON; we use compact form and sign that exact string.
        return json.dumps(body, separators=(",", ":"), sort_keys=True)
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body)


def sign_request(
    *,
    api_secret: str,
    method: str,
    path: str,
    query: Optional[Mapping[str, Any]] = None,
    body: Any = None,
    timestamp: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Compute the signature for a Delta REST request.

    Returns
    -------
    (timestamp, signature, body_string)
        ``timestamp``  : unix seconds as a string (matches what goes in the header).
        ``signature``  : hex HMAC-SHA256.
        ``body_string``: the exact body that must be sent on the wire so the
                         server's signature recomputation matches.
    """
    ts = str(timestamp if timestamp is not None else int(time.time()))
    qs = canonical_query_string(query)
    body_str = canonical_body(body)
    payload = f"{method.upper()}{ts}{path}{qs}{body_str}"
    return ts, sign_payload(api_secret, payload), body_str


def auth_headers(
    *,
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
    query: Optional[Mapping[str, Any]] = None,
    body: Any = None,
    user_agent: str = "aera-trading-bot/0.1",
    timestamp: Optional[int] = None,
) -> Tuple[Dict[str, str], str]:
    """Build the dict of HTTP headers + the canonical body string.

    The caller MUST send ``body_str`` verbatim — passing ``json={...}`` to httpx
    would re-serialise the dict and could break the signature.
    """
    ts, sig, body_str = sign_request(
        api_secret=api_secret,
        method=method,
        path=path,
        query=query,
        body=body,
        timestamp=timestamp,
    )
    headers = {
        "api-key": api_key,
        "timestamp": ts,
        "signature": sig,
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if body_str:
        headers["Content-Type"] = "application/json"
    return headers, body_str


def ws_auth_payload(
    *,
    api_key: str,
    api_secret: str,
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Authentication frame for the Delta websocket (private channels).

    The signed payload mirrors the REST format with method=``GET`` and a
    fixed sentinel path ``"/live"`` — this is what the Delta docs specify.
    """
    ts, sig, _ = sign_request(
        api_secret=api_secret,
        method="GET",
        path="/live",
        timestamp=timestamp,
    )
    return {
        "type": "auth",
        "payload": {
            "api-key": api_key,
            "signature": sig,
            "timestamp": ts,
        },
    }
