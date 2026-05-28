"""HMAC signing for Delta Exchange.

These tests verify the canonical signing string, signature stability, and
header construction. They use Delta's own published example shape so the
signature output is deterministic and can be eyeballed.
"""
from __future__ import annotations

import hashlib
import hmac

import pytest

from aera.markets.delta_signing import (
    auth_headers,
    canonical_body,
    canonical_query_string,
    sign_payload,
    sign_request,
    ws_auth_payload,
)


SECRET = "test_secret_abcdef0123456789"
KEY = "test_api_key_xyz"


def test_sign_payload_matches_raw_hmac():
    data = "GET1700000000/v2/wallet/balances"
    expected = hmac.new(
        SECRET.encode("utf-8"), data.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert sign_payload(SECRET, data) == expected


def test_canonical_query_string_is_deterministic():
    qs = canonical_query_string({"b": "2", "a": "1"})
    assert qs == "?a=1&b=2"
    assert canonical_query_string(None) == ""
    assert canonical_query_string({}) == ""


def test_canonical_body_serialises_dicts_compact_and_sorted():
    body = {"b": 2, "a": 1}
    assert canonical_body(body) == '{"a":1,"b":2}'
    assert canonical_body(None) == ""
    assert canonical_body("") == ""
    assert canonical_body("raw") == "raw"


def test_sign_request_builds_signature_over_full_string():
    ts, sig, body_str = sign_request(
        api_secret=SECRET,
        method="GET",
        path="/v2/wallet/balances",
        timestamp=1_700_000_000,
    )
    assert ts == "1700000000"
    assert body_str == ""
    payload = "GET1700000000/v2/wallet/balances"
    expected = hmac.new(
        SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert sig == expected


def test_sign_request_with_query_and_body():
    ts, sig, body_str = sign_request(
        api_secret=SECRET,
        method="POST",
        path="/v2/orders",
        query={"recv_window": 5000},
        body={"product_id": 27, "size": 1, "side": "buy"},
        timestamp=1_700_000_000,
    )
    qs = "?recv_window=5000"
    expected_body = '{"product_id":27,"side":"buy","size":1}'
    assert body_str == expected_body
    payload = "POST1700000000/v2/orders" + qs + expected_body
    expected_sig = hmac.new(
        SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert sig == expected_sig


def test_auth_headers_complete_and_consistent():
    headers, body_str = auth_headers(
        api_key=KEY,
        api_secret=SECRET,
        method="POST",
        path="/v2/orders",
        body={"size": 1, "side": "buy"},
        user_agent="aera-tests/1.0",
        timestamp=1_700_000_000,
    )
    assert headers["api-key"] == KEY
    assert headers["timestamp"] == "1700000000"
    assert headers["User-Agent"] == "aera-tests/1.0"
    assert headers["Content-Type"] == "application/json"
    # signature recovers from secret + canonical payload
    expected = hmac.new(
        SECRET.encode("utf-8"),
        ("POST1700000000/v2/orders" + body_str).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert headers["signature"] == expected


def test_auth_headers_omit_content_type_when_no_body():
    headers, body_str = auth_headers(
        api_key=KEY,
        api_secret=SECRET,
        method="GET",
        path="/v2/positions/margined",
        timestamp=1_700_000_000,
    )
    assert body_str == ""
    assert "Content-Type" not in headers


def test_ws_auth_payload_structure():
    frame = ws_auth_payload(
        api_key=KEY, api_secret=SECRET, timestamp=1_700_000_000
    )
    assert frame["type"] == "auth"
    assert frame["payload"]["api-key"] == KEY
    assert frame["payload"]["timestamp"] == "1700000000"
    # signature is over "GET1700000000/live"
    expected = hmac.new(
        SECRET.encode("utf-8"),
        b"GET1700000000/live",
        hashlib.sha256,
    ).hexdigest()
    assert frame["payload"]["signature"] == expected


def test_sign_payload_rejects_non_string():
    with pytest.raises(TypeError):
        sign_payload(SECRET, 123)  # type: ignore[arg-type]
