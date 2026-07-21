#!/usr/bin/env python3
"""
patch_guest_limit.py

Adds a backend-enforced guest chat limit to flask_server.py.

- Signed guest token (HMAC-SHA256, stdlib only) carries the usage count,
  so it survives Render restarts with no database.
- After GUEST_CHAT_LIMIT messages, /chat returns a signup nudge instead
  of calling Claude.
- Requests carrying a valid X-API-Key (real users, once BACKEND_API_KEY
  is enabled) skip the guest limit entirely.

Run from the repo root:  python patch_guest_limit.py
Idempotent: running twice won't double-apply.
"""

import re, sys, io

FILE = "flask_server.py"

with io.open(FILE, "r", encoding="utf-8") as f:
    src = f.read()

if "GUEST_CHAT_LIMIT" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

# ── 1. Insert the guest-token helpers just before CHAT_SYSTEM ──────────────
helpers = '''# ── Guest chat limiting ────────────────────────────────────────────────────
# A signed token carries its own usage count, so we need no database and it
# survives a Render restart. The token is HMAC-signed with GUEST_TOKEN_SECRET
# (falls back to ANTHROPIC_API_KEY so it's never unsigned in practice). A
# request that already carries a valid X-API-Key is a real user and skips
# this entirely.
import hmac, hashlib, base64

GUEST_CHAT_LIMIT = int(os.environ.get("GUEST_CHAT_LIMIT", "3"))
_GUEST_SECRET = (os.environ.get("GUEST_TOKEN_SECRET")
                 or os.environ.get("ANTHROPIC_API_KEY", "")
                 or "serc-guest-fallback-secret").encode("utf-8")


def _guest_sign(count):
    body = str(int(count)).encode("utf-8")
    sig = hmac.new(_GUEST_SECRET, body, hashlib.sha256).digest()[:16]
    token = base64.urlsafe_b64encode(body + b"." + sig).decode("ascii")
    return token


def _guest_count_from_token(token):
    """Returns the validated count carried by a guest token, or 0 if the
    token is missing/tampered/malformed (fail closed to a fresh guest)."""
    if not token:
        return 0
    try:
        rawpad = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(rawpad.encode("ascii"))
        body, sig = raw.split(b".", 1)
        expected = hmac.new(_GUEST_SECRET, body, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(sig, expected):
            return 0
        return int(body.decode("utf-8"))
    except Exception:
        return 0


def _is_real_user():
    """True when a valid API key is present (auth is configured AND matches).
    Such requests bypass the guest limit."""
    configured_key = os.environ.get("BACKEND_API_KEY")
    if not configured_key:
        return False
    return request.headers.get("X-API-Key", "") == configured_key


'''

anchor = "CHAT_SYSTEM = "
idx = src.find(anchor)
if idx == -1:
    print("ERROR: could not find CHAT_SYSTEM anchor. Aborting, no changes written.")
    sys.exit(1)
src = src[:idx] + helpers + src[idx:]

# ── 2. Replace the chat() body with a guest-aware version ──────────────────
old_chat = '''@app.route("/chat", methods=["POST", "OPTIONS"])
@require_api_key
def chat():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json(silent=True) or {}
    history = data.get("messages")
    if not history:
        single = (data.get("message") or "").strip()
        history = [{"role": "user", "content": single}] if single else []
    clean = [m for m in history
             if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    if not clean:
        return jsonify({"reply": "What would you like to work on?"})

    try:
        reply = _claude_complete(CHAT_SYSTEM, clean, max_tokens=1024)
        return jsonify({"reply": reply})
    except RuntimeError as e:
        return jsonify({"reply": "(Layla backend missing ANTHROPIC_API_KEY)"}), 500
    except Exception as e:
        return jsonify({"reply": "Error: " + str(e)}), 500'''

new_chat = '''@app.route("/chat", methods=["POST", "OPTIONS"])
@require_api_key
def chat():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json(silent=True) or {}
    history = data.get("messages")
    if not history:
        single = (data.get("message") or "").strip()
        history = [{"role": "user", "content": single}] if single else []
    clean = [m for m in history
             if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    if not clean:
        return jsonify({"reply": "What would you like to work on?"})

    # ── Guest limit ─────────────────────────────────────────────────────
    # Real users (valid X-API-Key) are unlimited. Guests get GUEST_CHAT_LIMIT
    # messages, tracked via a signed token echoed back on each reply.
    real_user = _is_real_user()
    guest_used = 0
    if not real_user:
        token = request.headers.get("X-Guest-Token", "") or data.get("guest_token", "")
        guest_used = _guest_count_from_token(token)
        if guest_used >= GUEST_CHAT_LIMIT:
            return jsonify({
                "reply": ("You've used your " + str(GUEST_CHAT_LIMIT) +
                          " free Layla messages. Sign up for a free account to keep going — "
                          "you'll get the full PCB Workspace with unlimited chats, the robot "
                          "control layer, and vision tools."),
                "limit_reached": True,
                "guest_used": guest_used,
                "guest_limit": GUEST_CHAT_LIMIT,
            }), 200

    try:
        reply = _claude_complete(CHAT_SYSTEM, clean, max_tokens=1024)
        resp = {"reply": reply}
        if not real_user:
            new_count = guest_used + 1
            resp["guest_token"] = _guest_sign(new_count)
            resp["guest_used"] = new_count
            resp["guest_limit"] = GUEST_CHAT_LIMIT
            resp["guest_remaining"] = max(0, GUEST_CHAT_LIMIT - new_count)
        return jsonify(resp)
    except RuntimeError as e:
        return jsonify({"reply": "(Layla backend missing ANTHROPIC_API_KEY)"}), 500
    except Exception as e:
        return jsonify({"reply": "Error: " + str(e)}), 500'''

if old_chat not in src:
    print("ERROR: chat() route didn't match the expected source exactly.")
    print("       No changes written. (Has the file been edited since?)")
    sys.exit(1)

src = src.replace(old_chat, new_chat)

with io.open(FILE, "w", encoding="utf-8") as f:
    f.write(src)

print("OK — patched flask_server.py")
print("  * Added guest-token helpers (signed, stdlib only)")
print("  * /chat now enforces GUEST_CHAT_LIMIT (default 3) for guests")
print("  * Valid X-API-Key requests bypass the limit")
print()
print("Optional env vars on Render:")
print("  GUEST_CHAT_LIMIT   (default 3)")
print("  GUEST_TOKEN_SECRET (defaults to ANTHROPIC_API_KEY if unset)")
