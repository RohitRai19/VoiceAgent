"""
Salesforce Cases MCP server (stdio) for this app.

Tools for support Cases (create, get, search, update, comment, picklist values)
plus read-only SOQL. It logs in with this app's own Salesforce settings from .env:

  - a Connected App: SF_LOGIN_URL + SF_CLIENT_ID + SF_CLIENT_SECRET, or
  - a Salesforce CLI login alias: SF_ORG_ALIAS (sf org login web -a <alias>).

Run standalone (e.g. to plug into Claude Desktop or any MCP client):

    python salesforce_case_server.py

The chat app starts it automatically through mcp_client.py.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

ORG_ALIAS = os.getenv("SF_ORG_ALIAS", "").strip()
API_VERSION = os.getenv("SF_API_VERSION", "v62.0")
LOGIN_URL = os.getenv("SF_LOGIN_URL", "").rstrip("/")
CLIENT_ID = os.getenv("SF_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("SF_CLIENT_SECRET", "")

TOKEN_TTL_S = 15 * 60

server = MCPServer(
    "salesforce-cases",
    instructions="Create, read and update Salesforce support Cases in the configured org.",
)

_auth: dict[str, Any] = {"token": None, "url": None, "at": 0.0}


def _sf_cli() -> str:
    return shutil.which("sf.cmd" if sys.platform == "win32" else "sf") or shutil.which("sf") or "sf"


def _login(force: bool = False) -> tuple[str, str]:
    """Access token + instance URL. Connected App (client credentials) when SF_CLIENT_ID is set,
    otherwise the Salesforce CLI's saved login."""
    if not force and _auth["token"] and time.time() - _auth["at"] < TOKEN_TTL_S:
        return _auth["token"], _auth["url"]
    if CLIENT_ID and CLIENT_SECRET and LOGIN_URL:
        resp = requests.post(
            f"{LOGIN_URL}/services/oauth2/token",
            data={"grant_type": "client_credentials",
                  "client_id": CLIENT_ID,
                  "client_secret": CLIENT_SECRET},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Connected App login failed: {resp.text[:300]}")
        data = resp.json()
        _auth.update(token=data["access_token"], url=data["instance_url"].rstrip("/"), at=time.time())
        return _auth["token"], _auth["url"]


    proc = subprocess.run(
        [_sf_cli(), "org", "display", "-o", ORG_ALIAS, "--json"],
        capture_output=True, text=True, timeout=60,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    result = payload.get("result") or {}
    token, url = result.get("accessToken"), result.get("instanceUrl")
    if not token or not url:
        msg = payload.get("message") or proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"Salesforce CLI could not provide a session for org '{ORG_ALIAS}'. "
            f"Run: sf org login web -a {ORG_ALIAS}. Details: {msg[:300]}"
        )
    _auth.update(token=token, url=url.rstrip("/"), at=time.time())
    return token, _auth["url"]


LOGIN_HINT = (f"The Salesforce session for org '{ORG_ALIAS}' has expired. "
              f"The administrator must run `sf org login web -a {ORG_ALIAS}` on the server, then retry.")


def _safe(fn):
    """Return errors as text so the calling LLM can read and relay them."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            text = str(exc)
            if "INVALID_SESSION_ID" in text or "could not provide a session" in text:
                return f"Error: {LOGIN_HINT}"
            return f"Error: {text}"
        except requests.RequestException as exc:
            return f"Error: could not reach Salesforce: {exc}"
    return wrapper


def _request(method: str, path: str, *, params=None, body=None, _retry=True):
    token, base = _login()
    resp = requests.request(
        method,
        f"{base}/services/data/{API_VERSION}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params=params,
        json=body,
        timeout=30,
    )
    if resp.status_code == 401 and _retry:
        _login(force=True)
        return _request(method, path, params=params, body=body, _retry=False)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Salesforce {resp.status_code}: {json.dumps(detail)[:600]}")
    return resp.json() if resp.content else {}


def _strip(node):
    """Drop Salesforce's 'attributes' metadata from records and nested relationship records."""
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k != "attributes"}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _soql(query: str) -> list[dict]:
    data = _request("GET", "/query", params={"q": query})
    records = _strip(data.get("records", []))
    if not records and "count()" in query.lower().replace(" ", ""):
        return [{"count": data.get("totalSize", 0)}]
    return records


def _resolve_case_id(case_number_or_id: str) -> str:
    value = case_number_or_id.strip()
    if len(value) in (15, 18) and value.startswith("500"):
        return value
    rows = _soql(f"SELECT Id FROM Case WHERE CaseNumber = '{value.zfill(8)}' LIMIT 1")
    if not rows:
        raise RuntimeError(f"No Case with number {value}")
    return rows[0]["Id"]


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


CASE_FIELDS = "Id, CaseNumber, Subject, Status, Priority, Origin, Type, Reason, Description, " \
              "Contact.Name, Account.Name, Owner.Name, CreatedDate, LastModifiedDate"


# ---- tools --------------------------------------------------------------------

@server.tool()
@_safe
def create_case(
    subject: str,
    description: str = "",
    priority: str = "Medium",
    origin: str = "Web",
    status: str = "New",
    type: str = "",
    contact_email: str = "",
    account_name: str = "",
) -> str:
    """Create a new Salesforce support Case. priority is Low, Medium or High.
    origin is Web, Phone or Email. If contact_email matches a Contact, the Case is linked to it;
    if account_name matches an Account, the Case is linked to it. Returns the new CaseNumber and Id."""
    body: dict[str, Any] = {
        "Subject": subject,
        "Description": description or None,
        "Priority": priority,
        "Origin": origin,
        "Status": status,
    }
    if type:
        body["Type"] = type
    if contact_email:
        rows = _soql(f"SELECT Id FROM Contact WHERE Email = '{_esc(contact_email)}' LIMIT 1")
        if rows:
            body["ContactId"] = rows[0]["Id"]
    if account_name:
        rows = _soql(f"SELECT Id FROM Account WHERE Name = '{_esc(account_name)}' LIMIT 1")
        if rows:
            body["AccountId"] = rows[0]["Id"]
    body = {k: v for k, v in body.items() if v is not None}
    created = _request("POST", "/sobjects/Case", body=body)
    case = _soql(f"SELECT {CASE_FIELDS} FROM Case WHERE Id = '{created['id']}'")[0]
    return json.dumps(case, default=str)


@server.tool()
@_safe
def get_case(case_number_or_id: str) -> str:
    """Fetch one Case by its CaseNumber (e.g. 00001026) or 18-character Id, including its comments."""
    case_id = _resolve_case_id(case_number_or_id)
    case = _soql(f"SELECT {CASE_FIELDS} FROM Case WHERE Id = '{case_id}'")[0]
    case["Comments"] = _soql(
        f"SELECT CommentBody, CreatedBy.Name, CreatedDate FROM CaseComment "
        f"WHERE ParentId = '{case_id}' ORDER BY CreatedDate DESC LIMIT 20"
    )
    return json.dumps(case, default=str)


@server.tool()
@_safe
def search_cases(
    status: str = "",
    priority: str = "",
    subject_contains: str = "",
    contact_email: str = "",
    limit: int = 10,
) -> str:
    """List Cases filtered by status (New, Working, Escalated, Closed), priority, a word in the subject,
    or the contact's email. All filters optional. Newest first."""
    where = []
    if status:
        where.append(f"Status = '{_esc(status)}'")
    if priority:
        where.append(f"Priority = '{_esc(priority)}'")
    if subject_contains:
        where.append(f"Subject LIKE '%{_esc(subject_contains)}%'")
    if contact_email:
        where.append(f"Contact.Email = '{_esc(contact_email)}'")
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    limit = max(1, min(int(limit), 50))
    rows = _soql(f"SELECT {CASE_FIELDS} FROM Case{clause} ORDER BY CreatedDate DESC LIMIT {limit}")
    return json.dumps(rows, default=str)


@server.tool()
@_safe
def update_case(
    case_number_or_id: str,
    status: str = "",
    priority: str = "",
    subject: str = "",
    description: str = "",
) -> str:
    """Update a Case's status (New, Working, Escalated, Closed), priority, subject or description.
    Only the fields you pass are changed."""
    body = {k: v for k, v in {
        "Status": status, "Priority": priority, "Subject": subject, "Description": description,
    }.items() if v}
    if not body:
        return "Nothing to update: pass at least one field."
    case_id = _resolve_case_id(case_number_or_id)
    _request("PATCH", f"/sobjects/Case/{case_id}", body=body)
    case = _soql(f"SELECT {CASE_FIELDS} FROM Case WHERE Id = '{case_id}'")[0]
    return json.dumps(case, default=str)


@server.tool()
@_safe
def add_case_comment(case_number_or_id: str, comment: str, public: bool = False) -> str:
    """Add a comment to a Case. public=True makes it visible to the customer in portals."""
    case_id = _resolve_case_id(case_number_or_id)
    created = _request("POST", "/sobjects/CaseComment", body={
        "ParentId": case_id, "CommentBody": comment, "IsPublished": public,
    })
    return json.dumps({"ok": True, "comment_id": created.get("id"), "case_id": case_id})


@server.tool()
@_safe
def case_picklist_values() -> str:
    """Return the valid Status, Priority, Origin and Type values for Cases in this org."""
    desc = _request("GET", "/sobjects/Case/describe")
    wanted = {"Status", "Priority", "Origin", "Type", "Reason"}
    out = {
        f["name"]: [v["value"] for v in f.get("picklistValues", []) if v.get("active")]
        for f in desc.get("fields", []) if f["name"] in wanted
    }
    return json.dumps(out)


@server.tool()
@_safe
def run_soql(query: str, limit: int = 50) -> str:
    """Run any read-only SOQL query against the org (Accounts, Contacts, Opportunities, custom objects...).
    For row queries a LIMIT is appended if missing (limit, max 200). Aggregate queries such as
    SELECT COUNT() ... are sent as-is; do not pass limit for them. Returns the records as JSON."""
    q = query.strip().rstrip(";")
    low = q.lower()
    if not low.startswith("select"):
        return "Error: only SELECT queries are allowed."
    # COUNT()/SUM()/... with LIMIT either errors (COUNT(Id)) or silently caps the
    # result (COUNT() ... LIMIT 1 returns 1), so never add one to aggregates.
    is_aggregate = bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", low)) or " group by " in low
    if " limit " not in low and not is_aggregate:
        q += f" LIMIT {max(1, min(int(limit), 200))}"
    return json.dumps(_soql(q), default=str)


if __name__ == "__main__":
    server.run(transport="stdio")
