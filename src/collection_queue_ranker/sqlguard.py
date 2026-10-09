"""Read-only guard for every statement the package sends to a database.

Searching the text for substrings (does it contain `into `?) both lets statements through and
refuses harmless text. The guard removes comments and string literals first, then looks at whole
word tokens: the statement must start with SELECT or WITH, hold a single statement, and contain no
write or administrative keyword. It is a seatbelt, not a security boundary: also connect with a
database login that can only read.
"""

from __future__ import annotations

import re

from collection_queue_ranker.errors import UnsafeSqlError

_FORBIDDEN = frozenset({
    "insert", "update", "delete", "drop", "truncate", "alter", "merge", "exec", "execute",
    "create", "grant", "revoke", "into", "declare", "set", "backup", "restore", "shutdown",
    "kill", "bulk", "openrowset", "opendatasource", "xp_cmdshell", "use", "attach", "detach",
    "pragma", "vacuum", "replace", "call", "load_extension",
})
_ALLOWED_START = frozenset({"select", "with"})

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENT = re.compile(r'"[^"]*"|\[[^\]]*\]|`[^`]*`')
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _strip(sql: str) -> str:
    """Drop comments, string literals and quoted identifiers, keeping the statement skeleton."""
    text = _BLOCK_COMMENT.sub(" ", sql)
    text = _LINE_COMMENT.sub(" ", text)
    text = _STRING.sub("''", text)
    return _QUOTED_IDENT.sub('""', text)


def assert_read_only(sql: str) -> None:
    """Raise `UnsafeSqlError` unless `sql` is one plain SELECT / WITH statement."""
    body = _strip(sql).strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if not body:
        raise UnsafeSqlError("empty statement")
    if ";" in body:
        raise UnsafeSqlError("more than one statement")
    words = [w.lower() for w in _WORD.findall(body)]
    if not words or words[0] not in _ALLOWED_START:
        raise UnsafeSqlError(f"statement must start with SELECT or WITH, not {words[:1]}")
    hits = sorted({w for w in words if w in _FORBIDDEN})
    if hits:
        raise UnsafeSqlError(f"read-only source: forbidden keyword(s) {hits}")
