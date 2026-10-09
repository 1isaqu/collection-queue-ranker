"""The documentation makes claims about the code. These tests keep the claims true."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from collection_queue_ranker.cli import build_parser
from collection_queue_ranker.config import Settings
from collection_queue_ranker.contract import CONTRACT

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOCS = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md"))}

# Documented in prose under the table rather than as one row each.
PROSE_SETTINGS = {
    "model.close_params", "model.break_params", "model.ranker_params",
    "source.sources", "source.tables",
}


def _sections(text: str) -> dict[str, tuple[str, str]]:
    """`## `table` note` headings -> (note, body up to the next `## ` heading)."""
    found = {}
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    for part in parts:
        head, _, body = part.partition("\n")
        match = re.match(r"`(\w+)`(.*)", head)
        if match:
            found[match.group(1)] = (match.group(2).strip(), body)
    return found


def test_the_data_contract_page_lists_exactly_the_tables_in_the_code():
    sections = _sections(DOCS["data-contract.md"])
    assert set(sections) == set(CONTRACT)
    assert f"reads {_number_word(len(CONTRACT))} tables" in DOCS["data-contract.md"]


@pytest.mark.parametrize("table", sorted(CONTRACT))
def test_each_table_section_matches_required_and_optional_columns(table):
    note, body = _sections(DOCS["data-contract.md"])[table]
    rows = re.findall(r"^\| `(\w+)` \| (required|optional) \|", body, flags=re.M)
    spec = CONTRACT[table]
    assert {c for c, need in rows if need == "required"} == set(spec.required)
    assert {c for c, need in rows if need == "optional"} == set(spec.optional)
    assert ("optional table" in note) == spec.optional_table


def _number_word(n: int) -> str:
    return {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}[n]


def _walk(value: dict, path: str = "") -> dict[str, object]:
    flat = {}
    for key, item in value.items():
        name = f"{path}{key}"
        if isinstance(item, dict) and name not in PROSE_SETTINGS:
            flat.update(_walk(item, f"{name}."))
        else:
            flat[name] = item
    return flat


def _documented_defaults() -> dict[str, object]:
    section = README.split("\n## Configuration\n", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `([a-z_0-9]+\.[a-z_0-9]+)` \| `([^|]*)` \|", section, flags=re.M)
    return {key: json.loads(default) for key, default in rows}


def test_the_readme_configuration_table_matches_the_defaults_in_code():
    actual = _walk(json.loads(json.dumps(Settings().to_dict())))
    documented = _documented_defaults()
    expected = {k: v for k, v in actual.items() if k not in PROSE_SETTINGS}
    assert documented == expected


def _subcommands() -> set[str]:
    parser = build_parser()
    action = next(a for a in parser._actions if getattr(a, "choices", None))
    return set(action.choices)


def _commands_in(text: str) -> set[str]:
    return set(re.findall(r"(?:`|^[ \t]*)cqr ([a-z][a-z-]*)", text, flags=re.M))


@pytest.mark.parametrize("name", ["README.md", *sorted(DOCS)])
def test_every_command_a_page_mentions_exists(name):
    text = README if name == "README.md" else DOCS[name]
    assert _commands_in(text) <= _subcommands()


def test_the_readme_mentions_every_command():
    named = {c for c in _subcommands() if f"cqr {c}" in README or f"`{c}`" in README}
    assert named == _subcommands()


def test_paths_named_in_the_docs_exist():
    pages = {"README.md": README, **DOCS}
    pattern = re.compile(r"`((?:docs|examples|src|tests)/[\w./-]+?)`")
    missing = [
        f"{page}: {path}" for page, text in pages.items() for path in pattern.findall(text)
        if not (ROOT / path.rstrip(".")).exists()
    ]
    assert missing == []


def test_every_variable_in_the_env_example_is_read_by_the_code():
    names = re.findall(r"^([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(encoding="utf-8"),
                       flags=re.M)
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    unread = []
    for name in names:
        if name.startswith("CQR_DB_"):
            used = re.search(rf"_get\(.{name.removeprefix('CQR_DB_')}.", source) is not None
        else:
            used = name in source
        if not used:
            unread.append(name)
    assert unread == []
    assert "change-me" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_the_example_settings_file_loads_and_matches_the_defaults_it_restates():
    from collection_queue_ranker.config import load_settings

    loaded = load_settings(ROOT / "examples" / "settings.example.json")
    assert loaded.labels == Settings().labels
    assert loaded.queue == Settings().queue
