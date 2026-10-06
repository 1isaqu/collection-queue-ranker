import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from collection_queue_ranker import synth
from collection_queue_ranker.contract import CONTRACT
from collection_queue_ranker.db.sqlite import SqliteSource
from collection_queue_ranker.enrichment import (
    CachedProvider,
    NullProvider,
    TtlCache,
    enrich_frame,
    make_enricher,
)
from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.serving.app import create_app, score_rows
from collection_queue_ranker.snapshots import build_snapshots

# ---- enrichment ---------------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, missing=("auto|99",)):
        self.calls, self.missing = [], set(missing)

    def fetch(self, keys):
        self.calls.append(list(keys))
        return {k: {"external_score": 100 + int(k.split("|")[1]), "external_rating": "B"}
                for k in keys if k not in self.missing}


FRAME = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2, 2, 99]})


def test_enrich_frame_asks_once_per_debtor_and_leaves_gaps_missing():
    provider = FakeProvider()
    out = enrich_frame(FRAME, provider)
    assert provider.calls == [["auto|1", "auto|2", "auto|99"]]
    assert out["external_score"].tolist()[:3] == [101.0, 102.0, 102.0]
    assert pd.isna(out["external_score"].iloc[3]) and out["external_score"].dtype.kind == "f"
    assert out["external_rating"].tolist()[:2] == ["B", "B"] and pd.isna(out["external_rating"].iloc[3])
    assert "external_score" not in FRAME   # the input is not modified


def test_the_null_provider_adds_empty_columns():
    out = enrich_frame(FRAME, NullProvider())
    assert out["external_score"].isna().all()


def test_the_cache_honours_its_time_to_live(tmp_path):
    clock = {"now": 1_000.0}
    cache = TtlCache(tmp_path / "c.json", ttl_days=1, clock=lambda: clock["now"])
    cache.put_many({"k": {"v": 1}})
    assert cache.get("k") == {"v": 1}
    clock["now"] += 86_399
    assert cache.get("k") == {"v": 1}
    clock["now"] += 2
    assert cache.get("k") is None and cache.get("never") is None


def test_the_cache_is_persisted_between_runs(tmp_path):
    cache = TtlCache(tmp_path / "sub" / "c.json", ttl_days=30)
    cache.put_many({"k": {"v": 1}})
    cache.save()
    assert TtlCache(tmp_path / "sub" / "c.json", ttl_days=30).get("k") == {"v": 1}
    json.loads((tmp_path / "sub" / "c.json").read_text(encoding="utf-8"))


def test_a_cached_provider_calls_the_vendor_only_for_unknown_keys(tmp_path):
    inner = FakeProvider()
    cached = CachedProvider(inner, TtlCache(tmp_path / "c.json", ttl_days=30))
    first = cached.fetch(["auto|1", "auto|2"])
    second = cached.fetch(["auto|2", "auto|3"])
    assert inner.calls == [["auto|1", "auto|2"], ["auto|3"]]
    assert second["auto|2"] == first["auto|2"] and cached.calls == 2
    cached.fetch(["auto|1", "auto|2", "auto|3"])
    assert cached.calls == 2   # everything answered from the cache


def test_the_enricher_hook_adds_columns_to_every_snapshot(demo, demo_settings):
    settings = demo_settings.with_overrides({"features": {"extra_numeric": ["external_score"]}})
    provider = FakeProvider(missing=())
    snaps = build_snapshots(demo["raw"], demo["prep"], settings, demo["as_of"],
                            enricher=make_enricher(provider))
    assert "external_score" in snaps and len(provider.calls) == snaps["snapshot_at"].nunique()
    assert snaps["external_score"].notna().all()


# ---- synthetic data -----------------------------------------------------------------------------


def test_the_generator_is_deterministic_and_follows_the_contract():
    a, b = synth.generate(80, seed=11, months=10), synth.generate(80, seed=11, months=10)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
        assert set(CONTRACT[name].required) <= set(a[name].columns)
    c = synth.generate(80, seed=12, months=10)
    assert not a["contact_events"].equals(c["contact_events"])


def test_the_generated_world_is_internally_consistent():
    f = synth.generate(300, seed=3, months=14)
    assert f["debts"]["debtor_id"].isin(f["debtors"]["debtor_id"]).all()
    assert f["contact_events"]["phone_id"].isin(f["debtor_phones"]["phone_id"]).all()
    assert f["contact_events"]["occurred_at"].max() <= pd.Timestamp(synth.DEFAULT_END) + pd.Timedelta(days=1)
    first = f["agreements"][f["agreements"]["installment_no"] == 0]
    assert first["status_id"].isin(f["agreement_statuses"]["status_id"]).all()
    paid = f["agreements"][f["agreements"]["paid_amount"] > 0]
    assert paid["paid_at"].notna().all() and (paid["paid_at"] >= paid["issued_at"]).all()
    assert (f["agreements"]["paid_at"].dropna() <= pd.Timestamp(synth.DEFAULT_END)).all()
    assert set(f["agents"]["login"]) >= {"autodialer_01", "test_user"}


def test_a_debtor_who_signs_does_so_after_the_debt_entered_and_an_answered_call():
    f = synth.generate(400, seed=5, months=14)
    first = f["agreements"][f["agreements"]["installment_no"] == 0]
    first = first[first["status_id"] != 6]
    debts = f["debts"].groupby("debtor_id")["entered_at"].min()
    ok = first.assign(entered=first["debtor_id"].map(debts))
    legacy = ok["issued_at"] < ok["entered"]
    assert legacy.mean() < 0.5 and (~legacy).any()   # a few legacy agreements, most are not


def test_the_database_writer_round_trips_through_the_sqlite_adapter(tmp_path):
    frames = synth.generate(60, seed=2, months=8)
    path = synth.write_sqlite(frames, tmp_path / "d" / "x.db")
    source = SqliteSource("auto", path)
    assert set(source.table_columns("contact_events")) >= set(CONTRACT["contact_events"].required)
    back = source.query("SELECT COUNT(*) AS n FROM debts")
    assert back["n"].iloc[0] == len(frames["debts"])
    synth.write_sqlite(frames, path)   # overwriting an existing database is allowed


def test_the_demo_builder_writes_one_database_per_line_with_overlapping_ids(tmp_path):
    paths = synth.build_demo(tmp_path, n_debtors=60, seed=1)
    assert set(paths) == {"auto", "consumer"} and all(p.is_file() for p in paths.values())
    a = SqliteSource("auto", paths["auto"]).query("SELECT MIN(debtor_id) AS m FROM debtors")
    b = SqliteSource("consumer", paths["consumer"]).query("SELECT MIN(debtor_id) AS m FROM debtors")
    assert a["m"].iloc[0] == b["m"].iloc[0] == 1   # ids restart in each line on purpose


# ---- serving ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(trained):
    result, out = trained
    return TestClient(create_app(out / "bundle.joblib")), result


def _rows(result, n=6):
    frame = result.split.holdout.head(n)
    columns = [c for c in frame.columns if c != "debtor_id"]
    return json.loads(frame[columns].to_json(orient="records", date_format="iso"))


def test_health_and_model_endpoints(client):
    http, result = client
    assert http.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = http.get("/model").json()
    assert info["holdout_start"] == result.bundle.meta["holdout_start"]
    assert "settings" not in info and "balance" in info["features"]["close"]
    assert info["features"]["ranker"]


def test_score_returns_probabilities_values_and_a_ranking(client):
    http, result = client
    response = http.post("/score", json={"rows": _rows(result)})
    assert response.status_code == 200
    scores = response.json()["scores"]
    assert len(scores) == 6 and sorted(s["rank"] for s in scores) == [1, 2, 3, 4, 5, 6]
    assert all(0 <= s["p_close"] <= 1 and 0 <= s["p_break"] <= 1 for s in scores)
    top = min(scores, key=lambda s: s["rank"])
    assert top["ranker_score"] == max(s["ranker_score"] for s in scores)


def test_score_without_a_contact_factor_assumes_no_adjustment(client):
    _, result = client
    rows = [{k: v for k, v in row.items() if k != "contact_factor"} for row in _rows(result, 3)]
    scored = score_rows(result.bundle, rows)
    assert all(s["adjusted_value"] == pytest.approx(s["expected_value"]) for s in scored)


def test_missing_feature_columns_are_a_422_that_names_them(client):
    http, result = client
    row = {k: v for k, v in _rows(result, 1)[0].items() if k not in ("balance", "answer_rate")}
    response = http.post("/score", json={"rows": [row]})
    assert response.status_code == 422
    assert "balance" in response.json()["detail"] and "answer_rate" in response.json()["detail"]
    assert http.post("/score", json={"rows": []}).status_code == 422


def test_an_app_without_a_bundle_degrades_to_503():
    http = TestClient(create_app())
    assert http.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert http.get("/model").status_code == 503
    assert http.post("/score", json={"rows": [{"a": 1}]}).status_code == 503


def test_serving_without_the_extra_installed_is_a_clear_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("no fastapi")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ConfigError, match="serve"):
        create_app()
