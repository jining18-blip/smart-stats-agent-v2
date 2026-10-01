# -*- coding: utf-8 -*-
"""통계 핵심 로직 검증.

app.py는 스트림릿 앱이라 그대로 import할 수 없으므로, 순수 함수 구간만 잘라서
실행한 뒤 검증한다. 유의성 문자·사후검정·설계 자동판별은 이 도구의 간판
기능이므로 회귀(regression)를 반드시 잡아야 한다.
"""
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

APP = Path(__file__).resolve().parents[1] / "app.py"


def _load(start_marker, end_marker):
    src = APP.read_text(encoding="utf-8")
    seg = src[src.index(start_marker):src.index(end_marker)]
    ns = {"np": np, "pd": pd, "stats": stats,
          "studentized_range": studentized_range}
    exec(seg, ns)
    return ns


@pytest.fixture(scope="module")
def fns():
    return _load("def compact_letter_display", "def posthoc_not_sig")


@pytest.fixture(scope="module")
def design():
    return _load("_BLOCK_KEYS =", "def _josa")


# ---------------------------------------------------------------- 유의성 문자
def test_cld_basic_chain(fns):
    cld = fns["compact_letter_display"]
    res = cld(list("ABC"), {frozenset({"A", "B"}), frozenset({"B", "C"})})
    assert res == {"A": "a", "B": "ab", "C": "b"}


def test_cld_all_different(fns):
    cld = fns["compact_letter_display"]
    assert cld(list("ABC"), set()) == {"A": "a", "B": "b", "C": "c"}


def test_cld_all_same(fns):
    cld = fns["compact_letter_display"]
    pairs = {frozenset(p) for p in itertools.combinations("ABC", 2)}
    assert cld(list("ABC"), pairs) == {"A": "a", "B": "a", "C": "a"}


def test_cld_invariant_random(fns):
    """무작위 유의성 조합에서 '문자를 공유한다 ⟺ 유의차가 없다'가 항상 성립해야 한다."""
    cld = fns["compact_letter_display"]
    rng = random.Random(20260726)
    for _ in range(300):
        groups = [chr(65 + i) for i in range(rng.randint(3, 6))]
        not_sig = {frozenset({a, b}) for a, b in itertools.combinations(groups, 2)
                   if rng.random() < 0.5}
        letters = cld(groups, not_sig)
        for a, b in itertools.combinations(groups, 2):
            shared = bool(set(letters[a]) & set(letters[b]))
            assert shared == (frozenset({a, b}) in not_sig), (groups, not_sig, letters)


# ---------------------------------------------------------------- 사후검정
@pytest.fixture(scope="module")
def crd_model():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "trt": np.repeat(list("ABCD"), 5),
        "y": np.concatenate([rng.normal(m, 2, 5) for m in (10, 12, 15, 10.5)])})
    return df, ols("y ~ C(trt)", data=df).fit()


def test_tukey_matches_statsmodels(fns, crd_model):
    df, model = crd_model
    mine = fns["posthoc_from_model"](model, df, "trt", "Tukey HSD")["table"]
    ref = pairwise_tukeyhsd(df["y"], df["trt"])
    ref_df = pd.DataFrame(ref._results_table.data[1:],
                          columns=ref._results_table.data[0])
    for _, row in mine.iterrows():
        hit = ref_df[((ref_df.group1 == row["그룹1"]) & (ref_df.group2 == row["그룹2"])) |
                     ((ref_df.group1 == row["그룹2"]) & (ref_df.group2 == row["그룹1"]))]
        assert row["p(보정)"] == pytest.approx(float(hit["p-adj"].iloc[0]), abs=1e-4)


def test_dunnett_matches_scipy(fns, crd_model):
    from scipy.stats import dunnett
    df, model = crd_model
    out = fns["posthoc_from_model"](model, df, "trt", "Dunnett", control="A")["table"]
    others = ["B", "C", "D"]
    ref = dunnett(*[df.loc[df.trt == g, "y"].values for g in others],
                  control=df.loc[df.trt == "A", "y"].values)
    for i, g in enumerate(others):
        mine = out.loc[out["처리구"] == g, "p(동시보정)"].item()
        assert mine == pytest.approx(float(ref.pvalue[i]), abs=0.01)


def test_posthoc_uses_model_error_term(fns):
    """난괴법에서는 블록을 넣은 모형의 오차를 써야 하므로 CRD와 결과가 달라야 한다."""
    rng = np.random.default_rng(11)
    block = {1: -25, 2: 0, 3: 28}
    rows = [{"반복": b, "처리": t, "y": 500 + e + block[b] + rng.normal(0, 6)}
            for b in (1, 2, 3)
            for t, e in [("T1", 0), ("T2", 14), ("T3", 26), ("T4", 8)]]
    d = pd.DataFrame(rows)
    crd = fns["posthoc_from_model"](ols("y ~ C(처리)", data=d).fit(), d, "처리", "Tukey HSD")
    rcbd = fns["posthoc_from_model"](ols("y ~ C(처리) + C(반복)", data=d).fit(),
                                     d, "처리", "Tukey HSD")
    assert len(rcbd["not_sig"]) < len(crd["not_sig"])


# ---------------------------------------------------------------- 설계 자동판별
def test_detects_rcbd_with_text_blocks(design):
    d = pd.DataFrame({"반복": np.tile(["I", "II", "III"], 4),
                      "처리": np.repeat(["T1", "T2", "T3", "T4"], 3),
                      "수량": np.random.default_rng(0).normal(500, 20, 12)})
    res = design["detect_design"](d)
    assert res["design"].startswith("난괴법")
    assert res["blk"] == "반복" and res["trt"] == "처리"


def test_detects_rcbd_with_numeric_codes(design):
    """엑셀에서 반복을 1,2,3으로 적어도 난괴법으로 인식해야 한다(예전엔 CRD로 오인)."""
    d = pd.DataFrame({"반복": np.tile([1, 2, 3], 4),
                      "처리": np.repeat([1, 2, 3, 4], 3),
                      "수량": np.random.default_rng(0).normal(500, 20, 12)})
    res = design["detect_design"](d)
    assert res["design"].startswith("난괴법")
    assert res["blk"] == "반복" and res["trt"] == "처리"
    assert "수량" in res["ys"] and "반복" not in res["ys"]


def test_integer_measurement_is_not_treated_as_code(design):
    """폭우일(7,8,10,12일)처럼 값이 띄엄띄엄한 정수 측정값은 코드로 오인하면 안 된다."""
    d = pd.DataFrame({"처리": np.repeat(["A", "B"], 4),
                      "폭우일": [7, 8, 12, 7, 13, 10, 7, 14]})
    _, _, promoted = design["split_code_columns"](d)
    assert "폭우일" not in promoted


def test_year_column_not_promoted(design):
    d = pd.DataFrame({"연도": list(range(2016, 2025)) * 2,
                      "지역": ["봉화군"] * 9 + ["영양군"] * 9,
                      "생산량": np.random.default_rng(3).normal(3000, 300, 18)})
    _, _, promoted = design["split_code_columns"](d)
    assert "생산량" not in promoted


def test_crd_without_blocks(design):
    d = pd.DataFrame({"처리": np.repeat(["A", "B", "C"], 5),
                      "수량": np.random.default_rng(1).normal(500, 20, 15)})
    res = design["detect_design"](d)
    assert res["design"].startswith("완전임의배치")
    assert res["blk"] is None
