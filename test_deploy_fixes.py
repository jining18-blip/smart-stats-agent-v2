# -*- coding: utf-8 -*-
"""2026-10-01 배포 전 검증에서 고친 오류들의 회귀 테스트."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


def _rcbd_numeric_blocks():
    rng = np.random.default_rng(11)
    blk = {1: -25, 2: 0, 3: 28}
    rows = [{"처리": t, "반복": b, "수량": 500 + e + blk[b] + rng.normal(0, 6)}
            for b in (1, 2, 3) for t, e in [("T1", 0), ("T2", 8), ("T3", 12), ("T4", 5)]]
    return pd.DataFrame(rows)


def _app(df, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["files"] = {"d": df.copy()}
    at.session_state["cur_key"] = "d"
    at.session_state["df"] = df.copy()
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def test_anova_screen_uses_numeric_block_column():
    """반복을 1,2,3으로 적어도 분산분석 화면이 반복을 블록으로 잡고 수량을 측정값으로 골라야 한다."""
    at = _app(_rcbd_numeric_blocks(), menu_choice="📊 통계분석", stat_sub="📈 분산분석")
    sel = {s.label[:5]: s.value for s in at.main.selectbox}
    assert sel["측정값"] == "수량"
    assert sel["반복(블록"] == "반복"
    [b for b in at.main.button if b.key == "__btn_anova"][0].click().run()
    assert not at.exception
    aov = [d.value for d in at.main.dataframe if "PR(>F)" in d.value.columns][0]
    p_trt = float(aov.loc[[i for i in aov.index if "처리" in i][0], "PR(>F)"])
    assert p_trt == pytest.approx(0.0372, abs=1e-3)   # CRD로 잘못 분석하면 0.839


def test_two_way_without_replication_is_blocked_not_crashing():
    df = pd.DataFrame({"A": np.repeat(list("abc"), 2), "B": ["x", "y"] * 3,
                       "y": [1.0, 2.0, 3.0, 2.5, 4.0, 1.5]})
    at = _app(df, menu_choice="📊 통계분석", stat_sub="📈 분산분석")
    at.main.radio[1].set_value("이원배치 (요인 2개 + 상호작용)").run()
    [b for b in at.main.button if "이원배치" in b.label][0].click().run()
    assert not at.exception
    assert any("반복이 2개 이상" in e.value for e in at.main.error)


def test_repeated_measures_same_column_is_blocked_not_crashing():
    df = pd.DataFrame({"품종": np.repeat(["A", "B"], 4), "구분": ["x"] * 8,
                       "y": np.arange(8, dtype=float)})
    at = _app(df, menu_choice="📊 통계분석", stat_sub="📈 분산분석")
    at.main.radio[1].set_value("🔁 반복측정 (같은 개체 시기별 조사)").run()
    [b for b in at.main.button if "반복측정" in b.label][0].click().run()
    assert not at.exception


def test_engine_defers_land_rent_duplicate_check_to_ui():
    """'임차료' 열을 장비 임차료로 분류하면 UI가 통과시키므로 엔진이 다시 막으면 안 된다."""
    import sys
    sys.path.insert(0, str(ROOT))
    from economic_core import calculate_row_economics
    d = pd.DataFrame({"처리": ["A", "A", "B", "B"], "수량": [100, 110, 120, 125],
                      "단가": [1000] * 4, "비료비": [1, 2, 3, 4], "임차료": [50, 50, 60, 60]})
    row, _ = calculate_row_economics(d, "처리", "수량", "단가", ["비료비", "임차료"],
                                     land_cash_rent_per_10a=10000)
    assert float(row["_경영비"].iloc[0]) == pytest.approx(1 + 50 + 10000)
