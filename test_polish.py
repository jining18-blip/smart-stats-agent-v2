# -*- coding: utf-8 -*-
"""2026-10-01 배포 전 다듬기(문장·표기·UX·안전성) 회귀 테스트."""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SRC = APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ns():
    seg = SRC[SRC.index("_JOSA_DIGIT_JONG ="):SRC.index("def interpret_anova")]
    env = {"re": re, "np": np, "pd": pd, "cv_grade": lambda cv: "양호"}
    exec(seg, env)
    return env


@pytest.mark.parametrize("word,pair,expected", [
    ("엽수(개)", "은/는", "는"), ("수량(kg/10a)", "은/는", "은"), ("당도(°Brix)", "은/는", "는"),
    ("607.6", "으로/로", "으로"), ("607.1", "으로/로", "로"), ("494.5", "으로/로", "로"),
    ("153,520", "으로/로", "으로"), ("1.8%", "으로/로", "로"), ("서울", "으로/로", "로"),
    ("청양", "이/가", "이"), ("수비초", "이/가", "가"), ("'처리1'", "은/는", "은"),
    ("Tukey HSD", "으로/로", "로"), ("던컨(Duncan)", "으로/로", "으로"),
    ("ANOVA", "을/를", "를"), ("10a", "을/를", "을"), ("120kg", "이/가", "이"),
    ("완전임의배치(CRD)", "으로/로", "로"), ("난괴법(RCBD)", "으로/로", "으로"),
])
def test_josa(ns, word, pair, expected):
    assert ns["_josa"](word, pair) == expected


def test_p_text(ns):
    p = ns["_ptxt"]
    assert p(2.4e-12) == "p<0.0001"
    assert p(0.0) == "p<0.0001"
    assert p(0.03717) == "p=0.0372"
    assert p(0.0372, 3, spaced=True) == "p = 0.037"
    assert p(0.0004, 3, spaced=True) == "p < 0.001"
    assert p(float("nan")) == "p=-"


def test_report_sentence_has_correct_josa_and_p(ns):
    means = pd.DataFrame({"mean": [607.625, 494.625]}, index=["처리2", "대조구"])
    txt = ns["report_sentence_anova"]("처리구", "수량(kg/10a)", 1e-12, means,
                                      {"처리2": "a", "대조구": "b"}, {"CV": 1.8}, "Tukey HSD")
    assert "처리2가 607.6으로 가장 높았고" in txt
    assert "대조구가 494.6으로 가장 낮았다" in txt
    assert "p<0.0001" in txt and "0.0000" not in txt
    assert "Tukey HSD(p<0.05)로 실시" in txt


def test_no_zero_p_format_left():
    """문장 속 p값을 :.4f로 직접 찍으면 0.0000이 나온다 — 모두 _ptxt를 써야 한다."""
    assert not re.findall(r"p ?= ?\{[^{}]*:\.[34]f\}", SRC)


def test_no_implicit_pyplot_state():
    """동시 사용자 환경에서 남의 그림을 건드리지 않도록 plt 전역 함수 대신 fig/ax 메서드를 쓴다."""
    bad = re.findall(r"plt\.(tight_layout|xticks|yticks|title|xlabel|ylabel|legend|gca|gcf)\(", SRC)
    assert not bad, bad


def test_price_upload_reads_once():
    seg = SRC[SRC.index('key="price_up")'):SRC.index("###### 🌐 KAMIS 농산물 가격 자동 조회")]
    assert "_price_up_sig" in seg and "st.rerun()" in seg


def test_csv_extension_case_insensitive():
    assert 'if uf.name.lower().endswith(".csv"):' in SRC


def test_py310_compatible_syntax():
    """requirements 권장 Python 3.10~3.12 — 3.12 전용 f-string 문법을 쓰면 안 된다."""
    ast.parse(SRC, feature_version=(3, 10))


def _econ_app():
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    [b for b in at.button if b.label == "💰 경제성"][0].click().run()
    files = at.session_state["files"]
    k = [k for k in files if "경제성" in k][0]
    d = files[k]
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.session_state["files"] = {k: d.copy()}
    at.session_state["cur_key"] = k
    at.session_state["df"] = d.copy()
    at.session_state["menu_choice"] = "💰 경제성분석"
    at.session_state["econ_entry_mode"] = "direct"
    at.session_state["econ_mode"] = "📗 소득분석"
    at.run()
    return at


def test_econ_screen_polish():
    at = _econ_app()
    assert not at.exception
    # 단위 확인 전에는 빨간 오류가 아니라 안내로 보여 주고, 실행 버튼은 잠근다
    assert not [e for e in at.main.error if "단위" in e.value]
    assert any("단위 확인란" in i.value for i in at.main.info)
    assert [b for b in at.main.button if b.key == "__btn_econ"][0].disabled
    # 개발용 표시는 일반 사용자에게 보이지 않는다
    assert not [b for b in at.main.button if b.key == "econ_selftest"]
    assert not any("UX v3.6" in c.value for c in at.main.caption)
    assert any(n.label.startswith("농업노임") for n in at.main.number_input)


def test_no_dual_josa_placeholders():
    """'을(를)'처럼 조사를 둘 다 쓰지 않고 _josa로 고른다."""
    assert not re.findall(r"을\(를\)|이\(가\)|은\(는\)|\(으\)로", SRC)
