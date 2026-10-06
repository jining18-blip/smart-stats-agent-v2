# -*- coding: utf-8 -*-
"""2026-10-01 4차 수정 회귀 테스트.

1. 입력 방식 이름에 PDF 표시(내부 값은 그대로)
2·4. 버전1 데이터 점검 이식 + 요약 행 빼기·원클릭 자동 제외 + 보강 검사
3. p값 0.000000 / 표 실수 1.000000 표시
5. PDF 부품이 없을 때 안내 문구
"""
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
    """세션 구역 앞까지(함수 정의부)를 실행해 순수 함수를 꺼내 쓴다."""
    env = {"__name__": "round3_mod", "__file__": str(APP)}
    exec(compile(SRC[:SRC.index("# ================================================================ 세션")],
                 str(APP), "exec"), env)
    return env


def _mixed_df():
    """평균·전체 평균 행이 섞인 난괴법 자료(사용자 신고 사례와 같은 모양)."""
    vals = {"대조구": [(89.5, 2349.7), (91.1, 2349.2), (94.3, 2391.3), (90.0, 2253.8)],
            "처리1": [(93.3, 2559.9), (100.6, 2587.2), (91.1, 2528.0), (97.0, 2621.4)],
            "처리2": [(95, 2700), (96, 2750), (97, 2800), (94, 2690)],
            "처리3": [(100, 3000), (101, 3013), (99, 3020), (100, 3021)],
            "처리4": [(92, 2600), (93, 2610), (94, 2620), (95, 2630)]}
    rows = []
    for t, v in vals.items():
        for r, (a, b) in enumerate(v, 1):
            rows.append([t, r, a, b])
        rows.append([t + " 평균", None, np.mean([x[0] for x in v]), np.mean([x[1] for x in v])])
    rows.append(["전체 평균", None, 95.0, 2600.0])
    return pd.DataFrame(rows, columns=["처리구", "반복", "초장(cm)", "수량(kg/10a)"])


# ---------------------------------------------------------------- 1. 입력 방식 이름
def test_input_mode_label_shows_pdf_but_keeps_value():
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    r = at.sidebar.radio(key="data_input_mode")
    assert r.options[0] == "📁 파일 (Excel·CSV·PDF)"
    assert r.value == "📁 Excel/CSV"            # 내부 값·비교문은 그대로
    assert 'if _input_mode == "📁 Excel/CSV":' in SRC


# ---------------------------------------------------------------- 5. PDF 부품 안내
def test_missing_module_message(ns):
    m = ns["_missing_module_msg"]
    t = m("시험.pdf", ModuleNotFoundError("No module named 'pdfplumber'", name="pdfplumber"))
    assert "PDF 읽기 부품(pdfplumber)" in t and "requirements.txt" in t and "No module" not in t
    t = m("옛.xls", ImportError("Missing optional dependency 'xlrd'. Install xlrd >= 2.0.1"))
    assert "xlrd" in t and "옛 엑셀" in t
    assert "except ImportError as e:" in SRC and "_missing_module_msg(uf.name, e)" in SRC


# ---------------------------------------------------------------- 3. 표시 형식
def test_smart_cell_text(ns):
    f = ns["_smart_cell_text"]
    assert f(1.0) == "1" and f(2.7) == "2.7" and f(3013.5) == "3013.5" and f(np.float64(0.039)) == "0.039"
    assert f(0.0) == "0" and f(1.2e-8) == "1.20e-08" and f(5) == "5" and f("처리1") == "처리1"
    assert f(float("nan")) == "nan"          # 결측 표시는 예전 그대로


def test_oneclick_and_multi_summary_p_use_pcell():
    assert '"p-value": _pcell(pval, 4),' in SRC
    assert '"p-value": round(pval, 4)' not in SRC
    assert 'notes.append({"항목": tr, "p-value": _pcell(pval, 4),' in SRC


def test_multi_summary_interp_reads_text_p(ns):
    d = pd.DataFrame({"g": list("AABB"), "y": [1, 2, 9, 10]})
    _, rep = ns["interp_multi_summary"]("g", "Tukey HSD", [{"항목": "y", "p-value": "<0.0001"}], d, ["y"])
    assert "p<0.0001" in rep and "할 수 없었다" not in rep


# ---------------------------------------------------------------- 2·4. 데이터 점검
def test_summary_rows_detected(ns):
    d = _mixed_df()
    assert ns["find_summary_rows"](d) == [4, 9, 14, 19, 24, 25]
    f = ns["data_checkup"](d)
    errs = [x["title"] for x in f if x["level"] == "error"]
    assert errs == ["평균·합계 같은 요약 행이 들어 있습니다."]
    assert not [x for x in f if x["level"] == "warn"]   # 요약 행의 빈 반복 칸은 결측으로 따로 세지 않는다


@pytest.mark.parametrize("df,expected", [
    (pd.DataFrame({"의견": ["가격이 평균", "좋아요", "작년 평균", "평균"], "점수": [1, 2, 3, 4]}), [3]),
    (pd.DataFrame({"처리구": ["가중평균", "가중평균", "대조", "대조"], "반복": [1, 2, 1, 2], "y": [1, 2, 3, 4]}), []),
    (pd.DataFrame({"처리구": ["A", "A", "B", "B", "계"], "y": [1, 2, 3, 4, 10]}), [4]),
    (pd.DataFrame({"처리구": ["A", "A", "B", "B", "A 평균", "총평균"], "y": [1, 2, 3, 4, 1.5, 2.5]}), [4, 5]),
    (pd.DataFrame({"처리구": ["A", "A", "B", "B", "CV(%)", "LSD(0.05)", "평균(Mean)"], "y": range(7)}), [4, 5, 6]),
    (pd.DataFrame({"품종": ["SE", "SE", "청양", "청양"], "y": [1, 2, 3, 4]}), []),   # 영문 약어 품종명은 빼지 않는다
])
def test_summary_rows_no_false_positive(ns, df, expected):
    assert ns["find_summary_rows"](df) == expected


@pytest.mark.parametrize("kind", ["실험", "경제성", "설문", "반복측정", "분할구", "프로빗"])
def test_samples_pass_checkup(ns, kind):
    c = ns["checkup_counts"](ns["data_checkup"](ns["make_sample"](kind)))
    assert c["error"] == 0 and c["warn"] == 0, c


def test_checkup_extra_checks(ns):
    chk = ns["data_checkup"]
    slip = pd.DataFrame({"처리구": np.repeat(["대조구", "처리1"], 4), "반복": [1, 2, 3, 4] * 2,
                         "초장": [89.5, 91.1, 943, 90.0, 93, 95, 94, 96]})
    assert any("혼자 크게 튀는 값" in x["title"] and "943" in x["detail"] for x in chk(slip))
    inc = pd.DataFrame({"처리구": np.repeat(["A", "B", "C"], 4), "반복": [1, 2, 3, 4] * 3,
                        "발병률(%)": [0, 2, 10, 1, 0, 0, 5, 0, 20, 25, 30, 22]})
    assert not [x for x in chk(inc) if x["level"] in ("error", "warn")]
    dup = pd.DataFrame({"처리구": list("AAABBB"), "반복": [1, 1, 2, 1, 2, 3], "수량": [1, 2, 3, 4, 5, 6.5]})
    assert any("같은 처리·같은 반복" in x["title"] for x in chk(dup))
    copied = pd.DataFrame({"처리구": ["대조구"] * 3 + ["처리1"] * 3, "반복": [1, 2, 3] * 2,
                           "수량": [605.0] * 3 + [645.0] * 3})
    assert any("반복 간 값이 거의 같습니다" in x["title"] for x in chk(copied))
    head = pd.DataFrame([["처리구", "반복", "수량"], ["대조구", 1, 600]], columns=["2025 결과", "열", "열_2"])
    assert any("첫 행이 변수명" in x["title"] for x in chk(head))
    name = pd.DataFrame({"처리구": ["대조구", "대조 구", "처리1", "처리1"], "반복": [1, 2, 1, 2], "y": [1, 2, 3, 4]})
    assert any("다르게 적힌" in x["title"] for x in chk(name))
    blank = pd.DataFrame({"처리구": ["A", None, "B"], "y": [1, None, 2]})
    assert any("모든 칸이 빈 행" in x["title"] for x in chk(blank))


def test_drop_row_positions_with_duplicate_index(ns):
    d = pd.DataFrame({"a": ["x", "평균", "y"]}, index=[0, 0, 1])
    out = ns["drop_row_positions"](d, ns["summary_row_positions"](d))
    assert out["a"].tolist() == ["x", "y"]


def _app_with(df, menu, **state):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.session_state["files"] = {"평균섞임.xlsx": df.copy()}
    at.session_state["menu_choice"] = menu
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def test_popup_badge_and_drop_button():
    at = _app_with(_mixed_df(), "⚡ 원클릭 보고서")
    assert not at.exception
    assert any("고쳐야 할 문제" in e.value for e in at.sidebar.error)
    assert any("방금 불러온" in m.value for m in at.main.markdown)
    assert any("자동으로 빼고 분석합니다" in i.value for i in at.main.info)
    [b for b in at.main.button if b.key == "ck_fix_popup"][0].click().run()
    assert not at.exception
    assert len(at.session_state["df"]) == 20
    assert len(at.session_state["files"]["평균섞임.xlsx"]) == 26      # 원본은 그대로
    assert any("분석 준비 완료" in s.value for s in at.sidebar.success)
    assert not any("방금 불러온" in m.value for m in at.main.markdown)


def test_oneclick_excludes_summary_rows():
    at = _app_with(_mixed_df(), "⚡ 원클릭 보고서")
    [b for b in at.main.button if b.label == "🚀 원클릭 분석 시작"][0].click().run()
    assert not at.exception
    ap = at.session_state["autopilot"]
    assert ap["ok"]
    assert any("요약 행 6개를 빼고 분석했습니다" in m for m in ap["msgs"])
    assert "난괴법(RCBD)" in ap["design"]["design"]
    p = ap["summary"].set_index("측정 항목")["p-value"]
    assert p["수량(kg/10a)"] == "<0.0001"
    assert "0.0000" not in ap["abstract"]


def test_stat_tabs_show_checkup_and_notice():
    at = _app_with(_mixed_df(), "📊 통계분석", stat_sub="📋 데이터")
    assert not at.exception
    assert any("요약 행이 들어 있습니다" in m.value for m in at.main.markdown)
    assert [b for b in at.main.button if b.key == "ck_fix_data"]
    assert not any("방금 불러온" in m.value for m in at.main.markdown)    # 데이터 화면에서는 상자 생략
    at.session_state["stat_sub"] = "📈 분산분석"
    at.run()
    assert not at.exception
    assert any("요약 행 6개가 섞여 있습니다" in e.value for e in at.main.error)


def test_clean_sample_has_no_popup():
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    [b for b in at.button if b.label == "🌱 실험"][0].click().run()
    assert not at.exception
    assert not any("방금 불러온" in m.value for m in at.main.markdown)
    assert any("분석 준비 완료" in s.value for s in at.sidebar.success)


def test_new_keys_are_pinned_safely():
    assert '"checkup_ack",' in SRC and '"ck_fix_")' in SRC and '"_checkup")' in SRC


def test_manual_mentions_new_features():
    i = SRC.index('    _MANUAL = "')
    manual = ast.literal_eval(SRC[i:SRC.index("\n", i)].split("=", 1)[1].strip()).replace("**", "")
    code = SRC[:i] + SRC[SRC.index("\n", i):]
    for lab in ("📁 파일 (Excel·CSV·PDF)", "🧹 요약 행 빼기", "분석 준비 완료"):
        assert lab in manual and lab in code, lab
    assert "pdfplumber" in manual


# ---------------------------------------------------------------- 문의처
def test_contact_shown_everywhere():
    assert 'CONTACT_NAME = "경상북도농업기술원 영양고추연구소 이효진"' in SRC
    assert 'CONTACT_EMAIL = "hyo99@korea.kr"' in SRC
    i = SRC.index('    _MANUAL = "')
    manual = ast.literal_eval(SRC[i:SRC.index("\n", i)].split("=", 1)[1].strip())
    assert "## 10. 문의" in manual and "hyo99@korea.kr" in manual
    assert "가입·로그인 문의: {CONTACT_NAME}" in SRC                  # 로그인 화면
    assert "캡처해 **{CONTACT_EMAIL}**" in SRC                          # 오류 도움받기 상자
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    assert not at.exception
    cap = " ".join(c.value for c in at.sidebar.caption)
    assert "문의" in cap and "경상북도농업기술원 영양고추연구소" in cap and "hyo99@korea.kr" in cap
