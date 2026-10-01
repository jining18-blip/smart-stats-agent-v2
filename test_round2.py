# -*- coding: utf-8 -*-
"""2026-10-01 2차 수정(해석 문장·PDF·엑셀 그래프·경제성 양식 등) 회귀 테스트."""
import io
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SRC = APP.read_text(encoding="utf-8")


def _seg(start, end, extra=None):
    ns = {"np": np, "pd": pd, "re": re, "io": io, "stats": stats, "Counter": Counter,
          "q_ref": lambda c: f"Q('{c}')"}
    ns.update(extra or {})
    exec(SRC[SRC.index("_JOSA_DIGIT_JONG ="):SRC.index("def report_sentence_anova")], ns)
    if start:
        exec(SRC[SRC.index(start):SRC.index(end)], ns)
    return ns


@pytest.fixture(scope="module")
def ns():
    return _seg(None, None)


# ---------------------------------------------------------------- 회귀·로지스틱
def test_regression_table_matches_statsmodels(ns):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x1": rng.normal(10, 2, 40), "x2": rng.normal(5, 1, 40)})
    d["y"] = 3 + 2 * d["x1"] + rng.normal(0, 1, 40)
    m = sm.OLS(d["y"], sm.add_constant(d[["x1", "x2"]])).fit()
    coef, easy, rep, cautions = ns["interp_regression"](m, "y", ["x1", "x2"], d)
    row = coef.set_index("변수").loc["x1"]
    assert row["회귀계수"] == pytest.approx(round(m.params["x1"], 4))
    beta = m.params["x1"] * d["x1"].std() / d["y"].std()
    assert float(row["표준화계수(β)"]) == pytest.approx(beta, abs=1e-3)
    assert row["유의성"] == "***"
    assert "x1" in rep and "x2" in rep and "0.0000" not in rep
    assert rep.startswith("○ ")


def test_logit_uses_unpenalized_mle_and_odds_ratio(ns):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    y = (rng.random(300) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(int)
    res = sm.Logit(y, sm.add_constant(pd.DataFrame({"x": x}))).fit(disp=0)
    tbl, easy, rep = ns["interp_logit"](res, "발병", ["x"], "발병", acc=0.8)
    assert float(tbl.set_index("변수").loc["x", "오즈비(OR)"]) == pytest.approx(np.exp(res.params["x"]), abs=1e-3)
    assert "오즈" in rep


def test_logit_screen_guesses_positive_class_exactly():
    seg = SRC[SRC.index("_pos_guess = next("):SRC.index('pos_label = (st.selectbox(')]
    assert '"발병"' in seg and "v.strip().lower() in" in seg   # '미발병'을 고르지 않도록 정확히 일치


# ---------------------------------------------------------------- 반복측정 구형성
def test_greenhouse_geisser_matches_box_formula(ns):
    rng = np.random.default_rng(3)
    n, k = 15, 5
    X = rng.normal(size=(n, k)) @ rng.normal(size=(k, k))
    d = pd.DataFrame([(i, j, X[i, j]) for i in range(n) for j in range(k)], columns=["s", "t", "y"])
    S = np.cov(X, rowvar=False)
    m, md, mr = S.mean(), np.diag(S).mean(), S.mean(axis=1)
    box = (k ** 2 * (md - m) ** 2) / ((k - 1) * ((S ** 2).sum() - 2 * k * (mr ** 2).sum() + k ** 2 * m ** 2))
    assert ns["gg_epsilon"](d, "s", "t", "y") == pytest.approx(box, abs=1e-9)


# ---------------------------------------------------------------- 설문 해석
def test_choice_line_handles_ties_and_skew(ns):
    t = pd.DataFrame({"성별": ["남", "여"], "빈도(명)": [30, 30], "비율(%)": [50.0, 50.0]})
    line = ns["_svy_choice_line"]("성별", t, "빈도(명)", "비율(%)")
    assert "같았다" in line and "쏠림" not in line
    t2 = pd.DataFrame({"답": ["예", "아니오"], "빈도(명)": [80, 20], "비율(%)": [80.0, 20.0]})
    assert "쏠림" in ns["_svy_choice_line"]("문항", t2, "빈도(명)", "비율(%)")


def test_crosstab_report_only_claims_significance_when_tested(ns):
    ct = pd.DataFrame([[20, 2], [3, 18]], index=["A", "B"], columns=["예", "아니오"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    easy, rep = ns["interp_crosstab"]("집단", "응답", ct, chi2, p, dof, 0)
    assert "유의하였다" in rep and "A는 '예'" in rep
    easy2, rep2 = ns["interp_crosstab"]("집단", "응답", ct)          # 검정 못 한 경우
    assert "유의" not in rep2


def test_survey_text_top_words_skip_predicates(ns):
    words = ns["_top_words"](["엑셀 저장이 편리했으면", "엑셀 시트가 많아요", "시트 정리 필요합니다", "엑셀"])
    assert "엑셀" in words and not any(w.endswith("으면") for w in words)


# ---------------------------------------------------------------- 엑셀 그래프·양식
def test_make_xlsx_stacked_and_note_only_with_chart():
    seg = SRC[SRC.index("def make_xlsx(df, title"):SRC.index("def _rewrite_chart_sheet_refs")]
    assert "stacked=False, max_series=3" in seg
    assert seg.count("a2.value = None") >= 2          # 그래프가 없으면 '숫자를 고치면…' 안내도 없음
    assert 'if sig:' in seg                            # 유의성 안내는 유의성 문자가 있을 때만


def test_econ_template_and_area_detection():
    ns = {"io": io, "pd": pd, "np": np}
    exec(SRC[SRC.index("_ECON_TEMPLATE_COLS = ["):SRC.index("def _area_prefill")], ns)
    raw = ns["econ_input_template_xlsx"]()
    d = pd.read_excel(io.BytesIO(raw), sheet_name="입력")
    assert {"처리구", "반복", "조사면적(a)", "수량(kg)", "판매단가(원/kg)", "자가노동시간"} <= set(d.columns)
    # 비목은 모두 '~비'로 끝나 경영비로 자동 선택된다
    costs = [c for c in d.columns if c not in ("처리구", "반복", "조사면적(a)", "수량(kg)",
                                               "판매단가(원/kg)", "자가노동시간")]
    assert costs and all(c.endswith("비") for c in costs)
    assert ns["_area_from_data"](d)[0] == 10.0
    d2 = d.copy(); d2.loc[0, "조사면적(a)"] = 5
    v, col, warn = ns["_area_from_data"](d2)
    assert v is None and warn
    assert ns["_area_from_data"](pd.DataFrame({"조사면적(㎡)": [500, 500]}))[0] == pytest.approx(5.0)
    assert "빈 서식" not in SRC and 'key="p_econ_guide_template"' in SRC


def test_price_db_obsolete_columns_removed():
    seg = SRC[SRC.index("_PRICE_DB_OBSOLETE_COLS"):SRC.index("def price_db_warnings")]
    assert '"갱신일"' in seg and '"환산식"' in seg and '"갱신일": ' not in seg.split("def default_price_db")[1]


# ---------------------------------------------------------------- PDF
def test_pdf_table_extraction_with_borders():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(4, 2)); ax.axis("off")
    ax.table(cellText=[["A", "1", "1,200"], ["B", "2", "980"]], colLabels=["trt", "rep", "yield"], loc="center")
    buf = io.BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig)
    ns = {"io": io, "re": re, "np": np, "pd": pd}
    exec(SRC[SRC.index("def find_numeric_like"):SRC.index("def _polish_figure")], ns)
    exec(SRC[SRC.index("def _pdf_table_to_df"):SRC.index("def image_to_dataframe")], ns)
    tabs, text_pages, n_pages = ns["pdf_to_tables"](buf.getvalue())
    assert len(tabs) == 1 and text_pages == 1
    d = tabs[0][1]
    assert list(d.columns) == ["trt", "rep", "yield"]
    assert d["yield"].tolist() == [1200, 980]            # '1,200' → 숫자
    assert 'type=["xlsx", "xls", "csv", "pdf"]' in SRC


# ---------------------------------------------------------------- 화면 흐름
def _app(df, **state):
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.session_state["files"] = {"d": df.copy()}
    at.session_state["cur_key"] = "d"
    at.session_state["df"] = df.copy()
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _rcbd():
    rng = np.random.default_rng(11)
    blk = {1: -25, 2: 0, 3: 28}
    return pd.DataFrame([{"처리": t, "반복": b, "수량": 500 + e + blk[b] + rng.normal(0, 6)}
                         for b in (1, 2, 3) for t, e in [("T1", 0), ("T2", 8), ("T3", 12), ("T4", 5)]])


def test_regression_screen_shows_korean_table_and_report():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(10, 2, 30)}); d["y"] = 2 * d["x"] + rng.normal(0, 1, 30)
    at = _app(d, menu_choice="📊 통계분석", stat_sub="📉 회귀분석", lin_y="y", lin_x=["x"])
    [b for b in at.main.button if "회귀분석 실행" in b.label][0].click().run()
    assert not at.exception
    assert any("표준화계수(β)" in df_.value.columns for df_ in at.main.dataframe)
    assert any(c.value.startswith("○ y에 대한 회귀분석 결과") for c in at.main.code)


def test_multi_summary_uses_numeric_block_by_default():
    at = _app(_rcbd(), menu_choice="📊 통계분석", stat_sub="📈 분산분석")
    at.main.radio[1].set_value("📊 여러 형질 한 표에 (요약표)").run()
    assert [s.value for s in at.main.selectbox if s.key == "ms_blk"] == ["반복"]
    [b for b in at.main.button if "요약표 만들기" in b.label][0].click().run()
    assert not at.exception
    rep = [c.value for c in at.main.code if c.value.startswith("○ ")][0]
    assert "p=0.0372" in rep                       # 블록을 넣은 RCBD p값 (CRD면 0.8392)


def test_two_way_offers_block():
    rng = np.random.default_rng(5)
    rows = [{"A": a, "B": b, "반복": r, "y": 10 + (a == "a2") * 2 + rng.normal(0, 1)}
            for a in ("a1", "a2") for b in ("b1", "b2") for r in (1, 2, 3)]
    at = _app(pd.DataFrame(rows), menu_choice="📊 통계분석", stat_sub="📈 분산분석")
    at.main.radio[1].set_value("이원배치 (요인 2개 + 상호작용)").run()
    assert [s.value for s in at.main.selectbox if s.key == "tw_blk"] == ["반복"]
    [b for b in at.main.button if "이원배치" in b.label][0].click().run()
    assert not at.exception
    assert any("블록(반복) 효과를 모형에 포함" in c.value for c in at.main.code)


def test_crosstab_default_skips_id_column():
    d = pd.DataFrame({"응답자ID": [f"R{i:03d}" for i in range(40)],
                      "성별": ["남", "여"] * 20, "만족": ["예", "아니오", "예", "예"] * 10})
    at = _app(d, menu_choice="📋 설문조사 분석", svy_type="🔀 교차분석(집단 비교)")
    assert [s.value for s in at.main.selectbox if s.key == "ct_r"] == ["성별"]


def test_stat_method_text_screen_breaks_lines():
    ns = {"re": re}
    exec(SRC[SRC.index("def sentence_lines"):SRC.index("def build_abstract")], ns)
    out = ns["sentence_lines"]("가는 하였다. 나는 하였다. 다는 하였다.")
    assert out.split("\n") == ["가는 하였다.", "나는 하였다.", "다는 하였다."]


def test_manual_mentions_only_existing_screen_labels():
    """사용설명서에 적은 버튼·메뉴 이름이 실제 화면 코드에 있어야 한다(설명서-화면 불일치 방지)."""
    import ast as _ast
    i = SRC.index('    _MANUAL = "')
    manual = _ast.literal_eval(SRC[i:SRC.index("\n", i)].split("=", 1)[1].strip()).replace("**", "")
    code = SRC[:i] + SRC[SRC.index("\n", i):]
    labels = ["⚡ 원클릭 보고서", "📥 경제성 분석 입력 양식 받기 (엑셀)",
              "📚 경제성 분석에 어떤 자료를 준비해야 하나요?", "자료의 기준 면적",
              "📊 올린 데이터에서 자동으로 채우기", "🧮 이 숫자가 어떻게 나온 건가요?",
              "➕ 이 결과를 보고서에 담기", "🆘 이 오류, 도움받기", "📄 상세 결과",
              "관심 범주", "🤖 AI 기능 켜기", "🧪 샘플 데이터", "🔮 새 데이터", "숫자로 변환"]
    for lab in labels:
        assert lab in manual, f"설명서에 없음: {lab}"
        assert lab in code, f"화면에 없음: {lab}"
    for gone in ("KAMIS", "kosis.kr/openapi", "빈 CSV 서식", "갱신일", "환산식"):
        assert gone not in manual, gone


def test_no_duplicate_top_level_definitions():
    """같은 이름의 함수를 두 번 정의하면 앞의 것이 덮여 다른 기능이 깨진다(예: 한글 XML용 _q)."""
    import ast as _ast
    tree = _ast.parse(SRC)
    names = [n.name for n in tree.body if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, dup


def test_report_files_build_with_new_captures():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(10, 2, 30), "z": rng.normal(5, 1, 30)})
    d["y"] = 2 * d["x"] + rng.normal(0, 1, 30)
    at = _app(d, menu_choice="📊 통계분석", stat_sub="📉 회귀분석", lin_y="y", lin_x=["x"])
    [b for b in at.main.button if "회귀분석 실행" in b.label][0].click().run()
    [b for b in at.main.button if "보고서에 담기" in b.label][0].click().run()
    at.session_state["stat_sub"] = "🔗 상관분석"; at.run()
    [b for b in at.main.button if "보고서에 담기" in b.label][0].click().run()
    at.session_state["menu_choice"] = "📑 보고서"; at.run()
    [x for x in at.main.checkbox if x.key == "gen_report"][0].check().run()
    assert not at.exception
    labels = [x.proto.label for x in at.get("download_button")]
    assert any("hwpx" in l for l in labels) and any("docx" in l for l in labels)


def test_floating_ai_answer_shown_and_empty_question_warned(monkeypatch):
    """떠 있는 AI 창: 빈 질문이면 안내, 질문하면 답이 바로 보이고, 지우기가 동작한다(API는 모의 응답)."""
    import json as _json
    import requests

    class _R:
        status_code = 200
        def __init__(self, js): self._js = js; self.text = _json.dumps(js)
        def json(self): return self._js

    monkeypatch.setattr(requests, "post",
                        lambda url, *a, **k: _R({"status": "completed", "output_text": "모의 답변"}))
    d = pd.DataFrame({"처리": list("AABB"), "y": [1.0, 2.0, 3.0, 4.0]})
    at = _app(d, menu_choice="📊 통계분석", stat_sub="📋 데이터", api_key="sk-test",
              ai_provider="ChatGPT (OpenAI)", ai_model_g="gpt-4.1-mini")
    [b for b in at.button if b.key == "gai_send"][0].click().run()
    assert any("질문을 입력" in w.value for w in at.warning)
    [t for t in at.text_area if t.key == "gai_q"][0].input("질문")
    [b for b in at.button if b.key == "gai_send"][0].click().run()
    assert not at.exception
    msgs = [cm.markdown[0].value for cm in at.get("chat_message")]
    assert msgs[-2:] == ["질문", "모의 답변"]
    [b for b in at.button if b.key == "gai_clear"][0].click().run()
    assert at.session_state["gai_hist"] == []


@pytest.mark.parametrize("provider,model,resp", [
    ("ChatGPT (OpenAI)", "gpt-4.1-mini",
     {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
      "output_text": "○ 결론\n- 위의 절차를 따라 파일의 제목행("}),
    ("Gemini (Google)", "gemini-2.5-flash",
     {"candidates": [{"content": {"parts": [{"text": "○ 결론\n- 위의 절차를 따라 파일의 제목행("}]},
                      "finishReason": "MAX_TOKENS"}]}),
])
def test_ai_truncated_answer_is_flagged(monkeypatch, provider, model, resp):
    """답이 출력 한도에 걸려 끊기면 잘린 채로 끝나지 않고 '끊겼습니다' 안내가 붙어야 한다."""
    import json as _json
    import requests
    sent = {}

    class _R:
        status_code = 200
        def __init__(self, js): self._js = js; self.text = _json.dumps(js)
        def json(self): return self._js

    def _post(url, *a, **k):
        sent["json"] = k.get("json")
        return _R(resp)
    monkeypatch.setattr(requests, "post", _post)
    d = pd.DataFrame({"처리": list("AABB"), "y": [1.0, 2.0, 3.0, 4.0]})
    at = _app(d, menu_choice="📊 통계분석", stat_sub="📋 데이터", api_key="k",
              ai_provider=provider, ai_model_g=model,
              gai_hist=[("q", "앞 질문"), ("a", "앞 답변")])
    [t for t in at.text_area if t.key == "gai_q"][0].input("이어서 설명해줘")
    [b for b in at.button if b.key == "gai_send"][0].click().run()
    assert not at.exception
    ans = [cm.markdown[0].value for cm in at.get("chat_message")][-1]
    assert "끊겼습니다" in ans
    body = _json.dumps(sent["json"], ensure_ascii=False)
    assert "[직전 대화]" in body and "앞 답변" in body          # 후속 질문에 앞 대화를 함께 보냄
    if provider.startswith("Gemini"):
        assert sent["json"]["generationConfig"]["maxOutputTokens"] >= 2000 + 4096


def test_claude_truncated_answer_is_flagged(monkeypatch):
    import anthropic

    class _Blk:
        type = "text"; text = "○ 결론\n- 위의 절차를 따라 파일의 제목행("

    class _Msg:
        content = [_Blk()]; stop_reason = "max_tokens"

    class _Cli:
        def __init__(self, *a, **k):
            self.messages = self
        def create(self, **k):
            return _Msg()
    monkeypatch.setattr(anthropic, "Anthropic", _Cli)
    d = pd.DataFrame({"처리": list("AABB"), "y": [1.0, 2.0, 3.0, 4.0]})
    at = _app(d, menu_choice="📊 통계분석", stat_sub="📋 데이터", api_key="k",
              ai_provider="Claude (Anthropic)", ai_model_g="claude-test")
    [t for t in at.text_area if t.key == "gai_q"][0].input("질문")
    [b for b in at.button if b.key == "gai_send"][0].click().run()
    assert "끊겼습니다" in [cm.markdown[0].value for cm in at.get("chat_message")][-1]
