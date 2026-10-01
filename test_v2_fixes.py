# -*- coding: utf-8 -*-
"""2026-09-30 V2 검증에서 고친 오류들의 회귀 테스트."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.formula.api import ols

APP = Path(__file__).resolve().parents[1] / "app.py"
SRC = APP.read_text(encoding="utf-8")


def _load(start, end, extra=None):
    ns = {"np": np, "pd": pd, "stats": stats, "studentized_range": studentized_range}
    ns.update(extra or {})
    exec(SRC[SRC.index(start):SRC.index(end)], ns)
    return ns


def test_module_level_code_does_not_shadow_functions():
    """메뉴 코드(모듈 최상위)의 반복문 변수가 함수 이름(_q 등)을 덮어쓰면 안 된다."""
    tree = ast.parse(SRC)
    funcs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    hits = []
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.ClassDef)):
            continue
        stack = [top]
        while stack:
            node = stack.pop()
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    continue
                if isinstance(ch, ast.For):
                    hits += [(ch.lineno, n.id) for n in ast.walk(ch.target)
                             if isinstance(n, ast.Name) and n.id in funcs]
                stack.append(ch)
    assert not hits, hits


def test_numeric_like_ignores_treatment_names():
    ns = _load("def find_numeric_like", "def to_numeric_clean")
    d = pd.DataFrame({"처리구": ["대조구", "처리1", "처리2", "처리3"] * 2,
                      "수량": ["1,200kg", "980 kg", "1,050", "1100"] * 2,
                      "초장": ["15.5cm", "16cm", "14.8", "15"] * 2})
    found = ns["find_numeric_like"](d)
    assert "처리구" not in found
    assert "수량" in found and "초장" in found


def test_duncan_pvalue_matches_judgement():
    ns = _load("def compact_letter_display", "def posthoc_not_sig")
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"t": np.repeat(list("ABCDE"), 4),
                      "y": np.concatenate([rng.normal(m, 2, 4) for m in (10, 11.5, 13, 14, 16)])})
    tb = ns["posthoc_from_model"](ols("y ~ C(t)", d).fit(), d, "t", "던컨(Duncan)")["table"]
    assert ((tb["판정"] == "유의(*)") == (tb["p(보정)"] < 0.05)).all()


def test_autopilot_keeps_converted_columns_and_writes_method():
    seg = SRC[SRC.index("def run_autopilot_engine"):SRC.index("def log_action")]
    assert "if c in orig_cat and c not in fixed:" in seg
    assert 'st.session_state.get("log", [])' not in seg


def test_login_uses_firebase_like_v1():
    """V2 로그인은 V1과 같은 Firebase 방식이어야 하고, 로그인 상태는 데이터 전환 시 유지돼야 한다."""
    assert "Supabase" not in SRC and "supabase" not in SRC
    assert "identitytoolkit.googleapis.com" in SRC
    assert "_ai_remember_load()" in SRC
    prefix_seg = SRC[SRC.index("_PIN_GLOBAL_PREFIX = ("):SRC.index("_PIN_BUTTON_EXACT = {")]
    for p in ('"auth_"', '"_auth_"', '"_ai_remember"', '"ssa_"'):
        assert p in prefix_seg, p
