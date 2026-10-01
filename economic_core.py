# -*- coding: utf-8 -*-
"""경제성 분석의 순수 계산·검증 엔진.

Streamlit UI와 분리해 동일 입력에는 동일 결과가 나오도록 하고, 잘못된 입력은
계산 전에 차단한다. 금액의 내부 계산은 반올림하지 않으며 표시에만 ROUND_HALF_UP을 쓴다.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, Optional

import numpy as np
import pandas as pd

EXACT_EXCLUDE_COST_NAMES = {
    "경영비", "생산비", "총비용", "비용합계", "경영비합계", "생산비합계",
    "자가노력비", "자가노동비", "토지용역비", "유동자본용역비", "고정자본용역비",
    "자본용역비", "총수입", "조수입", "소득", "순수익",
}
SUFFIX_EXCLUDE_COST_NAMES = {"합계", "총계"}
CONTAINS_EXCLUDE_COST_NAMES = {
    "자가노력비", "자가노동비", "토지용역비", "유동자본용역비", "고정자본용역비",
    "자본용역비",
}


def _norm_name(name) -> str:
    """열 이름을 중복계상 검사에 사용할 형태로 정규화한다."""
    text = str(name)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[\s_\-/]", "", text)
    text = re.sub(r"(원|천원|만원|10a|ha|kg)$", "", text, flags=re.I)
    return text.strip().lower()


def is_excluded_cost(name) -> bool:
    """합계·계산결과·별도 산정 항목을 비용 원자료에서 제외한다."""
    normalized = _norm_name(name)
    exact = {_norm_name(x) for x in EXACT_EXCLUDE_COST_NAMES}
    if normalized in exact:
        return True
    if any(normalized.endswith(_norm_name(x)) for x in SUFFIX_EXCLUDE_COST_NAMES):
        return True
    return any(len(_norm_name(token)) >= 4 and _norm_name(token) in normalized
               for token in CONTAINS_EXCLUDE_COST_NAMES)


def is_depreciation_cost(name) -> bool:
    """감가상각·상각 성격의 경영비 열인지 이름으로 보수적으로 판별한다."""
    normalized = _norm_name(name)
    return any(token in normalized for token in ("감가상각", "상각비", "시설상각", "대농구상각"))


def is_land_rent_cost(name) -> bool:
    """토지 임차료로 볼 가능성이 높은 비용명만 판별한다. 장비·시설 임차는 제외한다."""
    raw = str(name).strip()
    normalized = _norm_name(name)
    machine_hints = ("농기계", "기계", "장비", "시설", "하우스", "트랙터",
                     "관리기", "드론", "로봇", "스마트", "설비", "차량")
    if any(k in raw for k in machine_hints):
        return False
    land_hints = ("토지", "농지", "지대", "밭", "논", "경지", "부지", "전답")
    generic = {_norm_name(x) for x in ("임차료", "임대료", "지대", "임차비", "임대비")}
    return normalized in generic or (any(k in raw for k in land_hints)
                                     and any(k in raw for k in ("임차", "임대", "지대")))


def round_half_up(value, digits: int = 0):
    """회계 표시용 사사오입(ROUND_HALF_UP). NaN은 NaN으로 유지한다."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return np.nan
    try:
        quant = Decimal("1").scaleb(-int(digits))
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"반올림할 수 없는 값입니다: {value!r}") from exc
    return int(rounded) if digits == 0 else float(rounded)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """콤마와 단위가 섞인 문자열을 숫자로 바꾸되 잘못된 값은 NaN으로 둔다.

    회계식 음수 표기(△1,000 / ▲1,000 / (1,000))도 음수로 인식한다.
    예전에는 기호만 떨어져 나가 1000(양수)이 되어 부호가 뒤집혔다.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    # △1,000  ▲1,000  ▵1,000  (1,000)  ->  음수
    negative = (text.str.match(r"^\s*[△▲▵▽]") |
                text.str.match(r"^\s*\(.*\)\s*$"))
    cleaned = (text.str.replace(",", "", regex=False)
               .str.replace(r"[^\d.\-]", "", regex=True))
    values = pd.to_numeric(cleaned, errors="coerce")
    return values.where(~negative, -values.abs())


def _finite_nonnegative_scalar(value, label: str, errors: list[str],
                               *, minimum: float = 0.0,
                               maximum: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}은(는) 숫자여야 합니다.")
        return None
    if not math.isfinite(number):
        errors.append(f"{label}에 NaN 또는 무한대를 사용할 수 없습니다.")
    elif number < minimum:
        errors.append(f"{label}은(는) {minimum:g} 이상이어야 합니다.")
    elif maximum is not None and number > maximum:
        errors.append(f"{label}은(는) {maximum:g} 이하여야 합니다.")
    return number


def validate_economic_inputs(
    df: pd.DataFrame,
    treatment_col,
    quantity_col,
    price_col,
    cost_cols: Iterable,
    labor_col=None,
    land_cost_per_10a=0,
    land_cash_rent_per_10a=0,
    land_type="토지비 제외",
    source_area_a=10.0,
    byproduct_col=None,
    wage_per_hour=0,
    interest_rate=0,
    capital_months=6,
    fixed_asset_per_10a=0,
    fixed_asset_use_rate_percent=100.0,
    establishment_amort_per_10a=0,
    depreciation_cost_cols=None,
) -> list[str]:
    """경제성 계산을 왜곡할 설정·결측·음수·무한대를 사전에 검사한다."""
    errors: list[str] = []
    columns = list(df.columns)
    cost_cols = list(cost_cols or [])

    for col, label in ((treatment_col, "처리구"), (quantity_col, "수량"),
                       (price_col, "단가")):
        if col not in columns:
            errors.append(f"{label} 열을 찾을 수 없습니다: {col}")
    missing_costs = [c for c in cost_cols if c not in columns]
    if missing_costs:
        errors.append("자료에 없는 경영비 열: " + ", ".join(map(str, missing_costs)))
    depreciation_cost_cols = list(depreciation_cost_cols or [])
    missing_dep = [c for c in depreciation_cost_cols if c not in columns]
    if missing_dep:
        errors.append("자료에 없는 감가상각비 열: " + ", ".join(map(str, missing_dep)))
    not_in_cost = [c for c in depreciation_cost_cols if c not in cost_cols]
    if not_in_cost:
        errors.append("감가상각비 열은 경영비에 포함된 열 중에서 선택해야 합니다: "
                      + ", ".join(map(str, not_in_cost)))
    if not cost_cols:
        errors.append("경영비 열을 하나 이상 선택해야 합니다.")
    if quantity_col == price_col:
        errors.append("수량 열과 단가 열은 서로 달라야 합니다.")
    if treatment_col in (quantity_col, price_col):
        errors.append("처리구 열을 수량 또는 단가 열로 사용할 수 없습니다.")

    role_cols = [quantity_col, price_col] + cost_cols
    for optional in (byproduct_col, labor_col):
        if optional and optional != "(없음)":
            if optional not in columns:
                errors.append(f"선택한 열을 찾을 수 없습니다: {optional}")
            role_cols.append(optional)
    duplicates = sorted({str(c) for c in role_cols if role_cols.count(c) > 1})
    if duplicates:
        errors.append("한 열을 여러 역할로 중복 선택했습니다: " + ", ".join(duplicates))

    area = _finite_nonnegative_scalar(source_area_a, "자료 기준 면적", errors)
    if area == 0:
        errors.append("자료 기준 면적은 0보다 커야 합니다.")
    _finite_nonnegative_scalar(wage_per_hour, "농촌임료금", errors)
    _finite_nonnegative_scalar(interest_rate, "자본 이자율", errors, maximum=100)
    _finite_nonnegative_scalar(capital_months, "자본 적용기간", errors, maximum=12)
    _finite_nonnegative_scalar(fixed_asset_per_10a, "고정자산 부분현재가/평가액", errors)
    _finite_nonnegative_scalar(fixed_asset_use_rate_percent, "고정자산 작목부담률", errors, maximum=100)
    _finite_nonnegative_scalar(land_cost_per_10a, "자가토지 용역비", errors)
    _finite_nonnegative_scalar(land_cash_rent_per_10a, "토지 임차료", errors)
    _finite_nonnegative_scalar(establishment_amort_per_10a, "조성비 상각액", errors)

    risky = [c for c in cost_cols if is_excluded_cost(c)]
    if risky:
        errors.append("중복계상 위험 비용 열: " + ", ".join(map(str, risky)))
    if labor_col and labor_col != "(없음)":
        if any(any(key in _norm_name(c) for key in ("자가노력", "자가노동"))
               for c in cost_cols):
            errors.append("자가노력비 열과 자가노동시간×노임을 동시에 사용할 수 없습니다.")
    rent_cols = [c for c in cost_cols if is_land_rent_cost(c)]
    try:
        cash_rent_value = float(land_cash_rent_per_10a or 0)
    except (TypeError, ValueError):
        cash_rent_value = 0
    if rent_cols and cash_rent_value > 0:
        errors.append("경영비의 토지 임차료 열과 별도 토지 임차료를 동시에 적용할 수 없습니다: "
                      + ", ".join(map(str, rent_cols)))

    if treatment_col in columns:
        treatment = df[treatment_col]
        if treatment.isna().any() or treatment.astype(str).str.strip().eq("").any():
            errors.append("처리구 열에 결측 또는 빈 값이 있습니다.")

    checks = [(quantity_col, "수량"), (price_col, "단가")]
    checks += [(c, f"경영비 '{c}'") for c in cost_cols]
    if byproduct_col and byproduct_col != "(없음)":
        checks.append((byproduct_col, "부산물가액"))
    if labor_col and labor_col != "(없음)":
        checks.append((labor_col, "자가노동시간"))

    for col, label in checks:
        if col not in columns:
            continue
        numeric = _to_numeric_series(df[col])
        if numeric.isna().any():
            errors.append(f"{label} 열에 결측 또는 숫자로 변환할 수 없는 값이 있습니다.")
        finite = np.isfinite(numeric.dropna().to_numpy(dtype=float))
        if not finite.all():
            errors.append(f"{label} 열에 무한대가 있습니다.")
        if (numeric.dropna() < 0).any():
            errors.append(f"{label} 열에 음수가 있습니다.")

    return list(dict.fromkeys(errors))


def calculate_row_economics(
    data: pd.DataFrame,
    treatment_col,
    quantity_col,
    price_col,
    cost_cols: Iterable,
    byproduct_col=None,
    labor_col=None,
    wage_per_hour=0,
    interest_rate=0,
    capital_months=6,
    fixed_asset_per_10a=0,
    fixed_asset_use_rate_percent=100.0,
    land_cost_per_10a=0,
    land_cash_rent_per_10a=0,
    establishment_amort_per_10a=0,
    source_area_a=10.0,
    depreciation_cost_cols=None,
):
    """각 행에서 10a 경제성을 계산한 뒤 처리 평균으로 집계한다."""
    errors = validate_economic_inputs(
        data, treatment_col, quantity_col, price_col, cost_cols,
        labor_col=labor_col, land_cost_per_10a=land_cost_per_10a,
        land_cash_rent_per_10a=land_cash_rent_per_10a,
        land_type="토지비 적용" if (float(land_cost_per_10a or 0) > 0 or float(land_cash_rent_per_10a or 0) > 0) else "토지비 제외",
        source_area_a=source_area_a, byproduct_col=byproduct_col,
        wage_per_hour=wage_per_hour, interest_rate=interest_rate,
        capital_months=capital_months, fixed_asset_per_10a=fixed_asset_per_10a,
        fixed_asset_use_rate_percent=fixed_asset_use_rate_percent,
        establishment_amort_per_10a=establishment_amort_per_10a,
        depreciation_cost_cols=depreciation_cost_cols,
    )
    # calculate_row_economics는 임차료 중복 여부를 UI에서 land_type과 함께 검사한다.
    errors = [e for e in errors if not e.startswith("경영비의 임차료와 별도 토지용역비")]
    if errors:
        raise ValueError(" / ".join(errors))

    cost_cols = list(cost_cols)
    area = float(source_area_a)
    factor = 10.0 / area
    columns = [treatment_col, quantity_col, price_col] + cost_cols
    if byproduct_col:
        columns.append(byproduct_col)
    if labor_col:
        columns.append(labor_col)
    columns = list(dict.fromkeys(columns))
    d = data[columns].copy()

    numeric_cols = [quantity_col, price_col] + cost_cols
    if byproduct_col:
        numeric_cols.append(byproduct_col)
    if labor_col:
        numeric_cols.append(labor_col)
    for col in numeric_cols:
        d[col] = _to_numeric_series(d[col])

    d["_수량10a"] = d[quantity_col] * factor
    d["_주산물가액"] = d[quantity_col] * d[price_col] * factor
    d["_부산물가액"] = d[byproduct_col] * factor if byproduct_col else 0.0
    for col in cost_cols:
        d[f"__비용10a__{col}"] = d[col] * factor
    d["_경영비"] = d[[f"__비용10a__{c}" for c in cost_cols]].sum(axis=1)
    d["_조성비상각"] = float(establishment_amort_per_10a)
    d["_토지임차료"] = float(land_cash_rent_per_10a)
    d["_경영비"] += d["_조성비상각"] + d["_토지임차료"]
    d["_자가노동시간10a"] = d[labor_col] * factor if labor_col else 0.0
    d["_자가노력비"] = d["_자가노동시간10a"] * float(wage_per_hour)

    # 농촌진흥청 농산물소득조사 방식: 유동자본은 감가상각 성격의 비용을 제외한
    # 실제 유동자본액에 연이자율 × 산출계수 1/2 × 재포기간(월/12)을 적용한다.
    dep_cols = list(depreciation_cost_cols or [])
    if not dep_cols:
        dep_cols = [c for c in cost_cols if is_depreciation_cost(c)]
    dep_cols = [c for c in dep_cols if c in cost_cols]
    if dep_cols:
        d["_감가상각제외액"] = d[[f"__비용10a__{c}" for c in dep_cols]].sum(axis=1)
    else:
        d["_감가상각제외액"] = 0.0
    # 별도로 더한 조성비상각도 유동자본으로 보지 않는다.
    d["_감가상각제외액"] += float(establishment_amort_per_10a)
    d["_유동자본기준액"] = (d["_경영비"] - d["_감가상각제외액"]).clip(lower=0)
    d["_유동자본용역비"] = (d["_유동자본기준액"] * float(interest_rate) / 100
                         * 0.5 * (float(capital_months) / 12))

    # 고정자본은 부분현재가(또는 현재 평가액)에 해당 작목 부담률을 적용한 금액에
    # 연이자율을 곱한다.
    d["_고정자본기준액"] = (float(fixed_asset_per_10a)
                         * float(fixed_asset_use_rate_percent) / 100.0)
    d["_고정자본용역비"] = d["_고정자본기준액"] * float(interest_rate) / 100
    d["_토지용역비"] = float(land_cost_per_10a)
    d["_총수입"] = d["_주산물가액"] + d["_부산물가액"]
    d["_생산비"] = (d["_경영비"] + d["_자가노력비"] + d["_유동자본용역비"]
                 + d["_고정자본용역비"] + d["_토지용역비"])
    d["_소득"] = d["_총수입"] - d["_경영비"]
    d["_순수익"] = d["_총수입"] - d["_생산비"]

    group = d.groupby(treatment_col, sort=False, dropna=False)
    source_cols = [price_col, "_수량10a", "_주산물가액", "_부산물가액", "_경영비",
                   "_조성비상각", "_토지임차료", "_자가노동시간10a", "_자가노력비", "_감가상각제외액",
                   "_유동자본기준액", "_유동자본용역비", "_고정자본기준액",
                   "_고정자본용역비", "_토지용역비", "_총수입", "_생산비",
                   "_소득", "_순수익"]
    source_cols += [f"__비용10a__{c}" for c in cost_cols]
    summary = group[source_cols].mean().reset_index()
    summary["반복수"] = summary[treatment_col].map(group.size()).astype(int)
    summary = summary.rename(columns={f"__비용10a__{c}": c for c in cost_cols})
    if labor_col:
        summary[labor_col] = summary["_자가노동시간10a"]
    return d, summary


def validate_economic_results(row_data: pd.DataFrame, summary: pd.DataFrame,
                              tolerance: float = 1e-6) -> list[str]:
    """경제성 결과의 항등식을 행·처리 요약 양쪽에서 역산 검증한다."""
    errors: list[str] = []
    relations = [
        ("총수입", "_총수입", row_data["_주산물가액"] + row_data["_부산물가액"]),
        ("생산비", "_생산비", row_data["_경영비"] + row_data["_자가노력비"]
         + row_data["_유동자본용역비"] + row_data["_고정자본용역비"]
         + row_data["_토지용역비"]),
        ("소득", "_소득", row_data["_총수입"] - row_data["_경영비"]),
        ("순수익", "_순수익", row_data["_총수입"] - row_data["_생산비"]),
    ]
    for label, column, expected in relations:
        if not np.allclose(row_data[column], expected, rtol=tolerance, atol=tolerance, equal_nan=False):
            errors.append(f"행별 {label} 역산 검증에 실패했습니다.")
    if "_유동자본기준액" in row_data.columns and "_감가상각제외액" in row_data.columns:
        expected_wc = (row_data["_경영비"] - row_data["_감가상각제외액"]).clip(lower=0)
        if not np.allclose(row_data["_유동자본기준액"], expected_wc,
                           rtol=tolerance, atol=tolerance, equal_nan=False):
            errors.append("행별 유동자본 기준액 역산 검증에 실패했습니다.")
    for col in ["_수량10a", "_주산물가액", "_부산물가액", "_경영비", "_생산비",
                "_총수입", "_소득", "_순수익"]:
        if col in summary.columns and not np.isfinite(summary[col].to_numpy(dtype=float)).all():
            errors.append(f"처리 요약의 {col.lstrip('_')}에 NaN 또는 무한대가 있습니다.")
    return errors


_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
                   ast.Mult: operator.mul, ast.Div: operator.truediv}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def parse_calculation_basis(text) -> Optional[float]:
    """산출근거의 단순 사칙연산을 eval 없이 계산한다. 빈 값은 None."""
    raw = str(text or "").strip()
    if not raw:
        return None
    expr = raw.replace(",", "").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"\([^()]*[가-힣A-Za-z][^()]*\)", "", expr)
    expr = re.sub(r"[가-힣A-Za-z㎡%]+", " ", expr)
    expr = re.sub(r"[^0-9.()+\-*/\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    if not expr:
        raise ValueError("산출근거에서 계산식을 찾을 수 없습니다.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("산출근거의 계산식 형식이 올바르지 않습니다.") from exc

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("산출근거에서 0으로 나눌 수 없습니다.")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError("산출근거에는 숫자와 사칙연산만 사용할 수 있습니다.")

    value = float(evaluate(tree))
    if not math.isfinite(value):
        raise ValueError("산출근거 계산 결과가 유한한 숫자가 아닙니다.")
    return value


def validate_manual_budget_table(table: pd.DataFrame, table_name: str):
    """수동 부분예산 표의 금액을 엄격하게 검사하고 산출근거를 검산한다."""
    required = ["항목", "산출근거", "금액(원)"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        return pd.DataFrame(columns=required), [f"{table_name}에 필요한 열이 없습니다: {', '.join(missing)}"], pd.DataFrame()
    work = table[required].copy()
    nonblank = (work["항목"].fillna("").astype(str).str.strip().ne("")
                | work["산출근거"].fillna("").astype(str).str.strip().ne("")
                | work["금액(원)"].notna())
    work = work.loc[nonblank].copy()
    errors: list[str] = []
    checks: list[dict] = []
    valid_rows = []
    for display_row, (idx, row) in enumerate(work.iterrows(), start=1):
        item = str(row.get("항목", "") or "").strip()
        basis = str(row.get("산출근거", "") or "").strip()
        if not item:
            errors.append(f"{table_name} {display_row}행의 항목명이 비어 있습니다.")
            continue
        amount_raw = row.get("금액(원)")
        try:
            amount = float(str(amount_raw).replace(",", ""))
        except (TypeError, ValueError):
            errors.append(f"{table_name} {display_row}행 '{item}'의 금액이 숫자가 아닙니다.")
            continue
        if not math.isfinite(amount):
            errors.append(f"{table_name} {display_row}행 '{item}'의 금액에 NaN 또는 무한대를 사용할 수 없습니다.")
            continue
        if amount < 0:
            errors.append(f"{table_name} {display_row}행 '{item}'의 금액은 음수일 수 없습니다.")
            continue
        expected = None
        difference = None
        judgment = "수동 입력"
        if basis:
            try:
                expected = parse_calculation_basis(basis)
                difference = amount - expected
                tolerance = max(1.0, abs(expected) * 1e-9)
                judgment = "PASS" if abs(difference) <= tolerance else "확인 필요"
                if judgment == "확인 필요":
                    errors.append(
                        f"{table_name} {display_row}행 '{item}'의 입력 금액이 산출근거 계산값과 "
                        f"{round_half_up(difference):,}원 차이 납니다.")
            except ValueError as exc:
                errors.append(f"{table_name} {display_row}행 '{item}'의 산출근거 오류: {exc}")
                continue
        clean = row.copy()
        clean["항목"] = item
        clean["산출근거"] = basis
        clean["금액(원)"] = amount
        valid_rows.append(clean)
        checks.append({"구분": table_name, "항목": item, "산출근거 계산값": expected,
                       "입력 금액": amount, "차이": difference, "판정": judgment})
    cleaned = pd.DataFrame(valid_rows, columns=required)
    return cleaned, errors, pd.DataFrame(checks)


def calculate_partial_budget(data: pd.DataFrame, treatment_col, quantity_col, price_col,
                             variable_cost_cols: Iterable, byproduct_col=None,
                             source_area_a=10.0, adjustment_percent=10.0):
    """반복별 부분예산을 계산한 뒤 처리 평균으로 집계한다."""
    variable_cost_cols = list(variable_cost_cols or [])
    if treatment_col in (quantity_col, price_col):
        raise ValueError("처리구 열을 수량 또는 단가 열로 사용할 수 없습니다.")
    if quantity_col == price_col:
        raise ValueError("수량 열과 단가 열은 서로 달라야 합니다.")
    if not variable_cost_cols:
        raise ValueError("가변비용 열을 하나 이상 선택해야 합니다.")
    if any(is_excluded_cost(c) for c in variable_cost_cols):
        bad = [str(c) for c in variable_cost_cols if is_excluded_cost(c)]
        raise ValueError("부분예산에 합계·계산결과 열을 사용할 수 없습니다: " + ", ".join(bad))
    area = float(source_area_a)
    adjustment = float(adjustment_percent)
    if not math.isfinite(area) or area <= 0:
        raise ValueError("자료 기준 면적은 0보다 큰 유한한 숫자여야 합니다.")
    if not math.isfinite(adjustment) or not (0 <= adjustment < 100):
        raise ValueError("수량 조정률은 0 이상 100 미만이어야 합니다.")
    columns = [treatment_col, quantity_col, price_col] + variable_cost_cols
    if byproduct_col:
        columns.append(byproduct_col)
    columns = list(dict.fromkeys(columns))
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError("자료에 없는 열: " + ", ".join(map(str, missing)))
    d = data[columns].copy()
    if d[treatment_col].isna().any() or d[treatment_col].astype(str).str.strip().eq("").any():
        raise ValueError("처리구 열에 결측 또는 빈 값이 있습니다.")
    numeric_cols = [quantity_col, price_col] + variable_cost_cols + ([byproduct_col] if byproduct_col else [])
    for col in numeric_cols:
        d[col] = _to_numeric_series(d[col])
        if d[col].isna().any():
            raise ValueError(f"'{col}' 열에 결측 또는 숫자로 변환할 수 없는 값이 있습니다.")
        if not np.isfinite(d[col].to_numpy(dtype=float)).all():
            raise ValueError(f"'{col}' 열에 무한대가 있습니다.")
        if (d[col] < 0).any():
            raise ValueError(f"'{col}' 열에 음수가 있습니다.")

    factor = 10.0 / area
    d["_조정전수량"] = d[quantity_col] * factor
    d["_조정수량"] = d["_조정전수량"] * (1 - adjustment / 100.0)
    d["_주산물편익"] = d["_조정수량"] * d[price_col]
    d["_부산물편익"] = d[byproduct_col] * factor if byproduct_col else 0.0
    d["_총편익"] = d["_주산물편익"] + d["_부산물편익"]
    d["_가변비용"] = d[variable_cost_cols].sum(axis=1) * factor
    d["_순편익"] = d["_총편익"] - d["_가변비용"]
    repetitions = d.groupby(treatment_col, sort=False).size()
    summary = (d.groupby(treatment_col, as_index=False, sort=False)
               .agg(**{quantity_col: (quantity_col, "mean"),
                       price_col: (price_col, "mean"),
                       "조정 전 수량": ("_조정전수량", "mean"),
                       "조정수량": ("_조정수량", "mean"),
                       "주산물 편익": ("_주산물편익", "mean"),
                       "부산물 편익": ("_부산물편익", "mean"),
                       "총편익": ("_총편익", "mean"),
                       "가변비용": ("_가변비용", "mean"),
                       "순편익": ("_순편익", "mean")}))
    summary["반복수"] = summary[treatment_col].map(repetitions).astype(int)
    return d, summary, 0


def validate_partial_budget_results(row_data: pd.DataFrame, tolerance: float = 1e-6) -> list[str]:
    """부분예산 핵심 산식의 항등식을 검증한다."""
    errors = []
    checks = [
        ("총편익", row_data["_총편익"], row_data["_주산물편익"] + row_data["_부산물편익"]),
        ("순편익", row_data["_순편익"], row_data["_총편익"] - row_data["_가변비용"]),
    ]
    for label, actual, expected in checks:
        if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            errors.append(f"부분예산 {label} 검증에 실패했습니다.")
    return errors


def perform_dominance_analysis(summary: pd.DataFrame, treatment_col,
                               cost_col="가변비용", benefit_col="순편익",
                               control=None, tolerance=1e-9):
    """동일비용·단순지배·확장지배를 제거해 효율경계를 만든다."""
    required = [treatment_col, cost_col, benefit_col]
    missing = [c for c in required if c not in summary.columns]
    if missing:
        raise ValueError("지배분석에 필요한 열이 없습니다: " + ", ".join(map(str, missing)))
    data = summary.copy()
    for col in (cost_col, benefit_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any() or not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"'{col}'에 결측 또는 무한대가 있습니다.")
    data = data.sort_values([cost_col, benefit_col], ascending=[True, False]).reset_index(drop=True)
    status = [""] * len(data)
    reason = [""] * len(data)
    for _, group in data.groupby(cost_col, sort=False):
        if len(group) > 1:
            keep = group[benefit_col].idxmax()
            for idx in group.index:
                if idx != keep:
                    status[idx] = "D(동일비용)"
                    reason[idx] = "같은 가변비용에서 순편익이 더 낮음"
    best = -np.inf
    for idx in data.index:
        if status[idx]:
            continue
        value = float(data.loc[idx, benefit_col])
        if value <= best + tolerance:
            status[idx] = "D(단순지배)"
            reason[idx] = "더 높은 비용에도 순편익이 이전 처리 이하"
        else:
            best = value
    changed = True
    while changed:
        changed = False
        indices = [i for i in data.index if not status[i]]
        for position in range(1, len(indices) - 1):
            i1, i2, i3 = indices[position - 1], indices[position], indices[position + 1]
            c1, c2, c3 = map(float, [data.loc[i1, cost_col], data.loc[i2, cost_col], data.loc[i3, cost_col]])
            n1, n2, n3 = map(float, [data.loc[i1, benefit_col], data.loc[i2, benefit_col], data.loc[i3, benefit_col]])
            if c3 <= c1 or c2 <= c1 or c3 <= c2:
                continue
            frontier = n1 + (n3 - n1) * (c2 - c1) / (c3 - c1)
            if n2 < frontier - tolerance:
                status[i2] = "D(확장지배)"
                reason[i2] = "인접 효율처리를 잇는 효율경계보다 순편익이 낮음"
                changed = True
                break
    data["지배"] = status
    data["지배 사유"] = reason
    data["기준 처리"] = np.where(data[treatment_col].astype(str) == str(control), "예", "")
    return data


def calculate_mrr_table(dominance_df: pd.DataFrame, treatment_col,
                        cost_col="가변비용", benefit_col="순편익",
                        minimum_mrr=100.0, control=None):
    """비지배 처리의 연속 구간별 비용·순편익 증가액과 MRR을 계산한다."""
    minimum = float(minimum_mrr)
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("최소 MRR 기준은 0 이상의 유한한 숫자여야 합니다.")
    undominated = dominance_df[dominance_df["지배"].astype(str).eq("")].copy()
    undominated = undominated.sort_values(cost_col).reset_index(drop=True)
    cost_change, benefit_change, mrr = [np.nan], [np.nan], [np.nan]
    for i in range(1, len(undominated)):
        dc = float(undominated.loc[i, cost_col] - undominated.loc[i - 1, cost_col])
        db = float(undominated.loc[i, benefit_col] - undominated.loc[i - 1, benefit_col])
        cost_change.append(dc)
        benefit_change.append(db)
        mrr.append(db / dc * 100.0 if dc > 0 else np.nan)
    undominated["비용 증가액"] = cost_change
    undominated["순편익 증가액"] = benefit_change
    undominated["MRR(%)"] = mrr
    judgments = []
    for i, row in undominated.iterrows():
        if i == 0:
            judgments.append("기준 처리" if str(row[treatment_col]) == str(control) else "효율경계 시작점")
            continue
        dc, db, ratio = row["비용 증가액"], row["순편익 증가액"], row["MRR(%)"]
        if pd.isna(dc) or float(dc) <= 0:
            judgments.append("계산 제외(비용 증가액이 0 이하)")
        elif pd.isna(db) or float(db) <= 0:
            judgments.append("비권장(추가 비용에도 순편익 증가 없음)")
        elif float(row[benefit_col]) <= 0:
            judgments.append("비권장(순편익이 0 이하)")
        elif pd.isna(ratio):
            judgments.append("계산 불가")
        elif float(ratio) >= minimum:
            judgments.append(f"권장 후보 (MRR {ratio:.0f}% ≥ 기준 {minimum:.0f}%)")
        else:
            judgments.append(f"기준미달 (MRR {ratio:.0f}% < {minimum:.0f}%)")
    undominated["권장 여부 및 근거"] = judgments
    return undominated


def _investment_npv(cashflows, rate_percent: float) -> float:
    """연도 0부터의 순현금흐름 현재가치 합계."""
    rate = float(rate_percent) / 100.0
    if rate <= -1:
        raise ValueError("할인율은 -100%보다 커야 합니다.")
    return float(sum(float(cf) / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows)))


def _solve_irr(cashflows, max_rate=1_000_000.0):
    """NPV=0을 만드는 IRR(%)을 이분법으로 계산. 해가 없으면 NaN."""
    flows = [float(x) for x in cashflows]
    if not flows or not (any(x < 0 for x in flows) and any(x > 0 for x in flows)):
        return np.nan

    def f(r):
        return _investment_npv(flows, r)

    lo = -99.9999
    flo = f(lo)
    # 10%, 20% ... 식으로 상한을 확장해 부호가 바뀌는 구간을 찾는다.
    candidates = [10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 100000, max_rate]
    hi = None
    fhi = None
    for cand in candidates:
        val = f(cand)
        if flo == 0:
            return lo
        if val == 0:
            return float(cand)
        if flo * val < 0:
            hi, fhi = float(cand), val
            break
    if hi is None:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) < 1e-8 or abs(hi - lo) < 1e-9:
            return float(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float((lo + hi) / 2.0)


def calculate_investment_analysis(
    initial_investment,
    life_years,
    discount_rate,
    annual_benefit,
    annual_operating_cost,
    salvage_value=0,
    annual_benefit_growth_percent=0.0,
    annual_cost_growth_percent=0.0,
):
    """시설·농기계 등 장기투자의 NPV, 할인 B/C, IRR, 회수기간을 계산한다.

    비용·편익은 연말 발생을 기본으로 하며, 최초투자비는 0년차에 전액 지출한다.
    잔존가치는 마지막 연도의 편익으로 처리한다.
    """
    vals = {
        "최초투자비": initial_investment, "내용연수": life_years,
        "할인율": discount_rate, "연간편익": annual_benefit,
        "연간운영비": annual_operating_cost, "잔존가치": salvage_value,
        "편익증가율": annual_benefit_growth_percent,
        "비용증가율": annual_cost_growth_percent,
    }
    nums = {}
    for k, v in vals.items():
        try:
            nums[k] = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{k}은(는) 숫자여야 합니다.") from exc
        if not math.isfinite(nums[k]):
            raise ValueError(f"{k}에 NaN 또는 무한대를 사용할 수 없습니다.")
    years = int(nums["내용연수"])
    if years < 1 or abs(nums["내용연수"] - years) > 1e-9:
        raise ValueError("내용연수는 1 이상의 정수여야 합니다.")
    if nums["최초투자비"] < 0 or nums["연간편익"] < 0 or nums["연간운영비"] < 0 or nums["잔존가치"] < 0:
        raise ValueError("투자비·편익·운영비·잔존가치는 음수일 수 없습니다.")
    if nums["할인율"] <= -100:
        raise ValueError("할인율은 -100%보다 커야 합니다.")
    if nums["편익증가율"] <= -100 or nums["비용증가율"] <= -100:
        raise ValueError("연간 증가율은 -100%보다 커야 합니다.")

    r = nums["할인율"] / 100.0
    bg = nums["편익증가율"] / 100.0
    cg = nums["비용증가율"] / 100.0
    rows = [{"연도": 0, "편익": 0.0, "비용": nums["최초투자비"],
             "순현금흐름": -nums["최초투자비"], "할인계수": 1.0,
             "편익현재가": 0.0, "비용현재가": nums["최초투자비"],
             "순현재가": -nums["최초투자비"]}]
    cashflows = [-nums["최초투자비"]]
    for year in range(1, years + 1):
        benefit = nums["연간편익"] * ((1.0 + bg) ** (year - 1))
        cost = nums["연간운영비"] * ((1.0 + cg) ** (year - 1))
        if year == years:
            benefit += nums["잔존가치"]
        net = benefit - cost
        disc = 1.0 / ((1.0 + r) ** year)
        rows.append({"연도": year, "편익": benefit, "비용": cost,
                     "순현금흐름": net, "할인계수": disc,
                     "편익현재가": benefit * disc, "비용현재가": cost * disc,
                     "순현재가": net * disc})
        cashflows.append(net)

    table = pd.DataFrame(rows)
    pv_benefit = float(table["편익현재가"].sum())
    pv_cost = float(table["비용현재가"].sum())
    npv = float(table["순현재가"].sum())
    bcr = pv_benefit / pv_cost if pv_cost > 0 else np.nan
    irr = _solve_irr(cashflows)

    cumulative = table["순현금흐름"].cumsum().to_numpy(dtype=float)
    cumulative_disc = table["순현재가"].cumsum().to_numpy(dtype=float)

    def payback(cum, flows):
        for year in range(1, len(cum)):
            if cum[year] >= 0:
                prev = cum[year - 1]
                inc = float(flows[year])
                if inc <= 0:
                    return float(year)
                frac = max(0.0, min(1.0, -prev / inc))
                return float(year - 1 + frac)
        return np.nan

    simple_pb = payback(cumulative, table["순현금흐름"].to_numpy(dtype=float))
    disc_pb = payback(cumulative_disc, table["순현재가"].to_numpy(dtype=float))
    summary = {
        "NPV": npv,
        "할인 B/C": bcr,
        "IRR(%)": irr,
        "단순 회수기간(년)": simple_pb,
        "할인 회수기간(년)": disc_pb,
        "편익 현재가": pv_benefit,
        "비용 현재가": pv_cost,
        "경제성 판정": "경제성 있음" if npv > 0 else ("경제성 없음" if npv < 0 else "경계"),
    }
    return table, summary


def investment_sensitivity_table(initial_investment, life_years, discount_rate,
                                 annual_benefit, annual_operating_cost, salvage_value=0,
                                 change_rates=(-20, -10, 0, 10, 20)):
    """장기투자의 편익·운영비 변동에 따른 NPV 민감도 표."""
    rows = []
    for bchg in change_rates:
        row = {"편익 변동": f"{int(bchg):+d}%"}
        for cchg in change_rates:
            _, res = calculate_investment_analysis(
                initial_investment, life_years, discount_rate,
                float(annual_benefit) * (1 + float(bchg) / 100.0),
                float(annual_operating_cost) * (1 + float(cchg) / 100.0),
                salvage_value=salvage_value)
            row[f"비용 {int(cchg):+d}%"] = res["NPV"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_economic_self_test() -> pd.DataFrame:
    """앱 안에서 빠르게 실행할 수 있는 핵심 경제성 자가진단."""
    rows = []
    def record(name, func):
        try:
            func()
            rows.append({"테스트": name, "결과": "PASS", "상세": ""})
        except Exception as exc:  # 테스트 결과를 화면에 보여주기 위한 수집
            rows.append({"테스트": name, "결과": "FAIL", "상세": str(exc)[:120]})

    sample = pd.DataFrame({"처리": ["A", "A", "B", "B"], "수량": [100, 110, 120, 125],
                           "단가": [1000, 1000, 1000, 1000], "비료비": [10000, 11000, 13000, 14000]})
    def normal_case():
        row, summary = calculate_row_economics(sample, "처리", "수량", "단가", ["비료비"])
        assert not validate_economic_results(row, summary)
    def area_case():
        row, _ = calculate_row_economics(pd.DataFrame({"처리": ["A"], "수량": [50], "단가": [1000], "비용": [1000]}),
                                         "처리", "수량", "단가", ["비용"], source_area_a=5)
        assert row["_수량10a"].iloc[0] == 100
    def invalid_case():
        try:
            calculate_row_economics(pd.DataFrame({"처리": ["A"], "수량": [-1], "단가": [1000], "비용": [1]}),
                                    "처리", "수량", "단가", ["비용"])
        except ValueError:
            return
        raise AssertionError("음수 수량이 차단되지 않았습니다.")
    def manual_case():
        value = parse_calculation_basis("130,000원 × 10명")
        assert value == 1300000
    def rounding_case():
        assert round_half_up(2.5) == 3
    def partial_case():
        row, _, _ = calculate_partial_budget(sample, "처리", "수량", "단가", ["비료비"])
        assert not validate_partial_budget_results(row)
    def working_capital_case():
        d0 = pd.DataFrame({"처리":["A"], "수량":[1], "단가":[1],
                           "재료비":[800.0], "감가상각비":[200.0]})
        row, _ = calculate_row_economics(
            d0, "처리", "수량", "단가", ["재료비", "감가상각비"],
            interest_rate=12, capital_months=12,
            depreciation_cost_cols=["감가상각비"])
        assert abs(float(row["_유동자본기준액"].iloc[0]) - 800.0) < 1e-9
        assert abs(float(row["_유동자본용역비"].iloc[0]) - 48.0) < 1e-9
    def fixed_capital_case():
        d0 = pd.DataFrame({"처리":["A"], "수량":[1], "단가":[1], "비용":[1]})
        row, _ = calculate_row_economics(
            d0, "처리", "수량", "단가", ["비용"], interest_rate=5,
            fixed_asset_per_10a=100000, fixed_asset_use_rate_percent=50)
        assert abs(float(row["_고정자본용역비"].iloc[0]) - 2500.0) < 1e-9
    def investment_case():
        _, res = calculate_investment_analysis(1000, 3, 0, 600, 100)
        assert abs(float(res["NPV"]) - 500.0) < 1e-9
        assert abs(float(res["할인 B/C"]) - (1800/1300)) < 1e-9
    def land_rent_case():
        d0 = pd.DataFrame({"처리":["A"], "수량":[10], "단가":[1000], "비용":[1000]})
        row, _ = calculate_row_economics(
            d0, "처리", "수량", "단가", ["비용"], land_cash_rent_per_10a=2000,
            land_cost_per_10a=3000, interest_rate=0)
        assert abs(float(row["_경영비"].iloc[0]) - 3000.0) < 1e-9
        assert abs(float(row["_토지용역비"].iloc[0]) - 3000.0) < 1e-9
        assert abs(float(row["_소득"].iloc[0]) - 7000.0) < 1e-9
        assert abs(float(row["_생산비"].iloc[0]) - 6000.0) < 1e-9

    for name, func in [("정상 소득계산", normal_case), ("5a→10a 환산", area_case),
                       ("음수 입력 차단", invalid_case), ("산출근거 파싱", manual_case),
                       ("ROUND_HALF_UP", rounding_case), ("부분예산 역산", partial_case),
                       ("유동자본 1/2·상각제외", working_capital_case),
                       ("고정자본 작목부담률", fixed_capital_case),
                       ("장기투자 NPV/B-C", investment_case),
                       ("임차료 경영비·자가토지 기회비용", land_rent_case)]:
        record(name, func)
    return pd.DataFrame(rows)
