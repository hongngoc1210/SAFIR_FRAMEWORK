from __future__ import annotations

import datetime
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Flowable,
    KeepInFrame,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from huggingface_hub import InferenceClient

    _HAS_HF = True
except Exception:
    InferenceClient = None
    _HAS_HF = False


# -----------------------------------------------------------------------------
# Font setup
# -----------------------------------------------------------------------------
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception:
    pass


# -----------------------------------------------------------------------------
# Design tokens
# -----------------------------------------------------------------------------
NAVY = colors.HexColor("#0B1736")
NAVY_2 = colors.HexColor("#16213E")
ACCENT = colors.HexColor("#2F80ED")
BG = colors.HexColor("#F6F8FC")
CARD_BG = colors.HexColor("#FFFFFF")
SOFT_BLUE = colors.HexColor("#EAF2FF")
SOFT_GREEN = colors.HexColor("#EAF7EF")
SOFT_RED = colors.HexColor("#FDECEC")
SOFT_AMBER = colors.HexColor("#FFF6E5")
BORDER = colors.HexColor("#D9E2EC")
TEXT = colors.HexColor("#111827")
MUTED = colors.HexColor("#667085")
GREEN = colors.HexColor("#138A36")
RED = colors.HexColor("#C92A2A")
AMBER = colors.HexColor("#B7791F")
LIGHT_GREY = colors.HexColor("#EEF2F6")

PAGE_W, PAGE_H = A4

FACTOR_NAMES = [
    "Market Factor",
    "Size Factor",
    "Valuation (BP) Factor",
    "Profitability Factor",
    "Investment Factor",
    "News Effect Factor",
]


# -----------------------------------------------------------------------------
# Text / numeric helpers
# -----------------------------------------------------------------------------
def _safe_text(value: object) -> str:
    """Escape text for ReportLab Paragraph and remove control characters."""
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.escape(text).replace("\n", "<br/>")


def _truncate_text(text: object, max_chars: int = 520) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _contains_cjk(text: object) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _pct(value: float, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "N/A"


def _signed_pct(value: float, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:+.{digits}f}%"
    except Exception:
        return "N/A"


def _num(value: float, digits: int = 4) -> str:
    try:
        if not math.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _clip(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return 0.0


def _extract_balanced_json(raw: str) -> str:
    """Return the first balanced JSON object from mixed LLM text."""
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", raw, 0)

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(raw)):
        ch = raw[i]

        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]

    raise json.JSONDecodeError("Unbalanced JSON object", raw, start)


def _json_from_text(raw: str) -> Dict:
    """Parse JSON returned by an LLM, including Qwen outputs with thinking/text wrappers."""
    raw = (raw or "").strip()

    # Qwen thinking variants may prepend reasoning blocks. These break JSON parsing.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove common markdown wrappers anywhere in the response.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Normalize punctuation sometimes produced by multilingual models.
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    candidates = [raw]
    try:
        candidates.append(_extract_balanced_json(raw))
    except json.JSONDecodeError:
        pass

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        # Remove trailing commas before object/array endings.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise json.JSONDecodeError("Unable to parse JSON", raw, 0)


def _prediction_tone(prediction: str) -> Tuple[colors.Color, colors.Color, str]:
    p = str(prediction or "").strip().upper()
    if p in {"UP", "POSITIVE", "BULLISH", "RISE", "BUY"}:
        return GREEN, SOFT_GREEN, "Positive direction"
    if p in {"DOWN", "NEGATIVE", "BEARISH", "FALL", "SELL"}:
        return RED, SOFT_RED, "Negative direction"
    return AMBER, SOFT_AMBER, "Neutral / uncertain direction"


def _risk_level(var_estimate: float, lang: str) -> Tuple[str, colors.Color, colors.Color]:
    var_abs = abs(float(var_estimate or 0.0))
    if var_abs < 0.02:
        return ("低风险" if lang == "zh" else "Low risk"), GREEN, SOFT_GREEN
    if var_abs < 0.05:
        return ("中等风险" if lang == "zh" else "Moderate risk"), AMBER, SOFT_AMBER
    return ("高风险" if lang == "zh" else "High risk"), RED, SOFT_RED


def _return_bias(ret_low: float, ret_high: float, lang: str) -> Tuple[str, colors.Color, colors.Color]:
    try:
        mid = (float(ret_low) + float(ret_high)) / 2.0
    except Exception:
        mid = 0.0
    if mid > 0.001:
        return ("偏上行" if lang == "zh" else "Upside bias"), GREEN, SOFT_GREEN
    if mid < -0.001:
        return ("偏下行" if lang == "zh" else "Downside bias"), RED, SOFT_RED
    return ("区间震荡" if lang == "zh" else "Range-bound"), AMBER, SOFT_AMBER


def _factor_ordered(factors: Optional[Mapping[str, float]]) -> Dict[str, float]:
    factors = factors or {}
    ordered: Dict[str, float] = {}
    for name in FACTOR_NAMES:
        if name in factors:
            ordered[name] = float(factors[name])
    for name, value in factors.items():
        if name not in ordered:
            try:
                ordered[str(name)] = float(value)
            except Exception:
                ordered[str(name)] = 0.0
    return ordered


def _format_factors_for_prompt(factors: Mapping[str, float]) -> str:
    if not factors:
        return "- No factor contribution available."
    return "\n".join(f"- {name}: {_signed_pct(value, 2)}" for name, value in _factor_ordered(factors).items())


def _dominant_factors(factors: Mapping[str, float], n: int = 2) -> str:
    ordered = sorted(_factor_ordered(factors).items(), key=lambda kv: abs(kv[1]), reverse=True)
    if not ordered:
        return "No dominant factor available"
    return ", ".join(f"{name} ({_signed_pct(value, 2)})" for name, value in ordered[:n])


# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------
SAFIR_PROMPT_EN = """\
You are an institutional quantitative equity analyst and an Explainable AI reviewer.
Generate a compact one-page SAFIR model interpretation report for stock {stock_code}.

SAFIR stands for Structured Adaptive Financial Intelligence Report.

Important rules:
- Use ONLY the supplied news text and model outputs. Do not invent customers, contracts, macro events, management statements, or historical facts.
- Treat the model output as a probabilistic forecast, not a certainty.
- Explain the economic transmission chain: news event -> SAFIR factor attribution -> expected return/risk implication.
- If the prediction direction and factor contributions conflict, reconcile that tension explicitly.
- Avoid generic language. Each sentence must refer to the news, factor signs, confidence, VaR, or return interval.
- Do not provide direct buy/sell advice. Use risk-aware monitoring and positioning language.
- Keep every field concise because the PDF must fit exactly on one A4 page.

Input:
Stock code: {stock_code}
Analysis date: {date}
News text: {news_text}
Prediction: {prediction}
Model confidence: {confidence:.1%}
Expected return band: {ret_low:+.2%} to {ret_high:+.2%}
VaR estimate: {var_estimate:.2%}
Risk level: {risk_level}
Factor contributions:
{factors_str}
Dominant drivers: {dominant_factors}

Return ONLY valid JSON using this schema:
{{
  "executive_summary": "2 concise sentences combining direction, confidence, return band, and risk/reward balance.",
  "event_digest": "2 concise sentences explaining the financial catalyst in the news.",
  "quant_thesis": "2 concise sentences explaining the SAFIR factor logic and dominant drivers.",
  "factor_explanations": {{
    "Market Factor": "one specific sentence, max 22 words",
    "Size Factor": "one specific sentence, max 22 words",
    "Valuation (BP) Factor": "one specific sentence, max 22 words",
    "Profitability Factor": "one specific sentence, max 22 words",
    "Investment Factor": "one specific sentence, max 22 words",
    "News Effect Factor": "one specific sentence, max 22 words"
  }},
  "risk_assessment": "2 concise sentences explaining VaR and downside risk.",
  "outlook": "2 concise sentences with short-horizon scenario logic and risk-aware framing.",
  "monitoring_points": ["short watch item 1", "short watch item 2", "short watch item 3"],
  "model_caveat": "one sentence saying this is probabilistic and not investment advice"
}}
"""

SAFIR_PROMPT_ZH = """\
你是一位机构级量化股票分析师，也是一名 Explainable AI 审核专家。
请为股票 {stock_code} 生成一份紧凑的一页式 SAFIR 模型解释报告。

SAFIR 表示 Structured Adaptive Financial Intelligence Report。

重要规则：
- 只能使用给定新闻文本和模型输出，不要编造客户、合同、宏观事件、管理层表态或历史事实。
- 将模型输出视为概率预测，而不是确定性结论。
- 解释经济传导链条：新闻事件 -> SAFIR 因子归因 -> 预期收益/风险含义。
- 如果预测方向与因子贡献冲突，请明确解释这种张力。
- 避免空泛表达。每句话都要对应新闻、因子正负、置信度、VaR 或收益区间。
- 不要给出直接买入/卖出建议；使用风险约束下的观察与仓位表述。
- 每个字段保持简洁，因为 PDF 必须固定在一页 A4 内。

输入：
股票代码: {stock_code}
分析日期: {date}
新闻原文: {news_text}
预测方向: {prediction}
模型置信度: {confidence:.1%}
预期收益区间: {ret_low:+.2%} 至 {ret_high:+.2%}
VaR估计: {var_estimate:.2%}
风险等级: {risk_level}
因子贡献:
{factors_str}
主导因子: {dominant_factors}

请只输出合法 JSON，结构如下：
{{
  "executive_summary": "2句简洁摘要，综合方向、置信度、收益区间和风险收益比。",
  "event_digest": "2句简洁说明新闻中的财务催化事件。",
  "quant_thesis": "2句解释 SAFIR 因子逻辑和主导驱动。",
  "factor_explanations": {{
    "Market Factor": "一句具体解释，不超过22词",
    "Size Factor": "一句具体解释，不超过22词",
    "Valuation (BP) Factor": "一句具体解释，不超过22词",
    "Profitability Factor": "一句具体解释，不超过22词",
    "Investment Factor": "一句具体解释，不超过22词",
    "News Effect Factor": "一句具体解释，不超过22词"
  }},
  "risk_assessment": "2句解释 VaR 与下行风险。",
  "outlook": "2句给出短期情景逻辑和风险约束下的表述。",
  "monitoring_points": ["简短观察点1", "简短观察点2", "简短观察点3"],
  "model_caveat": "一句说明概率性和非投资建议属性"
}}
"""


class LLMEngine:
    """Prompt-backed SAFIR narrative generator using Qwen via Hugging Face.

    The engine uses Hugging Face InferenceClient when HF_TOKEN is available.
    If the token, provider, or selected model is unavailable, it falls back to
    the deterministic narrative so PDF generation never breaks.
    """

    def __init__(self, model: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.model = model or os.getenv("SAFIR_LLM_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.client = None

        if _HAS_HF and hf_token:
            try:
                self.client = InferenceClient(model=self.model, token=hf_token)
            except Exception:
                self.client = None

    def generate(
        self,
        *,
        stock_code: str,
        date: str,
        news_text: str,
        prediction: str,
        confidence: float,
        var_estimate: float,
        factors: Mapping[str, float],
        language: str = "en",
        ret_low: float = 0.0,
        ret_high: float = 0.0,
    ) -> Dict:
        factors = _factor_ordered(factors)
        if ret_low > ret_high:
            ret_low, ret_high = ret_high, ret_low

        risk_label, _, _ = _risk_level(var_estimate, language)
        prompt_template = SAFIR_PROMPT_ZH if language == "zh" else SAFIR_PROMPT_EN
        prompt = prompt_template.format(
            stock_code=stock_code,
            date=date,
            news_text=_truncate_text(news_text, 1100),
            prediction=prediction,
            confidence=float(confidence or 0.0),
            var_estimate=float(var_estimate or 0.0),
            ret_low=float(ret_low or 0.0),
            ret_high=float(ret_high or 0.0),
            risk_level=risk_label,
            factors_str=_format_factors_for_prompt(factors),
            dominant_factors=_dominant_factors(factors),
        )

        if self.client is None:
            return self._fallback(
                stock_code=stock_code,
                date=date,
                news_text=news_text,
                prediction=prediction,
                confidence=confidence,
                var_estimate=var_estimate,
                factors=factors,
                language=language,
                ret_low=ret_low,
                ret_high=ret_high,
            )

        try:
            system_msg = (
                "You are a strict JSON generator. Return exactly one valid JSON object. "
                "Do not include markdown, code fences, comments, explanations, or <think> blocks. "
                "The first character must be { and the last character must be }."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]

            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.1,
                )
            except Exception:
                # Some providers reject response_format even though chat_completion works.
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.1,
                )

            raw = response.choices[0].message.content.strip()
            if os.getenv("SAFIR_DEBUG_LLM", "0") == "1":
                print("[SAFIR LLM RAW BEGIN]")
                print(raw[:3000])
                print("[SAFIR LLM RAW END]")
            parsed = _json_from_text(raw)
            return self._normalize(
                parsed,
                stock_code=stock_code,
                date=date,
                news_text=news_text,
                prediction=prediction,
                confidence=confidence,
                var_estimate=var_estimate,
                factors=factors,
                language=language,
                ret_low=ret_low,
                ret_high=ret_high,
            )
        except Exception as exc:
            print(f"[SAFIR LLM] {type(exc).__name__}: {exc} - using deterministic fallback.")
            return self._fallback(
                stock_code=stock_code,
                date=date,
                news_text=news_text,
                prediction=prediction,
                confidence=confidence,
                var_estimate=var_estimate,
                factors=factors,
                language=language,
                ret_low=ret_low,
                ret_high=ret_high,
            )

    def _normalize(
        self,
        data: Dict,
        *,
        stock_code: str,
        date: str,
        news_text: str,
        prediction: str,
        confidence: float,
        var_estimate: float,
        factors: Mapping[str, float],
        language: str,
        ret_low: float,
        ret_high: float,
    ) -> Dict:
        fallback = self._fallback(
            stock_code=stock_code,
            date=date,
            news_text=news_text,
            prediction=prediction,
            confidence=confidence,
            var_estimate=var_estimate,
            factors=factors,
            language=language,
            ret_low=ret_low,
            ret_high=ret_high,
        )
        out = fallback.copy()
        if isinstance(data, dict):
            for key in ["executive_summary", "event_digest", "quant_thesis", "risk_assessment", "outlook", "model_caveat"]:
                if data.get(key):
                    out[key] = _truncate_text(data[key], 360)
            fe = fallback.get("factor_explanations", {}).copy()
            if isinstance(data.get("factor_explanations"), dict):
                for name, text in data["factor_explanations"].items():
                    fe[str(name)] = _truncate_text(text, 180)
            out["factor_explanations"] = fe
            pts = data.get("monitoring_points") or fallback["monitoring_points"]
            if not isinstance(pts, list):
                pts = [str(pts)]
            out["monitoring_points"] = [_truncate_text(p, 95) for p in pts[:3]]
        while len(out["monitoring_points"]) < 3:
            out["monitoring_points"].append(fallback["monitoring_points"][len(out["monitoring_points"])])
        return out

    def _fallback(
        self,
        *,
        stock_code: str,
        date: str,
        news_text: str,
        prediction: str,
        confidence: float,
        var_estimate: float,
        factors: Mapping[str, float],
        language: str,
        ret_low: float,
        ret_high: float,
    ) -> Dict:
        factors = _factor_ordered(factors)
        pos = str(prediction).lower() in {"positive", "up", "bullish", "rise", "buy"}
        risk_label, _, _ = _risk_level(var_estimate, language)
        factor_sum = sum(factors.values()) if factors else 0.0
        if factors:
            top_name, top_value = max(factors.items(), key=lambda kv: abs(kv[1]))
        else:
            top_name, top_value = "News Effect Factor", 0.0

        def factor_sentence_en(name: str, value: float) -> str:
            direction = "supports" if value >= 0 else "pressures"
            mechanism = {
                "Market Factor": "market risk appetite and liquidity transmission",
                "Size Factor": "scale perception and operating reach",
                "Valuation (BP) Factor": "book-value re-rating pressure",
                "Profitability Factor": "earnings quality and margin visibility",
                "Investment Factor": "capital deployment and execution discipline",
                "News Effect Factor": "headline sentiment and attention-driven demand",
            }.get(name, "cross-factor attribution")
            return f"{name} {direction} the signal by {_signed_pct(value, 2)} through {mechanism}."

        def factor_sentence_zh(name: str, value: float) -> str:
            direction = "支撑" if value >= 0 else "压制"
            mechanism = {
                "Market Factor": "市场风险偏好与流动性传导",
                "Size Factor": "规模认知与经营覆盖面",
                "Valuation (BP) Factor": "账面价值相关的估值重定价",
                "Profitability Factor": "盈利质量与利润率可见度",
                "Investment Factor": "资本投放与执行纪律",
                "News Effect Factor": "新闻情绪与注意力驱动需求",
            }.get(name, "交叉因子归因")
            return f"{name} 以 {_signed_pct(value, 2)} 的贡献{direction}信号，主要通过{mechanism}体现。"

        if language == "zh":
            return {
                "executive_summary": (
                    f"SAFIR 对 {stock_code} 给出{'正向' if pos else '负向/谨慎'}短期概率判断，置信度为 {_pct(confidence, 1)}。"
                    f"预期收益区间为 {_signed_pct(ret_low, 2)} 至 {_signed_pct(ret_high, 2)}，VaR 为 {_pct(abs(var_estimate), 2)}，风险等级为{risk_label}。"
                ),
                "event_digest": (
                    f"{date} 的新闻被视为公司层面的定价催化，可能改变市场对该股票的短期预期。"
                    "在缺少额外公告细节时，解释仅基于输入新闻和模型输出。"
                ),
                "quant_thesis": (
                    f"SAFIR 六类因子合计贡献为 {_signed_pct(factor_sum, 2)}，主导项为 {top_name}({_signed_pct(top_value, 2)})。"
                    "该结构说明新闻信号主要经由因子归因传导至收益与风险判断。"
                ),
                "factor_explanations": {name: factor_sentence_zh(name, value) for name, value in factors.items()},
                "risk_assessment": (
                    f"VaR 为 {_pct(abs(var_estimate), 2)}，代表当前设定下需要监控的下行损失边界。"
                    "因此该信号更适合结合仓位约束、波动变化和后续公告验证。"
                ),
                "outlook": (
                    f"短期情景偏向{'温和上行' if pos else '防御观察'}，但仍应视为事件驱动信号而非确定结论。"
                    "后续应重点确认价格成交、风险指标和新闻事实是否继续支持模型方向。"
                ),
                "monitoring_points": [
                    "公告后成交量与价格缺口是否确认方向",
                    "后续披露是否验证事件金额、时间表和执行路径",
                    "VaR 与实现波动率是否继续上升",
                ],
                "model_caveat": "SAFIR 输出为概率性模型解释，不构成投资建议。",
            }

        return {
            "executive_summary": (
                f"SAFIR assigns a {prediction.lower()} short-horizon view to {stock_code} with {_pct(confidence, 1)} confidence. "
                f"The expected return band is {_signed_pct(ret_low, 2)} to {_signed_pct(ret_high, 2)}, while VaR is {_pct(abs(var_estimate), 2)} and classified as {risk_label.lower()}."
            ),
            "event_digest": (
                f"The {date} news is treated as a company-specific catalyst that may alter short-term market expectations. "
                "Given limited disclosure context, the interpretation is restricted to the supplied news and model outputs."
            ),
            "quant_thesis": (
                f"The SAFIR factor aggregate is {_signed_pct(factor_sum, 2)}, led by {top_name} at {_signed_pct(top_value, 2)}. "
                "This suggests the news signal is transmitted through factor attribution into return and risk expectations."
            ),
            "factor_explanations": {name: factor_sentence_en(name, value) for name, value in factors.items()},
            "risk_assessment": (
                f"The VaR estimate of {_pct(abs(var_estimate), 2)} defines the downside band to monitor under the current setting. "
                "The signal should therefore be evaluated with position sizing, volatility changes, and subsequent disclosure quality."
            ),
            "outlook": (
                f"The short-horizon scenario leans toward {'controlled upside' if pos else 'defensive monitoring'}, but it remains an event-driven signal rather than a deterministic call. "
                "Confirmation should come from price-volume behavior, factor stability, and follow-up news evidence."
            ),
            "monitoring_points": [
                "Post-news volume, turnover, and price gap confirmation",
                "Follow-up disclosure on amount, timing, and execution path",
                "Changes in VaR and realized volatility over the next sessions",
            ],
            "model_caveat": "SAFIR is a probabilistic model explanation and does not constitute investment advice.",
        }


# -----------------------------------------------------------------------------
# Styles and drawing helpers
# -----------------------------------------------------------------------------
def _styles(lang: str, news_text: str = "") -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    cjk = lang == "zh" or _contains_cjk(news_text)

    body_font = "STSong-Light" if lang == "zh" else "Helvetica"
    table_font = "STSong-Light" if lang == "zh" else "Helvetica"
    news_font = "STSong-Light" if cjk else "Helvetica"
    title_font = "STSong-Light" if lang == "zh" else "Helvetica-Bold"

    return {
        "title": ParagraphStyle(
            "SAFIRTitle", parent=base["Title"], fontName=title_font, fontSize=17, leading=20,
            textColor=NAVY, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "SAFIRSubtitle", parent=base["BodyText"], fontName=body_font, fontSize=7.8, leading=9.5,
            textColor=MUTED, alignment=TA_LEFT, spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "SAFIRH1", parent=base["Heading1"], fontName=title_font, fontSize=9.4, leading=11.2,
            textColor=NAVY, spaceBefore=2, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "SAFIRBody", parent=base["BodyText"], fontName=body_font, fontSize=7.6, leading=9.6,
            textColor=TEXT, alignment=TA_LEFT, spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "SAFIRSmall", parent=base["BodyText"], fontName=body_font, fontSize=6.8, leading=8.6,
            textColor=MUTED, alignment=TA_LEFT,
        ),
        "micro": ParagraphStyle(
            "SAFIRMicro", parent=base["BodyText"], fontName=body_font, fontSize=6.2, leading=7.5,
            textColor=MUTED, alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "SAFIRTable", parent=base["BodyText"], fontName=table_font, fontSize=6.5, leading=8.1,
            textColor=TEXT,
        ),
        "table_header": ParagraphStyle(
            "SAFIRTableHeader", parent=base["BodyText"], fontName=table_font, fontSize=6.4, leading=7.5,
            textColor=colors.white,
        ),
        "card_label": ParagraphStyle(
            "SAFIRCardLabel", parent=base["BodyText"], fontName=body_font, fontSize=6.2, leading=7.3,
            textColor=MUTED, alignment=TA_LEFT,
        ),
        "card_value": ParagraphStyle(
            "SAFIRCardValue", parent=base["BodyText"], fontName=title_font, fontSize=11.8, leading=13.5,
            textColor=NAVY, alignment=TA_LEFT,
        ),
        "card_sub": ParagraphStyle(
            "SAFIRCardSub", parent=base["BodyText"], fontName=body_font, fontSize=6.0, leading=7.2,
            textColor=MUTED, alignment=TA_LEFT,
        ),
        "news": ParagraphStyle(
            "SAFIRNews", parent=base["BodyText"], fontName=news_font, fontSize=6.9, leading=8.6,
            textColor=TEXT, alignment=TA_LEFT, spaceAfter=1,
        ),
        "disclaimer": ParagraphStyle(
            "SAFIRDisclaimer", parent=base["BodyText"], fontName=body_font, fontSize=6.1, leading=7.3,
            textColor=MUTED, alignment=TA_CENTER,
        ),
    }


def _color_hex(color_obj: colors.Color) -> str:
    return color_obj.hexval()[2:]


def _value_style(name: str, base: ParagraphStyle, color: colors.Color, font_size: float = 11.8) -> ParagraphStyle:
    return ParagraphStyle(name, parent=base, textColor=color, fontSize=font_size, leading=font_size + 1.7)


def _draw_page_frame(canvas, doc):
    """SAFIR brand band and footer."""
    canvas.saveState()

    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.96 * cm, PAGE_W, 0.96 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - 0.96 * cm, 0.18 * cm, 0.96 * cm, fill=1, stroke=0)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(1.35 * cm, PAGE_H - 0.62 * cm, "SAFIR")
    canvas.setFont("Helvetica", 7.2)
    canvas.drawRightString(PAGE_W - 1.35 * cm, PAGE_H - 0.62 * cm, "Structured Adaptive Financial Intelligence Report")

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.35)
    canvas.line(1.35 * cm, 0.92 * cm, PAGE_W - 1.35 * cm, 0.92 * cm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(1.35 * cm, 0.62 * cm, "SAFIR model-generated explanation - research use only")
    canvas.drawRightString(PAGE_W - 1.35 * cm, 0.62 * cm, "One-page analyst brief")

    canvas.restoreState()


class ThinDivider(Flowable):
    def __init__(self, width: float = 15.8 * cm, color: colors.Color = BORDER, thickness: float = 0.4):
        super().__init__()
        self.width = width
        self.height = 0.08 * cm
        self.color = color
        self.thickness = thickness

    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.height / 2, self.width, self.height / 2)
        self.canv.restoreState()


def _progress_bar(value: float, color: colors.Color, width: float = 88, height: float = 6) -> Drawing:
    v = _clip(value, 0.0, 1.0)
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=LIGHT_GREY, strokeColor=None))
    d.add(Rect(0, 0, width * v, height, fillColor=color, strokeColor=None))
    return d


def _factor_bar(value: float, max_abs: float = 1.0, width: float = 64, height: float = 8) -> Drawing:
    v = _clip(float(value or 0.0), -max_abs, max_abs)
    center = width / 2
    d = Drawing(width, height)
    d.add(Rect(0, height / 2 - 0.8, width, 1.6, fillColor=LIGHT_GREY, strokeColor=None))
    d.add(Rect(center - 0.35, 0, 0.7, height, fillColor=BORDER, strokeColor=None))
    if v >= 0:
        bar_w = (v / max_abs) * center
        d.add(Rect(center, height / 2 - 1.9, bar_w, 3.8, fillColor=GREEN, strokeColor=None))
    else:
        bar_w = (abs(v) / max_abs) * center
        d.add(Rect(center - bar_w, height / 2 - 1.9, bar_w, 3.8, fillColor=RED, strokeColor=None))
    return d


def _badge(text: str, fg: colors.Color, bg: colors.Color, style: ParagraphStyle, width: float = 3.0 * cm) -> Table:
    value_style = ParagraphStyle(
        f"Badge_{re.sub(r'[^A-Za-z0-9]+', '_', str(text))}",
        parent=style,
        textColor=fg,
        alignment=TA_CENTER,
        fontSize=7.0,
        leading=8.4,
    )
    t = Table([[Paragraph(_safe_text(text), value_style)]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.3, fg),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]))
    return t


def _card(label: str, value: str, sub: str, s: Dict[str, ParagraphStyle], accent: colors.Color, soft_bg: colors.Color) -> Table:
    value_style = _value_style(f"CardValue_{re.sub(r'[^A-Za-z0-9]+', '_', label)}", s["card_value"], accent)
    t = Table([
        [Paragraph(_safe_text(label), s["card_label"])],
        [Paragraph(_safe_text(value), value_style)],
        [Paragraph(_safe_text(sub), s["card_sub"])],
    ], colWidths=[3.72 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("LINEABOVE", (0, 0), (0, 0), 1.8, accent),
        ("BACKGROUND", (0, 0), (0, 0), soft_bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 5.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _section_title(title: str, s: Dict[str, ParagraphStyle]) -> Table:
    t = Table([["", Paragraph(_safe_text(title), s["h1"])]], colWidths=[0.12 * cm, 15.58 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# -----------------------------------------------------------------------------
# Report blocks
# -----------------------------------------------------------------------------
def _header_block(lang: str, stock_code: str, date: str, prediction: str, confidence: float, s: Dict[str, ParagraphStyle]) -> Table:
    pred_fg, pred_bg, _ = _prediction_tone(prediction)

    if lang == "zh":
        title = "SAFIR 股票预测解释报告"
        subtitle = f"股票代码: {stock_code}    |    日期: {date}    |    Prompt-driven model interpretation"
    else:
        title = "SAFIR Stock Forecast Explanation Report"
        subtitle = f"Stock code: {stock_code}    |    Date: {date}    |    Prompt-driven model interpretation"

    left = [
        Paragraph(_safe_text(title), s["title"]),
        Paragraph(_safe_text(subtitle), s["subtitle"]),
    ]
    right = [
        _badge(str(prediction), pred_fg, pred_bg, s["table"], width=3.3 * cm),
        Spacer(1, 0.06 * cm),
        _progress_bar(float(confidence or 0.0), pred_fg, width=92, height=5.4),
        Paragraph(_safe_text(f"confidence {_pct(confidence, 1)}"), s["micro"]),
    ]

    t = Table([[left, right]], colWidths=[11.85 * cm, 3.85 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _make_kpi_cards(
    *,
    lang: str,
    prediction: str,
    confidence: float,
    var_estimate: float,
    ret_low: float,
    ret_high: float,
    s: Dict[str, ParagraphStyle],
) -> Table:
    pred_fg, pred_bg, pred_sub = _prediction_tone(prediction)
    risk_text, risk_fg, risk_bg = _risk_level(var_estimate, lang)
    ret_bias, ret_fg, ret_bg = _return_bias(ret_low, ret_high, lang)

    if lang == "zh":
        labels = ["预测方向", "置信度", "VaR 风险", "预期收益区间"]
        conf_sub = "Softmax probability"
        pred_sub = "模型分类输出"
    else:
        labels = ["Prediction", "Confidence", "VaR Estimate", "Expected Return"]
        conf_sub = "Softmax probability"

    cards = [
        _card(labels[0], str(prediction), pred_sub, s, pred_fg, pred_bg),
        _card(labels[1], _pct(confidence, 1), conf_sub, s, ACCENT, SOFT_BLUE),
        _card(labels[2], _pct(abs(var_estimate), 2), risk_text, s, risk_fg, risk_bg),
        _card(labels[3], f"{_signed_pct(ret_low, 2)} to {_signed_pct(ret_high, 2)}", ret_bias, s, ret_fg, ret_bg),
    ]
    row = Table([cards], colWidths=[3.925 * cm] * 4, hAlign="LEFT")
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row


def _make_text_box(title: str, text: str, s: Dict[str, ParagraphStyle], width: float = 15.7 * cm) -> Table:
    rows = [
        [Paragraph(f'<font color="#{_color_hex(MUTED)}">{_safe_text(title)}</font>', s["table_header"])],
        [Paragraph(_safe_text(text), s["body"])],
    ]
    t = Table(rows, colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BACKGROUND", (0, 0), (0, 0), BG),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def _make_news_box(news_text: str, s: Dict[str, ParagraphStyle], lang: str) -> Table:
    title = "Input News Evidence" if lang != "zh" else "输入新闻证据"
    fallback = "暂无新闻文本。" if lang == "zh" else "No news available for this date."
    content = _truncate_text(news_text or fallback, 420)
    t = Table([
        [Paragraph(f'<font color="#{_color_hex(MUTED)}">{_safe_text(title)}</font>', s["table_header"])],
        [Paragraph(_safe_text(content), s["news"])],
    ], colWidths=[15.7 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BACKGROUND", (0, 0), (0, 0), BG),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def _make_factor_table(
    factors: Mapping[str, float],
    factor_explanations: Mapping[str, str],
    s: Dict[str, ParagraphStyle],
    lang: str,
) -> Table:
    headers = ["因子", "贡献", "信号", "Prompt attribution"] if lang == "zh" else ["Factor", "Impact", "Signal", "Prompt attribution"]
    rows: List[List[object]] = [[Paragraph(h, s["table_header"]) for h in headers]]

    ordered = _factor_ordered(factors)
    if not ordered:
        empty = "暂无因子诊断数据" if lang == "zh" else "No factor diagnostics available"
        rows.append([Paragraph(empty, s["table"]), Paragraph("", s["table"]), Paragraph("", s["table"]), Paragraph("", s["table"])])
    else:
        max_abs = max([abs(float(v)) for v in ordered.values()] + [0.01])
        for name, value in ordered.items():
            v = float(value or 0.0)
            color = GREEN if v >= 0 else RED
            value_html = f'<font color="#{_color_hex(color)}">{_signed_pct(v, 2)}</font>'
            explanation = factor_explanations.get(name) or factor_explanations.get(str(name)) or ""
            explanation = _truncate_text(explanation, 155)
            rows.append([
                Paragraph(_safe_text(name), s["table"]),
                Paragraph(value_html, s["table"]),
                _factor_bar(v, max_abs=max(max_abs, 0.01), width=62, height=8),
                Paragraph(_safe_text(explanation), s["table"]),
            ])

    table = Table(rows, colWidths=[3.55 * cm, 1.55 * cm, 2.25 * cm, 8.35 * cm], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY_2),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    return table


def _make_two_column_analysis(llm_content: Mapping[str, object], s: Dict[str, ParagraphStyle], lang: str) -> Table:
    if lang == "zh":
        left_title = "Risk interpretation"
        right_title = "Outlook & monitoring"
    else:
        left_title = "Risk interpretation"
        right_title = "Outlook & monitoring"

    points = llm_content.get("monitoring_points") or []
    if not isinstance(points, list):
        points = [str(points)]
    points_text = "; ".join(f"{idx}) {_truncate_text(p, 85)}" for idx, p in enumerate(points[:3], 1))
    left = [
        Paragraph(f'<font color="#{_color_hex(MUTED)}">{_safe_text(left_title)}</font>', s["table_header"]),
        Paragraph(_safe_text(_truncate_text(llm_content.get("risk_assessment", ""), 350)), s["body"]),
    ]
    right = [
        Paragraph(f'<font color="#{_color_hex(MUTED)}">{_safe_text(right_title)}</font>', s["table_header"]),
        Paragraph(_safe_text(_truncate_text(llm_content.get("outlook", ""), 260)), s["body"]),
        Paragraph(_safe_text(points_text), s["small"]),
    ]
    t = Table([[left, right]], colWidths=[7.72 * cm, 7.72 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.45, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _build_story(
    *,
    lang: str,
    stock_code: str,
    date: str,
    news_text: str,
    prediction: str,
    confidence: float,
    var_estimate: float,
    factors: Mapping[str, float],
    ret_low: float,
    ret_high: float,
    llm_content: Mapping[str, object],
) -> List[object]:
    s = _styles(lang, news_text=news_text)
    story: List[object] = []

    if lang == "zh":
        summary_h = "1. Executive SAFIR brief"
        thesis_h = "2. Event-to-factor reasoning"
        factor_h = "3. SAFIR factor attribution"
        caveat = "SAFIR 输出由模型、新闻文本和提示词生成，仅用于研究展示与解释，不构成投资建议。"
    else:
        summary_h = "1. Executive SAFIR brief"
        thesis_h = "2. Event-to-factor reasoning"
        factor_h = "3. SAFIR factor attribution"
        caveat = "SAFIR output is generated from model scores, supplied news, and a controlled prompt. Research use only; not investment advice."

    story.append(_header_block(lang, stock_code, date, prediction, confidence, s))
    story.append(Spacer(1, 0.12 * cm))
    story.append(_make_kpi_cards(
        lang=lang,
        prediction=prediction,
        confidence=confidence,
        var_estimate=var_estimate,
        ret_low=ret_low,
        ret_high=ret_high,
        s=s,
    ))

    story.append(Spacer(1, 0.12 * cm))
    story.append(_section_title(summary_h, s))
    story.append(_make_text_box("Summary", _truncate_text(llm_content.get("executive_summary", ""), 430), s))

    story.append(Spacer(1, 0.10 * cm))
    story.append(_section_title(thesis_h, s))
    combined = f"{llm_content.get('event_digest', '')} {llm_content.get('quant_thesis', '')}"
    story.append(_make_text_box("Catalyst and quantitative thesis", _truncate_text(combined, 560), s))

    story.append(Spacer(1, 0.10 * cm))
    story.append(_make_news_box(news_text, s, lang))

    story.append(Spacer(1, 0.10 * cm))
    story.append(_section_title(factor_h, s))
    story.append(_make_factor_table(factors or {}, llm_content.get("factor_explanations", {}) or {}, s, lang))

    story.append(Spacer(1, 0.10 * cm))
    story.append(_make_two_column_analysis(llm_content, s, lang))

    story.append(Spacer(1, 0.08 * cm))
    story.append(ThinDivider(15.7 * cm))
    story.append(Paragraph(_safe_text(llm_content.get("model_caveat", caveat) or caveat), s["disclaimer"]))

    return story


# -----------------------------------------------------------------------------
# Public API - keep signature compatible with run_inference.py
# -----------------------------------------------------------------------------
def generate_sep_report(
    *,
    stock_code: str,
    date: str,
    news_text: str,
    prediction: str,
    confidence: float,
    var_estimate: float,
    factors: Mapping[str, float],
    ret_low: float,
    ret_high: float,
    output_dir: str,
    languages: Optional[Iterable[str]] = None,
    llm_model: str = "Qwen/Qwen3-4B-Thinking-2507",
) -> Dict[str, str]:
    """
    Generate one SAFIR one-page PDF per language and return {lang: file_path}.

    The function signature is compatible with the previous sep_report.py used by
    run_inference.py. It accepts the same model outputs and adds an optional
    llm_model argument for prompt-based narrative generation.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = list(languages or ["en"])
    langs = [lang for lang in langs if lang in {"en", "zh"}]
    if not langs:
        langs = ["en"]

    factors = _factor_ordered(factors)
    ret_low = float(ret_low or 0.0)
    ret_high = float(ret_high or 0.0)
    if ret_low > ret_high:
        ret_low, ret_high = ret_high, ret_low

    clean_code = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stock_code)) or "UNKNOWN"
    clean_date = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(date)) or "unknown-date"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    engine = LLMEngine(model=llm_model)
    paths: Dict[str, str] = {}

    for lang in langs:
        suffix = "ZH" if lang == "zh" else "EN"
        pdf_path = out_dir / f"safir_{clean_code}_{clean_date}_{suffix}_{ts}.pdf"

        print(f"[SAFIR] Generating prompt narrative ({lang.upper()})...")
        llm_content = engine.generate(
            stock_code=str(stock_code),
            date=str(date),
            news_text=str(news_text or ""),
            prediction=str(prediction),
            confidence=float(confidence or 0.0),
            var_estimate=float(var_estimate or 0.0),
            factors=factors,
            language=lang,
            ret_low=ret_low,
            ret_high=ret_high,
        )

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=1.38 * cm,
            leftMargin=1.38 * cm,
            topMargin=1.28 * cm,
            bottomMargin=1.12 * cm,
            title=f"SAFIR {stock_code} {date}",
            author="SAFIR",
        )

        content = _build_story(
            lang=lang,
            stock_code=str(stock_code),
            date=str(date),
            news_text=str(news_text or ""),
            prediction=str(prediction),
            confidence=float(confidence or 0.0),
            var_estimate=float(var_estimate or 0.0),
            factors=factors,
            ret_low=ret_low,
            ret_high=ret_high,
            llm_content=llm_content,
        )

        # One-page guarantee: KeepInFrame shrinks the whole story to the current
        # A4 content frame instead of spilling into a second page.
        frame_width = PAGE_W - doc.leftMargin - doc.rightMargin
        frame_height = PAGE_H - doc.topMargin - doc.bottomMargin
        story = [KeepInFrame(frame_width, frame_height, content, mode="shrink")]
        doc.build(story, onFirstPage=_draw_page_frame, onLaterPages=_draw_page_frame)
        paths[lang] = str(pdf_path)
        print(f"[SAFIR] PDF saved -> {pdf_path}")

    return paths


if __name__ == "__main__":
    demo = generate_sep_report(
        stock_code="600790.SH",
        date="2021-08-13",
        news_text=(
            "中泰化学披露三季报，公司2020年前三季度营业收入649.7亿元，同比增长0.71%；"
            "净利润15.2亿元，同比增长12.3%。公司拟收购绍兴柯桥国有土地使用权，"
            "用于建设轻纺数字物流港项目，投资金额约人民币31.72亿元。"
        ),
        prediction="Positive",
        confidence=0.83,
        var_estimate=0.02,
        factors={
            "Market Factor": 0.003,
            "Size Factor": 0.006,
            "Valuation (BP) Factor": -0.002,
            "Profitability Factor": 0.012,
            "Investment Factor": -0.004,
            "News Effect Factor": 0.015,
        },
        ret_low=0.010,
        ret_high=0.026,
        output_dir="./reports",
        languages=["en"],
    )
    print(demo)
