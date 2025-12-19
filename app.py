# -*- coding: utf-8 -*-
"""
Antigravity AIOps Agent (Clean app.py)
- UX: keep existing button names and general flow
- LLM calls: minimized (Generate Report only; cached)
- Run Diagnostics: pseudo active probe (no LLM) to enrich evidence
- Execute: no LLM; rule-based verification using verifier.py
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import graphviz
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Project modules (existing)
from data import TOPOLOGY
from logic import Alarm, simulate_cascade_failure
from verifier import verify_log_content, format_verification_report

# =========================
# Page config
# =========================
st.set_page_config(page_title="Antigravity Autonomous", page_icon="⚡", layout="wide")

# =========================
# Constants
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(APP_DIR, "configs")

DEFAULT_MODEL = os.environ.get("GOOGLE_MODEL", "gemma-3-12b-it")

# =========================
# Small utilities
# =========================
def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)

def _get_api_key() -> Optional[str]:
    # Streamlit secrets preferred; env fallback
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY")

def _read_device_config(device_id: str) -> str:
    """
    Reads config from ./configs/<device_id>.txt
    If not present, returns empty string.
    """
    if not device_id:
        return ""
    path = os.path.join(CONFIG_DIR, f"{device_id}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _graphviz_topology(topology: Dict[str, Any]) -> graphviz.Digraph:
    g = graphviz.Digraph()
    g.attr(rankdir="LR")
    # Nodes
    for node_id, node in topology.items():
        label = node_id
        try:
            ntype = getattr(node, "type", "") or ""
            layer = getattr(node, "layer", "")
            label = f"{node_id}\n{ntype} L{layer}"
        except Exception:
            pass
        g.node(node_id, label=label)
    # Edges based on node.children / parent if present
    for node_id, node in topology.items():
        children = []
        try:
            children = list(getattr(node, "children", []) or [])
        except Exception:
            children = []
        for c in children:
            if c in topology:
                g.edge(node_id, c)
    return g

def _find_target_node_id(topology: Dict[str, Any], node_type: Optional[str] = None, layer: Optional[int] = None, keyword: Optional[str] = None) -> Optional[str]:
    """
    Lightweight search. (Rule-based)
    """
    for node_id, node in topology.items():
        if node_type:
            try:
                if getattr(node, "type", None) != node_type:
                    continue
            except Exception:
                continue
        if layer is not None:
            try:
                if int(getattr(node, "layer", -1)) != int(layer):
                    continue
            except Exception:
                continue
        if keyword:
            if keyword in node_id:
                return node_id
            try:
                md = getattr(node, "metadata", {}) or {}
                for v in md.values():
                    if isinstance(v, str) and keyword in v:
                        return node_id
            except Exception:
                pass
        else:
            return node_id
    return None

# =========================
# Scenario handling (centralized, not scattered)
# =========================
@dataclass
class ScenarioContext:
    selected_scenario: str
    live_mode: bool
    target_device_id: Optional[str]
    alarms: List[Alarm]

def build_scenario_context(selected_scenario: str) -> ScenarioContext:
    alarms: List[Alarm] = []
    target_device_id: Optional[str] = None
    live_mode = False

    if "Live" in selected_scenario:
        live_mode = True
        # best-effort: WAN router
        target_device_id = _find_target_node_id(TOPOLOGY, node_type="ROUTER") or "WAN_ROUTER_01"
    elif "WAN全回線断" in selected_scenario:
        target_device_id = _find_target_node_id(TOPOLOGY, node_type="ROUTER") or "WAN_ROUTER_01"
        if target_device_id:
            alarms = simulate_cascade_failure(target_device_id, TOPOLOGY)
    elif "FW片系障害" in selected_scenario:
        target_device_id = _find_target_node_id(TOPOLOGY, node_type="FIREWALL") or "FW_01_PRIMARY"
        if target_device_id:
            alarms = [Alarm(target_device_id, "Heartbeat Loss", "WARNING")]
    elif "L2SWサイレント障害" in selected_scenario:
        target_device_id = _find_target_node_id(TOPOLOGY, node_type="SWITCH", layer=2) or "L2_SW"
        if target_device_id:
            alarms = [Alarm(target_device_id, "Silent Drop", "CRITICAL")]
    else:
        # default: no alarms
        target_device_id = _find_target_node_id(TOPOLOGY) or None

    return ScenarioContext(
        selected_scenario=selected_scenario,
        live_mode=live_mode,
        target_device_id=target_device_id,
        alarms=alarms,
    )

# =========================
# Active probe (pseudo) - NO LLM
# =========================
@dataclass
class ProbeResult:
    device_id: str
    created_at: float
    log_text: str

def synthesize_probe_log(ctx: ScenarioContext, device_id: str, device_config: str) -> ProbeResult:
    """
    Create a pseudo log that increases evidence for AI report.
    No LLM. Deterministic based on scenario & alarms.
    """
    alarm_lines = []
    for a in (ctx.alarms or []):
        try:
            alarm_lines.append(f"- {a.device_id} / {a.name} / {a.severity}")
        except Exception:
            pass

    # Simple heuristics: for WAN all links down, show interface down / bgp down
    scenario = ctx.selected_scenario
    lines: List[str] = []
    lines.append(f"[PROBE] scenario={scenario}")
    lines.append(f"[PROBE] target_device={device_id}")
    lines.append(f"[PROBE] ts={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("[ALARMS]")
    lines.extend(alarm_lines if alarm_lines else ["(no alarms)"])
    lines.append("")
    lines.append("[CONFIG_SNIPPET]")
    if device_config:
        snippet = device_config.strip().splitlines()
        lines.extend(snippet[:80])
        if len(snippet) > 80:
            lines.append("... (truncated)")
    else:
        lines.append("(no config)")
    lines.append("")
    lines.append("[OBSERVATION]")
    if "WAN全回線断" in scenario or "[WAN]" in scenario:
        lines.append("Interface GigabitEthernet0/0 is down, line protocol is down")
        lines.append("BGP neighbor 203.0.113.2 state = Idle")
        lines.append("Ping 203.0.113.2: 0/5 success")
    elif "FW片系障害" in scenario:
        lines.append("Cluster heartbeat lost on node primary; redundancy degraded")
        lines.append("Security zones: trust/untrust reachable; session sync warning")
    elif "L2SW" in scenario:
        lines.append("High temperature warning; fan speed abnormal")
        lines.append("MAC flapping detected on uplink port")
    else:
        lines.append("No abnormal observation detected in pseudo probe.")

    return ProbeResult(device_id=device_id, created_at=time.time(), log_text="\n".join(lines))

# =========================
# LLM call (ONLY for Generate Report) with retry
# =========================
def _is_retryable_google_error(e: Exception) -> bool:
    # 503 / 429 are retryable
    if isinstance(e, google_exceptions.ServiceUnavailable):
        return True
    if isinstance(e, google_exceptions.ResourceExhausted):
        return True
    # Some errors may come as generic GoogleAPICallError with status code
    status = getattr(e, "code", None)
    try:
        if callable(status):
            sc = status()
            return sc in (429, 503)
    except Exception:
        pass
    return False

def call_llm_once(api_key: str, prompt: str, model_name: str = DEFAULT_MODEL, max_attempts: int = 3) -> str:
    """
    Single logical call with retry/backoff. Kept in one place to avoid IF sprawl.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0.2},
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text:
                # fall back to candidates
                text = str(resp)
            return text
        except Exception as e:
            last_err = e
            if not _is_retryable_google_error(e) or attempt >= max_attempts:
                raise
            # exponential backoff + small jitter
            time.sleep(min(6.0, 0.8 * (2 ** (attempt - 1)) + (attempt * 0.1)))
    # unreachable
    raise last_err or RuntimeError("LLM call failed")

# =========================
# AI Analyst Report builder (one shot JSON)
# =========================
@dataclass
class AnalystBundle:
    analyst_report: str
    fix_plan: str
    recovery_commands: str
    verification_commands: str
    expectations: str  # keep as text to avoid brittle parsing

def _bundle_from_json(obj: Dict[str, Any]) -> AnalystBundle:
    return AnalystBundle(
        analyst_report=str(obj.get("analyst_report", "")).strip(),
        fix_plan=str(obj.get("fix_plan", "")).strip(),
        recovery_commands=str(obj.get("recovery_commands", "")).strip(),
        verification_commands=str(obj.get("verification_commands", "")).strip(),
        expectations=str(obj.get("expectations", "")).strip(),
    )

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object in text.
    """
    if not text:
        return None
    # If code-fenced
    fence = "```"
    if fence in text:
        parts = text.split(fence)
        # pick the largest chunk that looks like json
        candidates = []
        for p in parts:
            if "{" in p and "}" in p:
                candidates.append(p)
        candidates.sort(key=len, reverse=True)
        for c in candidates:
            s = c.strip()
            # remove leading language tag
            s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE)
            try:
                return json.loads(s)
            except Exception:
                continue
    # Otherwise attempt greedy brace match
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        s = text[start:end+1]
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def build_report_prompt(
    ctx: ScenarioContext,
    topology_context: Dict[str, Any],
    target_device_id: str,
    ci_context: Dict[str, Any],
    device_config: str,
    probe_log: str,
) -> str:
    """
    Operator-facing prompt:
    - です/ます調
    - no customer-facing apologies/PR lines
    - rely on CI/topology/config/probe facts
    - output strict JSON (single object)
    """
    facts = {
        "scenario": ctx.selected_scenario,
        "target_device": target_device_id,
        "ci": ci_context,
        "topology": topology_context,
        "config_excerpt": (device_config[:2000] if device_config else ""),
        "probe_log": (probe_log[:2500] if probe_log else ""),
        "alarms": [
            {"device_id": getattr(a, "device_id", ""), "name": getattr(a, "name", ""), "severity": getattr(a, "severity", "")}
            for a in (ctx.alarms or [])
        ]
    }

    return f"""
あなたはネットワーク運用者向けのAI分析官です。以下の入力事実のみから判断し、運用者向けの文言で出力してください。

重要な文体ルール:
- 必ず「です/ます調」で統一してください。
- 「現在、原因究明と復旧作業を最優先で進めております」「進捗状況は随時ご報告いたします」「検討を加速させます」などの顧客向け・対外向けの定型句は禁止です。
- 運用者がそのまま作業に使える、具体的・技術的な内容にしてください。
- 推測は「推定」と明示し、観測事実と分離してください。

出力形式は、必ず JSON 1オブジェクトのみです（前後に説明文を付けないでください）。
JSON のキーは次の5つを必ず含めてください:
- "analyst_report": 運用者向け 詳細レポート（章立て: 1.障害概要 2.影響 3.詳細情報 4.対応と特定根拠）
- "fix_plan": 運用者向け 修復プラン（手順・注意点を箇条書き）
- "recovery_commands": 復旧コマンド（そのまま実行できる形式。複数行可）
- "verification_commands": 正常性確認コマンド（そのまま実行できる形式。複数行可）
- "expectations": 各コマンドの期待結果（合否判定キーを含む。例: EXPECT_CONTAINS, EXPECT_NOT_CONTAINS, EXPECT_REGEX など）

入力事実:
{_safe_json_dumps(facts)}
""".strip()

# =========================
# Execute: rule-based verification (no LLM)
# =========================
def synthesize_verification_log(ctx: ScenarioContext) -> str:
    """
    Provide pseudo output for verification commands.
    Rule-based; keep simple but consistent with scenario.
    """
    scenario = ctx.selected_scenario
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[VERIFY] ts={ts}", f"[VERIFY] scenario={scenario}", ""]
    if "WAN全回線断" in scenario or "[WAN]" in scenario:
        lines += [
            "PING 203.0.113.2 (203.0.113.2): 56 data bytes",
            "--- 203.0.113.2 ping statistics ---",
            "5 packets transmitted, 0 packets received, 100% packet loss",
            "",
            "GigabitEthernet0/0 is down, line protocol is down",
        ]
    elif "FW片系障害" in scenario:
        lines += [
            "Redundancy state: DEGRADED",
            "Heartbeat: LOST (primary)",
            "Session sync: WARNING",
        ]
    elif "L2SW" in scenario:
        lines += [
            "Temperature: HIGH",
            "Fan status: FAIL",
            "Interface status: WARNING",
        ]
    else:
        lines += [
            "Ping: OK",
            "Interface: OK",
        ]
    return "\n".join(lines)

# =========================
# Session State (single place)
# =========================
def _ss_init() -> None:
    st.session_state.setdefault("probe_logs", {})              # device_id -> ProbeResult dict
    st.session_state.setdefault("analyst_cache", {})           # cache_key -> AnalystBundle dict
    st.session_state.setdefault("last_bundle", None)           # AnalystBundle dict
    st.session_state.setdefault("last_cache_key", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("llm_busy", False)

_ss_init()

# =========================
# Sidebar UI (kept similar)
# =========================
api_key = _get_api_key()

with st.sidebar:
    st.header("⚡ Scenario Controller")
    SCENARIO_MAP = {
        "基本・広域障害": ["正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害"],
        "WAN Router": ["4. [WAN] 電源障害：片系", "5. [WAN] 電源障害：両系", "6. [WAN] BGPルートフラッピング", "7. [WAN] FAN故障", "8. [WAN] メモリリーク"],
        "Firewall (Juniper)": ["9. [FW] 電源障害：片系", "10. [FW] 電源障害：両系", "11. [FW] FAN故障", "12. [FW] メモリリーク"],
        "L2 Switch": ["13. [L2SW] 電源障害：片系", "14. [L2SW] 電源障害：両系", "15. [L2SW] FAN故障", "16. [L2SW] メモリリーク"],
        "複合・その他": ["17. [WAN] 複合障害：電源＆FAN", "18. [Complex] 同時多発：FW & AP", "99. [Live] Cisco実機診断"],
    }
    selected_category = st.selectbox("対象カテゴリ:", list(SCENARIO_MAP.keys()))
    selected_scenario = st.radio("発生シナリオ:", SCENARIO_MAP[selected_category])
    st.markdown("---")
    if api_key:
        st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key:
            api_key = user_key

# =========================
# Build scenario context & common contexts
# =========================
ctx = build_scenario_context(selected_scenario)
target_device_id = ctx.target_device_id or ""
target_node = TOPOLOGY.get(target_device_id) if target_device_id else None

# Minimal CI context from node metadata
ci_context: Dict[str, Any] = {}
if target_node:
    try:
        ci_context = {
            "id": getattr(target_node, "id", target_device_id),
            "type": getattr(target_node, "type", ""),
            "layer": getattr(target_node, "layer", ""),
            "metadata": getattr(target_node, "metadata", {}) or {},
            "parent": getattr(target_node, "parent", None),
            "children": getattr(target_node, "children", []) or [],
        }
    except Exception:
        ci_context = {"id": target_device_id}

topology_context = {
    "nodes": {nid: {"type": getattr(n, "type", ""), "layer": getattr(n, "layer", ""), "metadata": getattr(n, "metadata", {}) or {}, "parent": getattr(n, "parent", None), "children": getattr(n, "children", []) or []} for nid, n in TOPOLOGY.items()},
}

device_config = _read_device_config(target_device_id)

# =========================
# Main UI
# =========================
st.title("⚡ Antigravity Autonomous AIOps Agent")

col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.subheader("Network Topology")
    try:
        st.graphviz_chart(_graphviz_topology(TOPOLOGY), use_container_width=True)
    except Exception as e:
        st.warning(f"Topology rendering failed: {e}")

    st.subheader("Active Alarms")
    if ctx.alarms:
        alarm_rows = []
        for a in ctx.alarms:
            alarm_rows.append({
                "device_id": getattr(a, "device_id", ""),
                "alarm": getattr(a, "name", ""),
                "severity": getattr(a, "severity", ""),
            })
        st.dataframe(alarm_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No active alarms in this scenario.")

with col_right:
    st.subheader("Operations Console")
    st.write(f"対象機器: **{target_device_id or 'N/A'}**")
    st.write(f"シナリオ: **{selected_scenario}**")

    # ---- Actions (UX preserved)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        run_diag = st.button("診断実行(Run Diagnostics)", use_container_width=True)
    with col_b:
        gen_report = st.button("詳細レポートを作成(Generate Report)", use_container_width=True)
    with col_c:
        gen_fix = st.button("修正プランを作成(Generate Fix)", use_container_width=True)
    with col_d:
        exec_btn = st.button("修復実行(Execute)", use_container_width=True)

    # --- Optional: "復旧コマンド" button beside Execute area (kept)
    col_exec, col_exec_cmd = st.columns([1, 1])
    with col_exec_cmd:
        show_recovery = st.button("復旧コマンド", use_container_width=True)

    # ---- Fixed display areas (always visible, grey if missing)
    bundle_dict = st.session_state.get("last_bundle") or {}
    bundle = _bundle_from_json(bundle_dict) if bundle_dict else AnalystBundle("", "", "", "", "")

    st.markdown("### 復旧コマンド (Recovery Config)")
    st.text_area("recovery_commands", value=bundle.recovery_commands or "", height=140, disabled=(not bool(bundle.recovery_commands)))

    st.markdown("### 正常性確認コマンド (Verification Commands)")
    st.text_area("verification_commands", value=bundle.verification_commands or "", height=140, disabled=(not bool(bundle.verification_commands)))

    st.markdown("### 期待結果 (Expectations / 判定キー)")
    st.text_area("expectations", value=bundle.expectations or "", height=180, disabled=(not bool(bundle.expectations)))

    # =========================
    # Action handlers (centralized)
    # =========================
    def handle_run_diagnostics() -> None:
        if not target_device_id:
            st.error("対象機器が特定できません。")
            return
        pr = synthesize_probe_log(ctx, target_device_id, device_config)
        st.session_state.probe_logs[target_device_id] = asdict(pr)
        st.success("能動プローブ（疑似）を実行し、材料を追加しました。")

    def handle_generate_report() -> None:
        if not api_key:
            st.error("API Key がありません。サイドバーで設定してください。")
            return
        if not target_device_id:
            st.error("対象機器が特定できません。")
            return

        # Gather probe log if available
        pr_dict = st.session_state.probe_logs.get(target_device_id) or {}
        probe_log = pr_dict.get("log_text", "")

        verification_context = {
            "probe_present": bool(probe_log),
            "alarm_count": len(ctx.alarms or []),
            "device_has_config": bool(device_config),
        }

        cache_key = "|".join([
            selected_scenario,
            target_device_id,
            _hash_text(_safe_json_dumps(topology_context)),
            _hash_text(device_config),
            _hash_text(_safe_json_dumps(verification_context)),
            _hash_text(probe_log),
        ])

        # Cache hit
        if cache_key in st.session_state.analyst_cache:
            st.session_state.last_bundle = st.session_state.analyst_cache[cache_key]
            st.session_state.last_cache_key = cache_key
            st.info("キャッシュ済みのAI Analyst Reportを再利用しました。")
            return

        prompt = build_report_prompt(
            ctx=ctx,
            topology_context=topology_context,
            target_device_id=target_device_id,
            ci_context=ci_context,
            device_config=device_config,
            probe_log=probe_log,
        )

        st.session_state.llm_busy = True
        try:
            raw = call_llm_once(api_key=api_key, prompt=prompt, model_name=DEFAULT_MODEL, max_attempts=3)
            obj = _extract_json(raw)
            if not obj:
                # fallback: store as plain report
                obj = {
                    "analyst_report": raw.strip(),
                    "fix_plan": "",
                    "recovery_commands": "",
                    "verification_commands": "",
                    "expectations": "",
                }
            b = _bundle_from_json(obj)
            st.session_state.last_bundle = asdict(b)
            st.session_state.analyst_cache[cache_key] = asdict(b)
            st.session_state.last_cache_key = cache_key
            st.success("AI Analyst Report を生成しました。")
        except Exception as e:
            st.session_state.last_error = f"{type(e).__name__}: {e}"
            st.error(f"AI生成に失敗しました: {st.session_state.last_error}")
        finally:
            st.session_state.llm_busy = False

    def handle_generate_fix() -> None:
        # No LLM: reuse last bundle
        bdict = st.session_state.get("last_bundle") or {}
        if not bdict:
            st.warning("先に「詳細レポートを作成(Generate Report)」を実行してください。")
            return
        st.info("AI Analyst Report の内容から修復プランを表示します（追加のAI呼び出しは行いません）。")

    def handle_execute() -> None:
        bdict = st.session_state.get("last_bundle") or {}
        if not bdict:
            st.warning("先に「詳細レポートを作成(Generate Report)」を実行してください。")
            return

        st.write("復旧コマンド適用（疑似）を開始します。")
        time.sleep(0.6)
        st.write("正常性確認（疑似）を実行します。")
        time.sleep(0.6)

        # Produce pseudo verification log and run verifier (rule-based)
        v_log = synthesize_verification_log(ctx)
        v = verify_log_content(v_log)
        rep = format_verification_report(v)

        st.markdown("### 正常性確認結果（ルールベース）")
        st.code(rep)

    # Trigger handlers (minimal conditions)
    if run_diag:
        handle_run_diagnostics()

    if gen_report:
        handle_generate_report()

    if gen_fix:
        handle_generate_fix()

    if exec_btn:
        handle_execute()

    if show_recovery:
        # show in expander without extra logic
        if bundle.recovery_commands:
            with st.expander("復旧コマンド（コピー用）", expanded=True):
                st.code(bundle.recovery_commands)
        else:
            st.info("復旧コマンドは未生成です。先にレポートを作成してください。")

    # =========================
    # Output sections
    # =========================
    st.markdown("---")
    st.markdown("## AI Analyst Report")
    if bundle.analyst_report:
        st.write(bundle.analyst_report)
    else:
        st.info("まだ生成されていません。")

    st.markdown("## 修復プラン (Fix Plan)")
    if bundle.fix_plan:
        st.write(bundle.fix_plan)
    else:
        st.info("まだ生成されていません。")

    if st.session_state.get("last_error"):
        st.warning(f"Last error: {st.session_state.last_error}")
