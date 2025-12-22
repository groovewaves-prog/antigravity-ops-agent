# -*- coding: utf-8 -*-
"""
AIOps Agent - Main Application (Improved v2)
=============================================
改善点:
1. グローバルレートリミッター統合
2. チャット機能のレート制限対応
3. infer_root_cause (バッチ処理対応) の利用
4. エラーハンドリング強化
"""

import streamlit as st
import graphviz
import os
import time
import google.generativeai as genai
import json
import re
import pandas as pd
from google.api_core import exceptions as google_exceptions

# モジュール群のインポート
from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import (
    run_diagnostic_simulation,
    generate_remediation_commands,
    generate_analyst_report,
    generate_analyst_report_streaming,
    generate_remediation_commands_streaming,
    compute_cache_hash,
    predict_initial_symptoms,
    generate_fake_log_by_ai,
    run_remediation_parallel_v2,
    RemediationEnvironment,
    RemediationResult
)
from verifier import verify_log_content, format_verification_report
from inference_engine import LogicalRCA
from rate_limiter import GlobalRateLimiter, RateLimitConfig

# --- ページ設定 ---
st.set_page_config(page_title="Antigravity Autonomous", page_icon="⚡", layout="wide")

# =====================================================
# レートリミッター初期化
# =====================================================
@st.cache_resource
def get_rate_limiter():
    """レートリミッターのシングルトンインスタンスを取得"""
    return GlobalRateLimiter(RateLimitConfig(
        rpm=30,
        rpd=14400,
        safety_margin=0.8
    ))

rate_limiter = get_rate_limiter()

# =====================================================
# ユーティリティ関数
# =====================================================
def find_target_node_id(topology, node_type=None, layer=None, keyword=None):
    """トポロジーから条件に合うノードIDを検索"""
    for node_id, node in topology.items():
        if node_type and node.type != node_type:
            continue
        if layer and node.layer != layer:
            continue
        if keyword:
            hit = False
            if keyword in node_id:
                hit = True
            for v in node.metadata.values():
                if isinstance(v, str) and keyword in v:
                    hit = True
            if not hit:
                continue
        return node_id
    return None


def load_config_by_id(device_id):
    """configsフォルダから設定ファイルを読み込む"""
    possible_paths = [f"configs/{device_id}.txt", f"{device_id}.txt"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return "Config file not found."


def generate_content_with_retry(model, prompt, stream=True, retries=3):
    """503エラー対策のリトライ付き生成関数（レートリミッター統合）"""
    for i in range(retries):
        try:
            # レート制限待機
            if not rate_limiter.wait_for_slot(timeout=60):
                raise RuntimeError("Rate limit timeout")
            rate_limiter.record_request()
            return model.generate_content(prompt, stream=stream)
        except google_exceptions.ServiceUnavailable:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                if i == retries - 1:
                    raise
                time.sleep(5 * (i + 1))
            else:
                raise
    return None


def _pick_first(mapping: dict, keys: list, default: str = "") -> str:
    """Return the first non-empty value for the given keys from mapping"""
    for k in keys:
        try:
            v = mapping.get(k, None)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            s = str(v)
            if s:
                return s
        elif isinstance(v, str):
            if v.strip():
                return v.strip()
        else:
            try:
                s = json.dumps(v, ensure_ascii=False)
                if s and s != "null":
                    return s
            except Exception:
                continue
    return default


def _build_ci_context_for_chat(target_node_id: str) -> dict:
    """チャット用のCIコンテキストを構築"""
    node = TOPOLOGY.get(target_node_id) if target_node_id else None
    md = (getattr(node, "metadata", None) or {}) if node else {}

    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer", "maker", "brand"], default=""),
        "os": _pick_first(md, ["os", "platform", "os_name", "software", "sw"], default=""),
        "model": _pick_first(md, ["model", "hw_model", "product", "sku"], default=""),
        "role": _pick_first(md, ["role", "type", "device_role"], default=""),
        "layer": _pick_first(md, ["layer", "level", "network_layer"], default=""),
        "site": _pick_first(md, ["site", "dc", "datacenter", "location"], default=""),
        "tenant": _pick_first(md, ["tenant", "customer", "org", "company"], default=""),
        "mgmt_ip": _pick_first(md, ["mgmt_ip", "management_ip", "management", "oob_ip"], default=""),
    }

    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf:
            ci["config_excerpt"] = conf[:1500]
    except Exception:
        pass

    return ci


def _safe_chunk_text(chunk) -> str:
    """google.generativeai の stream chunk から安全にテキストを取り出す"""
    try:
        t = getattr(chunk, "text", "")
        if t:
            return t
    except Exception:
        pass

    try:
        cands = getattr(chunk, "candidates", None) or []
        if not cands:
            return ""
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) or []
        out = []
        for p in parts:
            tx = getattr(p, "text", "")
            if tx:
                out.append(tx)
        return "".join(out)
    except Exception:
        return ""


def run_diagnostic_simulation_no_llm(selected_scenario, target_node_obj):
    """LLMを呼ばない疑似診断（503/コスト対策）"""
    device_id = getattr(target_node_obj, "id", "UNKNOWN") if target_node_obj else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[PROBE] ts={ts}",
        f"[PROBE] scenario={selected_scenario}",
        f"[PROBE] target_device={device_id}",
        "",
    ]

    recovered_devices = st.session_state.get("recovered_devices") or {}
    recovered_map = st.session_state.get("recovered_scenario_map") or {}

    if recovered_devices.get(device_id) and recovered_map.get(device_id) == selected_scenario:
        if "FW" in selected_scenario:
            lines += [
                "show chassis cluster status",
                "Redundancy group 0: healthy",
                "control link: up",
                "fabric link: up",
            ]
        elif "WAN" in selected_scenario or "WAN全回線断" in selected_scenario:
            lines += [
                "show ip interface brief",
                "GigabitEthernet0/0 up up",
                "show ip bgp summary",
                "Neighbor 203.0.113.2 Established",
                "ping 203.0.113.2 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]
        elif "L2SW" in selected_scenario:
            lines += [
                "show environment",
                "Fan: OK",
                "Temperature: OK",
                "show interface status",
                "Uplink: up",
            ]
        else:
            lines += [
                "show system alarms",
                "No active alarms",
                "ping 8.8.8.8 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]

        return {
            "status": "SUCCESS",
            "sanitized_log": "\n".join(lines),
            "verification_log": "N/A",
            "device_id": device_id,
        }

    if "WAN全回線断" in selected_scenario or "[WAN]" in selected_scenario:
        lines += [
            "show ip interface brief",
            "GigabitEthernet0/0 down down",
            "show ip bgp summary",
            "Neighbor 203.0.113.2 Idle",
            "ping 203.0.113.2 repeat 5",
            "Success rate is 0 percent (0/5)",
        ]
    elif "FW片系障害" in selected_scenario or "[FW]" in selected_scenario:
        lines += [
            "show chassis cluster status",
            "Redundancy group 0: degraded",
            "control link: down",
            "fabric link: up",
        ]
    elif "L2SW" in selected_scenario:
        lines += [
            "show environment",
            "Fan: FAIL",
            "Temperature: HIGH",
            "show interface status",
            "Uplink: flapping",
        ]
    else:
        lines += [
            "show system alarms",
            "No active alarms",
        ]

    return {
        "status": "SUCCESS",
        "sanitized_log": "\n".join(lines),
        "verification_log": "N/A",
        "device_id": device_id,
    }


def render_topology(alarms, root_cause_candidates):
    """トポロジー図の描画"""
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')

    alarm_map = {a.device_id: a for a in alarms}
    alarmed_ids = set(alarm_map.keys())
    root_cause_ids = {c['id'] for c in root_cause_candidates if c['prob'] > 0.6}
    node_status_map = {c['id']: c['type'] for c in root_cause_candidates}

    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9"
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"

        red_type = node.metadata.get("redundancy_type")
        if red_type:
            label += f"\n[{red_type} Redundancy]"
        vendor = node.metadata.get("vendor")
        if vendor:
            label += f"\n[{vendor}]"

        status_type = node_status_map.get(node_id, "Normal")

        if "Hardware/Physical" in status_type or "Critical" in status_type or "Silent" in status_type:
            color = "#ffcdd2"
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
        elif "Network/Unreachable" in status_type or "Network/Secondary" in status_type:
            color = "#cfd8dc"
            fontcolor = "#546e7a"
            label += "\n[Unreachable]"
        elif node_id in alarmed_ids:
            color = "#fff9c4"

        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)

    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent_node = TOPOLOGY.get(node.parent_id)
            if parent_node and parent_node.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values()
                           if n.redundancy_group == parent_node.redundancy_group and n.id != parent_node.id]
                for partner_id in partners:
                    graph.edge(partner_id, node_id)
    return graph


# =====================================================
# UI構築
# =====================================================
st.title("⚡ Antigravity Autonomous Agent")

# レート制限状況の表示
with st.sidebar:
    stats = rate_limiter.get_stats()
    st.caption(f"📊 API: {stats['requests_last_minute']}/{stats['rpm_limit']} RPM | Cache: {stats['cache_size']}")

api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

# --- サイドバー ---
with st.sidebar:
    st.header("⚡ Scenario Controller")
    SCENARIO_MAP = {
        "基本・広域障害": ["正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害"],
        "WAN Router": ["4. [WAN] 電源障害：片系", "5. [WAN] 電源障害：両系", "6. [WAN] BGPルートフラッピング", "7. [WAN] FAN故障", "8. [WAN] メモリリーク"],
        "Firewall (Juniper)": ["9. [FW] 電源障害：片系", "10. [FW] 電源障害：両系", "11. [FW] FAN故障", "12. [FW] メモリリーク"],
        "L2 Switch": ["13. [L2SW] 電源障害：片系", "14. [L2SW] 電源障害：両系", "15. [L2SW] FAN故障", "16. [L2SW] メモリリーク"],
        "複合・その他": ["17. [WAN] 複合障害：電源＆FAN", "18. [Complex] 同時多発：FW & AP", "99. [Live] Cisco実機診断"]
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

# --- セッション管理 ---
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"

for key in ["live_result", "messages", "chat_session", "trigger_analysis", "verification_result", 
            "generated_report", "verification_log", "last_report_cand_id", "logic_engine", 
            "recovered_devices", "recovered_scenario_map", "balloons_shown"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key in ["trigger_analysis", "balloons_shown"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

if "recovered_devices" not in st.session_state:
    st.session_state.recovered_devices = {}
if "recovered_scenario_map" not in st.session_state:
    st.session_state.recovered_scenario_map = {}
if "global_cache" not in st.session_state:
    st.session_state.global_cache = {}

GLOBAL_CACHE = st.session_state.global_cache

# エンジン初期化
if not st.session_state.logic_engine:
    st.session_state.logic_engine = LogicalRCA(TOPOLOGY)

# シナリオ切り替え時のリセット
if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.recovered_devices = {}
    st.session_state.recovered_scenario_map = {}
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.live_result = None
    st.session_state.trigger_analysis = False
    st.session_state.verification_result = None
    st.session_state.generated_report = None
    st.session_state.verification_log = None
    st.session_state.last_report_cand_id = None
    st.session_state.balloons_shown = False
    if "remediation_plan" in st.session_state:
        del st.session_state.remediation_plan
    st.rerun()

# =====================================================
# メインロジック
# =====================================================
alarms = []
target_device_id = None
root_severity = "CRITICAL"
is_live_mode = False

# 1. アラーム生成ロジック
if "Live" in selected_scenario:
    is_live_mode = True
elif "WAN全回線断" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    if target_device_id:
        alarms = simulate_cascade_failure(target_device_id, TOPOLOGY)
elif "FW片系障害" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    if target_device_id:
        alarms = [Alarm(target_device_id, "Heartbeat Loss", "WARNING")]
        root_severity = "WARNING"
elif "L2SWサイレント障害" in selected_scenario:
    target_device_id = "L2_SW_01"
    if target_device_id not in TOPOLOGY:
        target_device_id = find_target_node_id(TOPOLOGY, keyword="L2_SW")
    if target_device_id and target_device_id in TOPOLOGY:
        child_nodes = [nid for nid, n in TOPOLOGY.items() if n.parent_id == target_device_id]
        alarms = [Alarm(child, "Connection Lost", "CRITICAL") for child in child_nodes]
    else:
        st.error("Error: L2 Switch definition not found")
elif "複合障害" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    if target_device_id:
        alarms = [
            Alarm(target_device_id, "Power Supply 1 Failed", "CRITICAL"),
            Alarm(target_device_id, "Fan Fail", "WARNING")
        ]
elif "同時多発" in selected_scenario:
    fw_node = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    ap_node = find_target_node_id(TOPOLOGY, node_type="ACCESS_POINT")
    alarms = []
    if fw_node:
        alarms.append(Alarm(fw_node, "Heartbeat Loss", "WARNING"))
    if ap_node:
        alarms.append(Alarm(ap_node, "Connection Lost", "CRITICAL"))
    target_device_id = fw_node
else:
    if "[WAN]" in selected_scenario:
        target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    elif "[FW]" in selected_scenario:
        target_device_id = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    elif "[L2SW]" in selected_scenario:
        target_device_id = find_target_node_id(TOPOLOGY, node_type="SWITCH", layer=4)

    if target_device_id:
        if "電源障害：片系" in selected_scenario:
            alarms = [Alarm(target_device_id, "Power Supply 1 Failed", "WARNING")]
            root_severity = "WARNING"
        elif "電源障害：両系" in selected_scenario:
            if "FW" in target_device_id:
                alarms = [Alarm(target_device_id, "Power Supply: Dual Loss (Device Down)", "CRITICAL")]
            else:
                alarms = simulate_cascade_failure(target_device_id, TOPOLOGY, "Power Supply: Dual Loss (Device Down)")
        elif "BGP" in selected_scenario:
            alarms = [Alarm(target_device_id, "BGP Flapping", "WARNING")]
            root_severity = "WARNING"
        elif "FAN" in selected_scenario:
            alarms = [Alarm(target_device_id, "Fan Fail", "WARNING")]
            root_severity = "WARNING"
        elif "メモリ" in selected_scenario:
            alarms = [Alarm(target_device_id, "Memory High", "WARNING")]
            root_severity = "WARNING"

# 2. ★改善: バッチ処理対応の推論エンジン
# アラームをmsg_map形式に変換
msg_map = {}
for alarm in alarms:
    if alarm.device_id not in msg_map:
        msg_map[alarm.device_id] = []
    msg_map[alarm.device_id].append(alarm.message)

# infer_root_cause (バッチ処理対応) を使用
analysis_results = st.session_state.logic_engine.infer_root_cause(msg_map)

# 3. コックピット表示
selected_incident_candidate = None

st.markdown("### 🛡️ AIOps インシデント・コックピット")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📉 ノイズ削減率", "98.5%", "高効率稼働中")
with col2:
    st.metric("📨 処理アラーム数", f"{len(alarms) * 15 if alarms else 0}件", "抑制済")
with col3:
    st.metric("🚨 要対応インシデント", f"{len([c for c in analysis_results if c['prob'] > 0.6])}件", "対処が必要")
st.markdown("---")

df_data = []
for rank, cand in enumerate(analysis_results, 1):
    status = "⚪ 監視中"
    action = "👁️ 静観"

    if cand['prob'] > 0.8:
        status = "🔴 危険 (根本原因)"
        action = "🚀 自動修復が可能"
    elif cand['prob'] > 0.6:
        status = "🟡 警告 (被疑箇所)"
        action = "🔍 詳細調査を推奨"

    if "Network/Unreachable" in cand['type'] or "Network/Secondary" in cand['type']:
        status = "⚫ 応答なし (上位障害)"
        action = "⛔ 対応不要 (上位復旧待ち)"

    candidate_text = f"デバイス: {cand['id']} / 原因: {cand['label']}"
    if cand.get('verification_log'):
        candidate_text += " [🔍 Active Probe: 応答なし]"

    df_data.append({
        "順位": rank,
        "ステータス": status,
        "根本原因候補": candidate_text,
        "リスクスコア": cand['prob'],
        "推奨アクション": action,
        "ID": cand['id'],
        "Type": cand['type']
    })

df = pd.DataFrame(df_data)
st.info("💡 ヒント: インシデントの行をクリックすると、右側に詳細分析と復旧プランが表示されます。")

event = st.dataframe(
    df,
    column_order=["順位", "ステータス", "根本原因候補", "リスクスコア", "推奨アクション"],
    column_config={
        "リスクスコア": st.column_config.ProgressColumn("リスクスコア (0-1.0)", format="%.2f", min_value=0, max_value=1),
    },
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun"
)

if len(event.selection.rows) > 0:
    idx = event.selection.rows[0]
    sel_row = df.iloc[idx]
    for res in analysis_results:
        if res['id'] == sel_row['ID'] and res['type'] == sel_row['Type']:
            selected_incident_candidate = res
            break
else:
    selected_incident_candidate = analysis_results[0] if analysis_results else None

# 4. 画面分割
col_map, col_chat = st.columns([1.2, 1])

# === 左カラム: トポロジーと診断 ===
with col_map:
    st.subheader("🌐 Network Topology")

    current_root_node = None
    current_severity = "WARNING"

    if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
        current_root_node = TOPOLOGY.get(selected_incident_candidate["id"])
        if "Hardware/Physical" in selected_incident_candidate["type"] or "Critical" in selected_incident_candidate["type"] or "Silent" in selected_incident_candidate["type"]:
            current_severity = "CRITICAL"
        else:
            current_severity = "WARNING"
    elif target_device_id:
        current_root_node = TOPOLOGY.get(target_device_id)
        current_severity = root_severity

    st.graphviz_chart(render_topology(alarms, analysis_results), use_container_width=True)

    st.markdown("---")
    st.subheader("🛠️ Auto-Diagnostics")

    if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
        if not api_key:
            st.error("API Key Required")
        else:
            with st.status("Agent Operating...", expanded=True) as status:
                st.write("🔌 Connecting to device...")
                target_node_obj = TOPOLOGY.get(target_device_id) if target_device_id else None
                is_live = bool(st.session_state.get('api_connected')) and ('[Live]' in selected_scenario or 'Live' in selected_scenario)

                res = run_diagnostic_simulation(selected_scenario, target_node_obj, api_key) if is_live else run_diagnostic_simulation_no_llm(selected_scenario, target_node_obj)
                st.session_state.live_result = res

                if res["status"] == "SUCCESS":
                    st.write("✅ Log Acquired & Sanitized.")
                    status.update(label="Diagnostics Complete!", state="complete", expanded=False)
                    log_content = res.get('sanitized_log', "")
                    verification = verify_log_content(log_content)
                    st.session_state.verification_result = verification
                    st.session_state.trigger_analysis = True
                elif res["status"] == "SKIPPED":
                    status.update(label="No Action Required", state="complete")
                else:
                    st.write("❌ Connection Failed.")
                    status.update(label="Diagnostics Failed", state="error")
            st.rerun()

    if st.session_state.live_result:
        res = st.session_state.live_result
        if res["status"] == "SUCCESS":
            st.markdown("#### 📄 Diagnostic Results")
            with st.container(border=True):
                if selected_incident_candidate and selected_incident_candidate.get("verification_log"):
                    st.caption("🤖 Active Probe / Verification Log")
                else:
                    st.caption("📃 Collected Log Data")
                st.code(res["sanitized_log"][:3000], language="text")

            if st.session_state.verification_result:
                st.markdown("#### 🔎 Ground Truth Verification")
                report = format_verification_report(st.session_state.verification_result)
                st.markdown(report)

# === 右カラム: 詳細分析と復旧 ===
with col_chat:
    if selected_incident_candidate:
        st.subheader(f"📊 詳細分析: {selected_incident_candidate['id']}")

        st.markdown(f"""
        **デバイス**: `{selected_incident_candidate['id']}`  
        **原因**: `{selected_incident_candidate['label']}`  
        **リスクスコア**: `{selected_incident_candidate['prob']:.2f}`  
        **分類**: `{selected_incident_candidate['type']}`  
        **理由**: {selected_incident_candidate['reason']}
        """)

        if selected_incident_candidate.get('analyst_report'):
            with st.expander("🔍 AI Analyst Report", expanded=True):
                st.code(selected_incident_candidate['analyst_report'], language="text")

        if selected_incident_candidate.get('auto_investigation'):
            with st.expander("📋 推奨調査項目", expanded=False):
                for item in selected_incident_candidate['auto_investigation']:
                    st.markdown(f"- {item}")

    # チャット機能（レートリミッター統合）
    with st.expander("💬 Chat with AI Agent", expanded=False):
        _chat_target_id = ""
        try:
            if selected_incident_candidate:
                _chat_target_id = selected_incident_candidate.get("id", "") or ""
        except Exception:
            _chat_target_id = ""
        if not _chat_target_id:
            _chat_target_id = target_device_id if target_device_id else ""
        _chat_ci = _build_ci_context_for_chat(_chat_target_id) if _chat_target_id else {}
        if _chat_ci:
            _vendor = _chat_ci.get("vendor", "") or "Unknown"
            _os = _chat_ci.get("os", "") or "Unknown"
            _model = _chat_ci.get("model", "") or "Unknown"
            st.caption(f"対象機器: {_chat_target_id}   Vendor: {_vendor}   OS: {_os}   Model: {_model}")

        # クイック質問
        q1, q2, q3 = st.columns(3)
        if "chat_quick_text" not in st.session_state:
            st.session_state.chat_quick_text = ""

        with q1:
            if st.button("設定バックアップ", use_container_width=True):
                st.session_state.chat_quick_text = "この機器で、現在の設定を安全にバックアップする手順とコマンド例を教えてください。"
        with q2:
            if st.button("ロールバック", use_container_width=True):
                st.session_state.chat_quick_text = "この機器で、変更をロールバックする代表的な手順（候補）と注意点を教えてください。"
        with q3:
            if st.button("確認コマンド", use_container_width=True):
                st.session_state.chat_quick_text = "今回の症状を切り分けるために、まず実行すべき確認コマンドを優先度順に教えてください。"

        if st.session_state.chat_quick_text:
            st.info("クイック質問（コピーして貼り付け）")
            st.code(st.session_state.chat_quick_text)

        if st.session_state.chat_session is None and api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3-12b-it")
            st.session_state.chat_session = model.start_chat(history=[])

        tab1, tab2 = st.tabs(["💬 会話", "📝 履歴"])

        with tab1:
            if st.session_state.messages:
                last_msg = st.session_state.messages[-1]
                if last_msg["role"] == "assistant":
                    st.info("🤖 最新の回答")
                    with st.container(height=300):
                        st.markdown(last_msg["content"])

            st.markdown("---")
            prompt = st.text_area(
                "質問を入力してください:",
                height=70,
                placeholder="Ctrl+Enter または 送信ボタンで送信",
                key="chat_textarea"
            )

            col1, col2, col3 = st.columns([3, 1, 1])
            with col2:
                send_button = st.button("送信", type="primary", use_container_width=True)
            with col3:
                if st.button("クリア"):
                    st.session_state.messages = []
                    st.rerun()

            if send_button and prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})

                if st.session_state.chat_session:
                    target_id = ""
                    try:
                        if selected_incident_candidate:
                            target_id = selected_incident_candidate.get("id", "") or ""
                    except Exception:
                        target_id = ""
                    if not target_id:
                        try:
                            target_id = target_device_id
                        except Exception:
                            target_id = ""
                    ci = _build_ci_context_for_chat(target_id) if target_id else {}
                    ci_prompt = f"""あなたはネットワーク運用（NOC/SRE）の実務者です。
次の CI 情報と Config 抜粋を必ず参照して、具体的に回答してください。

【CI (JSON)】
{json.dumps(ci, ensure_ascii=False, indent=2)}

【ユーザーの質問】
{prompt}

回答ルール:
- CI/Config に基づく具体手順・コマンド例を提示する
- 追加確認が必要なら、質問は最小限に絞る
- 不明な前提は推測せず「CIに無いので確認が必要」と明記する
"""

                    with st.spinner("AI が回答を生成中..."):
                        try:
                            response = generate_content_with_retry(
                                st.session_state.chat_session.model,
                                ci_prompt,
                                stream=False
                            )
                            if response:
                                full_response = response.text if hasattr(response, "text") else str(response)
                                if not full_response.strip():
                                    full_response = "AI応答が空でした。"
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                            else:
                                st.error("AIからの応答がありませんでした。")
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
                    st.rerun()

        with tab2:
            if st.session_state.messages:
                history_container = st.container(height=400)
                with history_container:
                    for i, msg in enumerate(st.session_state.messages):
                        icon = "🤖" if msg["role"] == "assistant" else "👤"
                        with st.container(border=True):
                            st.markdown(f"{icon} **{msg['role'].upper()}** (メッセージ {i+1})")
                            st.markdown(msg["content"])
            else:
                st.info("会話履歴はまだありません。")

# ベイズ更新トリガー
if st.session_state.trigger_analysis and st.session_state.live_result:
    if st.session_state.verification_result:
        pass
    st.session_state.trigger_analysis = False
