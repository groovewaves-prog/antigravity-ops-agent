"""
Google Antigravity AIOps Agent - Main Application (Optimized Final)
"""
import streamlit as st
import graphviz
import os
import time
import logging
import google.generativeai as genai

# モジュールインポート
from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import run_diagnostic_simulation, generate_config_from_intent, generate_health_check_commands
from verifier import verify_log_content, format_verification_report

# =====================================================
# ロギング設定
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# 設定クラス
# =====================================================
class Config:
    # モデル設定
    MODEL_NAME = "gemma-3-12b-it"
    MODEL_TEMP = 0.0
    
    # セッション管理
    MAX_MESSAGES = 50
    MAX_MESSAGE_AGE = 3600
    CLEANUP_INTERVAL = 100
    
    # リトライ設定
    MAX_RETRIES = 3
    RETRY_BACKOFF = 1.0
    
    # フィルタリング
    SYSTEM_MESSAGE_KEYWORDS = ["診断結果に基づき", "障害報告", "以下の結果"]

# =====================================================
# ヘルパー関数
# =====================================================

def initialize_session_state():
    defaults = {
        'messages': [],
        'chat_session': None,
        'live_result': None,
        'trigger_analysis': False,
        'verification_result': None,
        'current_mode': None,
        'current_scenario': None,
        '_message_count': 0,
        'generated_conf': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def add_message(role: str, content: str):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    st.session_state._message_count += 1
    
    if st.session_state._message_count % Config.CLEANUP_INTERVAL == 0:
        cleanup_old_messages()

def cleanup_old_messages():
    messages = st.session_state.messages
    now = time.time()
    valid_msgs = []
    
    for msg in messages:
        age = now - msg.get("timestamp", 0)
        if age < Config.MAX_MESSAGE_AGE:
            valid_msgs.append(msg)
            
    if len(valid_msgs) > Config.MAX_MESSAGES:
        valid_msgs = valid_msgs[-Config.MAX_MESSAGES:]
        
    st.session_state.messages = valid_msgs
    logger.info(f"Cleaned up messages. Count: {len(valid_msgs)}")

def send_message_with_retry(chat_session, message: str) -> str:
    for attempt in range(Config.MAX_RETRIES):
        try:
            response = chat_session.send_message(message)
            return response.text
        except Exception as e:
            if attempt == Config.MAX_RETRIES - 1:
                raise e
            time.sleep(Config.RETRY_BACKOFF * (2 ** attempt))
    return "Error"

def load_config_by_id(device_id):
    """安全なConfig読み込み"""
    config_dir = "configs"
    if not os.path.exists(config_dir): return None
    
    safe_id = os.path.basename(device_id)
    path = os.path.join(config_dir, f"{safe_id}.txt")
    
    if not os.path.abspath(path).startswith(os.path.abspath(config_dir)):
        return None
        
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None

def render_topology(alarms, root_cause_node, root_severity="CRITICAL"):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    alarm_map = {a.device_id: a for a in alarms}
    alarmed_ids = set(alarm_map.keys())
    
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9"
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"
        
        if node.metadata.get("redundancy_type"):
            label += f"\n[{node.metadata['redundancy_type']} Redundancy]"
        if node.metadata.get("vendor"):
            label += f"\n[{node.metadata['vendor']}]"

        if root_cause_node and node_id == root_cause_node.id:
            node_severity = alarm_map[node_id].severity if node_id in alarm_map else root_severity
            color = "#ffcdd2" if node_severity == "CRITICAL" else "#fff9c4"
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
        elif node_id in alarmed_ids:
            color = "#fff9c4"
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=str(penwidth), fontcolor=fontcolor)
    
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent = TOPOLOGY.get(node.parent_id)
            if parent and parent.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() 
                           if n.redundancy_group == parent.redundancy_group and n.id != parent.id]
                for p in partners: graph.edge(p, node_id)
    return graph

# =====================================================
# メイン処理
# =====================================================

st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")
st.title("⚡ Antigravity AI Agent (Live Demo)")

# API Key
api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

# サイドバー
with st.sidebar:
    st.header("⚡ 運用モード選択")
    app_mode = st.radio("機能選択:", ("🚨 障害対応", "🔧 設定生成"))
    st.markdown("---")
    
    selected_scenario = "正常稼働"
    if app_mode == "🚨 障害対応":
        SCENARIO_MAP = {
            "基本・広域障害": ["正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害"],
            "WAN Router": ["4. [WAN] 電源障害：片系", "5. [WAN] 電源障害：両系", "6. [WAN] BGPルートフラッピング", "7. [WAN] FAN故障", "8. [WAN] メモリリーク"],
            "Firewall": ["9. [FW] 電源障害：片系", "10. [FW] 電源障害：両系", "11. [FW] FAN故障", "12. [FW] メモリリーク"],
            "L2 Switch": ["13. [L2SW] 電源障害：片系", "14. [L2SW] 電源障害：両系", "15. [L2SW] FAN故障", "16. [L2SW] メモリリーク"],
            "Live": ["99. [Live] Cisco実機診断"]
        }
        cat = st.selectbox("対象カテゴリ:", list(SCENARIO_MAP.keys()))
        selected_scenario = st.radio("発生シナリオ:", SCENARIO_MAP[cat])
    
    if api_key:
        st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

# セッション初期化
initialize_session_state()

if st.session_state.current_mode != app_mode:
    st.session_state.current_mode = app_mode
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.rerun()

# -----------------------------------------------------
# モードA: 障害対応
# -----------------------------------------------------
if app_mode == "🚨 障害対応":
    if st.session_state.current_scenario != selected_scenario:
        st.session_state.current_scenario = selected_scenario
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.session_state.live_result = None
        st.session_state.trigger_analysis = False
        st.session_state.verification_result = None
        st.rerun()

    # アラーム生成ロジック
    alarms = []
    root_severity = "CRITICAL"
    target_device_id = None

    if "WAN全回線断" in selected_scenario:
        target_device_id = "WAN_ROUTER_01"
        alarms = simulate_cascade_failure("WAN_ROUTER_01", TOPOLOGY)
    elif "FW片系障害" in selected_scenario:
        target_device_id = "FW_01_PRIMARY"
        alarms = [Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING")]
        root_severity = "WARNING"
    elif "L2SWサイレント障害" in selected_scenario:
        target_device_id = "L2_SW_01"
        alarms = [Alarm("AP_01", "Connection Lost", "CRITICAL"), Alarm("AP_02", "Connection Lost", "CRITICAL")]
    else:
        if "[WAN]" in selected_scenario: target_device_id = "WAN_ROUTER_01"
        elif "[FW]" in selected_scenario: target_device_id = "FW_01_PRIMARY"
        elif "[L2SW]" in selected_scenario: target_device_id = "L2_SW_01"

        if target_device_id:
            if "電源障害：片系" in selected_scenario:
                alarms = [Alarm(target_device_id, "Power Supply 1 Failed", "WARNING")]
                root_severity = "WARNING"
            elif "電源障害：両系" in selected_scenario:
                if target_device_id == "FW_01_PRIMARY":
                    alarms = [Alarm(target_device_id, "Power Supply: Dual Loss", "CRITICAL")]
                else:
                    alarms = simulate_cascade_failure(target_device_id, TOPOLOGY, "Power Supply: Dual Loss")
                root_severity = "CRITICAL"
            elif "BGP" in selected_scenario:
                alarms = [Alarm(target_device_id, "BGP Flapping", "WARNING")]
                root_severity = "WARNING"
            elif "FAN" in selected_scenario:
                alarms = [Alarm(target_device_id, "Fan Fail", "WARNING")]
                root_severity = "WARNING"
            elif "メモリ" in selected_scenario:
                alarms = [Alarm(target_device_id, "Memory High", "WARNING")]
                root_severity = "WARNING"

    # 推論実行
    root_cause = None
    reason = ""
    if alarms:
        engine = CausalInferenceEngine(TOPOLOGY)
        res = engine.analyze_alarms(alarms)
        root_cause = res.root_cause_node
        reason = res.root_cause_reason
        if res.severity == "CRITICAL": root_severity = "CRITICAL"
        elif res.severity == "WARNING": root_severity = "WARNING"

    # 画面描画
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Network Status")
        st.graphviz_chart(render_topology(alarms, root_cause, root_severity), use_container_width=True)
        
        if root_cause:
            if root_severity == "CRITICAL":
                st.markdown(f'<div style="color:#d32f2f;background:#fdecea;padding:10px;border-radius:5px;">🚨 緊急アラート：{root_cause.id} ダウン</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:#856404;background:#fff3cd;padding:10px;border-radius:5px;">⚠️ 警告：{root_cause.id} 異常検知 (稼働中)</div>', unsafe_allow_html=True)
            st.caption(f"理由: {reason}")
        
        if root_cause or ("[Live]" in selected_scenario):
            st.markdown("---")
            st.info("🛠 **自律調査エージェント**")
            if st.button("🚀 診断実行 (Auto-Diagnostic)", type="primary"):
                if not api_key: st.error("API Key Required")
                else:
                    with st.status("Agent Operating...", expanded=True) as status:
                        st.write("🔌 Executing Diagnostics...")
                        target_node_obj = TOPOLOGY.get(target_device_id) if target_device_id else None
                        
                        try:
                            res = run_diagnostic_simulation(selected_scenario, target_node_obj, api_key)
                            st.session_state.live_result = res
                            
                            if res["status"] == "SUCCESS":
                                st.write("✅ Data Acquired.")
                                log_content = res.get('sanitized_log', "")
                                verif = verify_log_content(log_content)
                                st.session_state.verification_result = verif
                                status.update(label="Complete!", state="complete", expanded=False)
                            elif res["status"] == "SKIPPED":
                                status.update(label="Skipped", state="complete")
                            else:
                                st.write("❌ Check Failed.")
                                st.session_state.verification_result = {"ping_status": "Conn Failed"}
                                status.update(label="Target Unreachable", state="error", expanded=False)
                            
                            st.session_state.trigger_analysis = True
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            if st.session_state.live_result:
                res = st.session_state.live_result
                if res["status"] == "SUCCESS":
                    st.success("🛡️ **Data Sanitized**: 機密情報はマスク処理済み")
                    with st.expander("📄 取得ログ (Sanitized)", expanded=True):
                        st.code(res["sanitized_log"], language="text")
                    if st.session_state.verification_result:
                        with st.expander("✅ 自動検証結果 (Rule-Based Check)", expanded=True):
                            v = st.session_state.verification_result
                            st.write(f"- **Ping**: {v.get('ping_status')}")
                            st.write(f"- **Interface**: {v.get('interface_status')}")
                            st.write(f"- **Hardware**: {v.get('hardware_status')}")
                elif res["status"] == "ERROR":
                    st.error(f"診断結果: {res['error']}")

    with col2:
        st.subheader("AI Analyst Report")
        if not api_key: st.stop()

        should_start = (st.session_state.chat_session is None) and (selected_scenario != "正常稼働")
        if should_start:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(Config.MODEL_NAME, generation_config={"temperature": Config.MODEL_TEMP})
                
                system_prompt = ""
                if st.session_state.live_result:
                    ld = st.session_state.live_result
                    log_c = ld.get('sanitized_log') or f"Error: {ld.get('error')}"
                    system_prompt = f"診断結果に基づきレポートを作成せよ。\nST: {ld['status']}\nLog: {log_c}"
                elif root_cause:
                    conf = load_config_by_id(root_cause.id)
                    system_prompt = f"障害報告: {root_cause.id} ({root_cause.type})\n理由: {reason}\nSeverity: {root_severity}"
                    if conf: system_prompt += f"\nConfig:\n{conf}"
                
                if system_prompt:
                    chat = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])
                    with st.spinner("Analyzing..."):
                        resp = send_message_with_retry(chat, "状況報告をお願いします。")
                        st.session_state.chat_session = chat
                        add_message("assistant", resp)
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.trigger_analysis and st.session_state.chat_session:
            ld = st.session_state.live_result
            log_c = ld.get('sanitized_log') or f"Error: {ld.get('error')}"
            verif_text = ""
            if st.session_state.verification_result:
                verif_text = format_verification_report(st.session_state.verification_result)
            
            prompt = f"""
            診断コマンドを実行しました。以下の結果に基づき『ネクストアクション実行レポート』を作成してください。
            【診断データ】ST: {ld['status']}, Log: {log_c}
            {verif_text}
            【出力要件】0.診断結論(最重要), 1.接続結果, 2.ログ分析, 3.推奨アクション
            """
            
            add_message("user", "診断結果を分析してください。")
            with st.spinner("Analyzing Diagnostic Data..."):
                try:
                    resp = send_message_with_retry(st.session_state.chat_session, prompt)
                    add_message("assistant", resp)
                except Exception as e: st.error(str(e))
            st.session_state.trigger_analysis = False
            st.rerun()

        chat_container = st.container(height=600)
        with chat_container:
            for msg in st.session_state.messages:
                if any(k in msg["content"] for k in Config.SYSTEM_MESSAGE_KEYWORDS): continue
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("質問..."):
            add_message("user", prompt)
            with chat_container:
                with st.chat_message("user"): st.markdown(prompt)
            if st.session_state.chat_session:
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                resp = send_message_with_retry(st.session_state.chat_session, prompt)
                                add_message("assistant", resp)
                                st.markdown(resp)
                            except Exception as e: st.error(str(e))

# -----------------------------------------------------
# モードB: 設定生成
# -----------------------------------------------------
elif app_mode == "🔧 設定生成":
    st.subheader("🔧 Intent-Based Config Generator")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("自然言語の指示(Intent)から、メーカー仕様に合わせたConfigを自動生成します。")
        tid = st.selectbox("対象機器:", list(TOPOLOGY.keys()))
        tnode = TOPOLOGY[tid]
        st.caption(f"Device: {tnode.metadata.get('vendor')} / {tnode.metadata.get('os')}")
        
        cconf = load_config_by_id(tid)
        with st.expander("現在のConfig"):
            st.code(cconf if cconf else "(No current config)")
        
        intent = st.text_area("Intent:", height=150, placeholder="例: Gi0/1にVLAN100を割り当てて。")
        if st.button("✨ Config生成", type="primary"):
            if not api_key or not intent: st.error("Missing Info")
            else:
                with st.spinner("Generating..."):
                    try:
                        gconf = generate_config_from_intent(tnode, cconf, intent, api_key)
                        st.session_state.generated_conf = gconf
                    except Exception as e: st.error(str(e))
    with c2:
        st.subheader("📝 Generated Config")
        if "generated_conf" in st.session_state:
            st.markdown(st.session_state.generated_conf)
            st.success("生成完了")
        
        st.markdown("---")
        st.subheader("🔍 Health Check")
        if st.button("正常性確認コマンド生成"):
             if not api_key: st.error("API Key Required")
             else:
                 with st.spinner("Generating..."):
                     try:
                         cmds = generate_health_check_commands(tnode, api_key)
                         st.code(cmds)
                     except Exception as e: st.error(str(e))
