import streamlit as st
import graphviz
import os
import google.generativeai as genai

from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm
from network_ops import run_diagnostic_commands 

st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- トポロジー描画 ---
def render_topology(alarms, root_cause_node):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    alarmed_ids = {a.device_id for a in alarms}
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9"
        penwidth = "1"
        if root_cause_node and node_id == root_cause_node.id:
            color = "#ffcdd2"
            penwidth = "3"
        elif node_id in alarmed_ids:
            color = "#fff9c4"
        graph.node(node_id, label=f"{node_id}\n({node.type})", fillcolor=color, color='black', penwidth=penwidth)
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent = TOPOLOGY.get(node.parent_id)
            if parent and parent.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() if n.redundancy_group == parent.redundancy_group and n.id != parent.id]
                for p in partners: graph.edge(p, node_id)
    return graph

# --- Config読み込み ---
def load_config_by_id(device_id):
    path = f"configs/{device_id}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return f.read()
    return None

# --- UI構築 ---
st.title("⚡ Antigravity AI Agent (Live Demo)")

api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

with st.sidebar:
    st.header("⚡ 運用モード選択")
    selected_scenario = st.radio(
        "シナリオ:", 
        ("正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害", "4. [Live] Cisco実機診断")
    )
    if not api_key:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"
    st.session_state.messages = []
    st.session_state.chat_session = None 
    st.session_state.live_result = None

if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.live_result = None
    st.rerun()

# --- モード分岐 ---
if selected_scenario != "4. [Live] Cisco実機診断":
    # === シミュレーションモード ===
    alarms = []
    if selected_scenario == "1. WAN全回線断":
        alarms = [Alarm("WAN_ROUTER_01", "Down", "CRITICAL"), Alarm("FW_01_PRIMARY", "Unreach", "WARNING"), Alarm("AP_01", "Unreach", "CRITICAL")]
    elif selected_scenario == "2. FW片系障害":
        alarms = [Alarm("FW_01_PRIMARY", "HB Loss", "WARNING")]
    elif selected_scenario == "3. L2SWサイレント障害":
        alarms = [Alarm("AP_01", "Lost", "CRITICAL"), Alarm("AP_02", "Lost", "CRITICAL")]

    root_cause = None
    reason = ""
    if alarms:
        engine = CausalInferenceEngine(TOPOLOGY)
        res = engine.analyze_alarms(alarms)
        root_cause = res.root_cause_node
        reason = res.root_cause_reason

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Topology")
        st.graphviz_chart(render_topology(alarms, root_cause), use_container_width=True)
        if root_cause:
            st.markdown(f'<div style="color:#d32f2f;background:#fdecea;padding:10px;border-radius:5px;">🚨 緊急アラート：{root_cause.id} ダウン</div>', unsafe_allow_html=True)
            st.caption(f"理由: {reason}")

    with col2:
        st.subheader("AI Chat")
        if not api_key: st.stop()
        
        # コンテナでスクロール固定
        chat_container = st.container(height=500)
        
        if st.session_state.chat_session is None and selected_scenario != "正常稼働":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0})
            
            conf = load_config_by_id(root_cause.id)
            sys_prompt = f"障害: {root_cause.id}。理由: {reason}。"
            if conf: sys_prompt += f"\nConfig:\n{conf}"
            else: sys_prompt += "\nConfigなし。"
            
            history = [{"role": "user", "parts": [sys_prompt + "\n状況報告とネクストアクションを提示せよ。"]}]
            chat = model.start_chat(history=history)
            try:
                response = chat.send_message("状況報告")
                st.session_state.chat_session = chat
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e: st.error(str(e))

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input("指示を入力..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"): st.markdown(prompt)
            if st.session_state.chat_session:
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            res = st.session_state.chat_session.send_message(prompt)
                            st.markdown(res.text)
                            st.session_state.messages.append({"role": "assistant", "content": res.text})

else:
    # === Live実機モード ===
    st.subheader("🌍 Real-world Network Diagnostic")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.info("Target: sandbox-iosxe-latest-1.cisco.com")
        if st.button("🚀 自律調査実行 (SSH)", type="primary"):
            if not api_key: st.error("API Key Required")
            else:
                with st.status("Autopilot Running...") as status:
                    st.write("Connecting to Cisco Sandbox...")
                    res = run_diagnostic_commands()
                    st.session_state.live_result = res
                    
                    if res["status"] == "SUCCESS":
                        status.update(label="Complete!", state="complete")
                    else:
                        # エラー時
                        status.update(label="Failed", state="error")
                        # ★ここを追加：詳細なエラー内容を赤字で表示
                        st.error(f"詳細エラー: {res['error']}")        
        if st.session_state.live_result and st.session_state.live_result["status"] == "SUCCESS":
            with st.expander("取得ログ (Sanitized)", expanded=True):
                st.code(st.session_state.live_result["sanitized_log"])

    with c2:
        st.subheader("AI Analysis")
        if st.session_state.live_result and st.session_state.live_result["status"] == "SUCCESS":
            chat_container = st.container(height=500)
            
            if st.session_state.chat_session is None:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0})
                log = st.session_state.live_result["sanitized_log"]
                sys_prompt = f"以下はCisco実機ログです。分析してください。\n{log}"
                history = [{"role": "user", "parts": [sys_prompt]}]
                chat = model.start_chat(history=history)
                with st.spinner("Analyzing..."):
                    res = chat.send_message("レポート作成")
                    st.session_state.chat_session = chat
                    st.session_state.messages.append({"role": "assistant", "content": res.text})

            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if prompt := st.chat_input("質問..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"): st.markdown(prompt)
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            res = st.session_state.chat_session.send_message(prompt)
                            st.markdown(res.text)
                            st.session_state.messages.append({"role": "assistant", "content": res.text})
