"""
Google Antigravity AIOps Agent - Network Operations Module
"""
import re
import os
import time
import google.generativeai as genai
from netmiko import ConnectHandler

SANDBOX_DEVICE = {
    'device_type': 'cisco_nxos',
    'host': 'sandbox-nxos-1.cisco.com',
    'username': 'admin',
    'password': 'Admin_1234!',
    'port': 22,
    'global_delay_factor': 2,
    'banner_timeout': 30,
    'conn_timeout': 30,
}

def sanitize_output(text: str) -> str:
    rules = [
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(encrypted password) \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN_COMMUNITY>'),
        (r'(username \S+ privilege \d+ secret \d+) \S+', r'\1 <HIDDEN_SECRET>'),
        (r'\b(?!(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.)\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b', '<MASKED_PUBLIC_IP>'),
        (r'([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}', '<MASKED_MAC>'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text

def generate_fake_log_by_ai(scenario_name, target_node, api_key):
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    # ★変更: gemma-3-12b-it
    model = genai.GenerativeModel(
        "gemma-3-12b-it",
        generation_config={"temperature": 0.0}
    )
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    model_name = target_node.metadata.get("model", "Generic Device")
    hostname = target_node.id

    status_instructions = ""
    if "電源" in scenario_name and "片系" in scenario_name:
        status_instructions = """
        【状態定義: 電源冗長稼働中 (片系ダウン)】
        1. ハードウェアステータス: Power Supply 1: **Faulty / Failed**, Power Supply 2: **OK**
        2. サービス影響: なし (インターフェース UP, Ping 成功)
        3. エラーログ: 電源障害を示すSyslogまたはTrapを含めること。
        """
    elif "電源" in scenario_name and "両系" in scenario_name:
        status_instructions = """
        【状態定義: 全電源喪失】
        1. ログ: "Connection Refused" または再起動直後のブートログのみ。
        """
    elif "FAN" in scenario_name:
        status_instructions = """
        【状態定義: ファン故障】
        1. ハードウェアステータス: Fan Tray 1 **Failure**
        2. 温度: 上昇中だが閾値内 (Warning)
        3. サービス影響: なし
        """
    elif "メモリ" in scenario_name:
        status_instructions = """
        【状態定義: メモリリーク】
        1. メモリ使用率: **98%以上**
        2. プロセス: 特定のプロセス（例: SSHD, FlowMonitor等）が異常消費している様子を明確に示すこと。
        3. Syslog: メモリ割り当て失敗 (Malloc Fail) を含めること。
        """
    elif "BGP" in scenario_name:
        status_instructions = """
        【状態定義: BGPフラッピング】
        1. BGP状態: 特定のNeighborが Idle / Active を繰り返している。
        2. 物理IF: UP/UP
        """
    elif "全回線断" in scenario_name:
        status_instructions = """
        【状態定義: 物理リンクダウン】
        1. 主要インターフェース: **DOWN / DOWN** (Carrier Loss)
        2. Ping: 100% Loss
        """

    prompt = f"""
    あなたはネットワーク機器のCLIシミュレーターです。
    指定された機器スペックと障害シナリオに基づき、エンジニアが調査を行った際の「コマンド実行ログ」を生成してください。

    **対象機器スペック**:
    - Hostname: {hostname}
    - Vendor: {vendor}
    - OS: {os_type}
    - Model: {model_name}

    **発生シナリオ**: {scenario_name}

    {status_instructions}

    **出力要件**:
    1. **{vendor} {os_type}** の構文として正しいコマンドと出力形式を使用すること。
    2. 解説やMarkdown装飾は不要。**CLIの生テキストのみ**を出力すること。
    3. 矛盾する情報は含めないこと。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def generate_config_from_intent(target_node, current_config, intent_text, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    # ★変更: gemma-3-12b-it
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""
    ネットワーク設定生成。
    対象: {target_node.id} ({vendor} {os_type})
    現在のConfig: {current_config}
    Intent: {intent_text}
    出力: 投入用コマンドのみ (Markdownコードブロック)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Config Gen Error: {e}"

def generate_health_check_commands(target_node, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    # ★変更: gemma-3-12b-it
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"Netmiko正常性確認コマンドを3つ生成せよ。対象: {vendor} {os_type}。出力: コマンドのみ箇条書き"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Command Gen Error: {e}"

def generate_remediation_commands(scenario, analysis_result, target_node, api_key):
    """
    障害シナリオと分析結果に基づき、復旧コマンドを生成する
    """
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
    あなたは熟練したネットワークエンジニアです。以下の障害に対する「復旧用コマンド（Config）」を作成してください。
    
    対象デバイス: {target_node.id} ({target_node.metadata.get('vendor')} {target_node.metadata.get('os')})
    発生シナリオ: {scenario}
    AI分析結果: {analysis_result}
    
    【要件】
    1. 復旧に必要な具体的なコマンドのみを列挙すること。
    2. 説明文は不要。Markdownコードブロック形式で出力すること。
    3. 特権モード(enable)等は省略し、設定モード等の主要コマンドから書くこと。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Remediation Gen Error: {e}"

def run_diagnostic_simulation(scenario_type, target_node=None, api_key=None):
    time.sleep(1.5)
    
    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode(): ssh.enable()
                prompt = ssh.find_prompt()
                raw_output = f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            return {"status": "ERROR", "sanitized_log": "", "error": str(e)}
        return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
            
    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        return {"status": "ERROR", "sanitized_log": "", "error": "Connection timed out"}

    else:
        if api_key and target_node:
            raw_output = generate_fake_log_by_ai(scenario_type, target_node, api_key)
            return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
        else:
            return {"status": "ERROR", "sanitized_log": "", "error": "API Key or Target Node Missing"}
