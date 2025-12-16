import json
import os
import re
import google.generativeai as genai
from enum import Enum
from typing import List, Dict, Any

# AIOpsの判定ステータス
class HealthStatus(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"

class InferenceEngine:
    def __init__(self, topology_file: str, config_dir: str):
        """
        InferenceEngineの初期化
        :param topology_file: トポロジー定義ファイルのパス (topology.json)
        :param config_dir: コンフィグファイルが格納されているディレクトリ (/configs)
        """
        self.topology = self._load_topology(topology_file)
        self.config_dir = config_dir
        
        # Google Generative AIの設定
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it to use the AI engine.")
        
        genai.configure(api_key=api_key)
        
        # コストと精度のバランスが良いモデルを選択
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _load_topology(self, path: str) -> Dict:
        """JSONファイルからトポロジー情報を読み込む"""
        if not os.path.exists(path):
            print(f"Warning: Topology file {path} not found.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _read_config(self, device_id: str) -> str:
        """
        デバイスIDに対応するコンフィグファイルを読み込む。
        ファイルが存在しない場合は、AIにその旨を伝えるテキストを返す。
        """
        config_path = os.path.join(self.config_dir, f"{device_id}.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading config file: {str(e)}"
        return "Config file not found in repository."

    def _sanitize_text(self, text: str) -> str:
        """
        LLMに送信する前に機密情報（パスワード、コミュニティ名など）をマスクする。
        """
        # 1. Junos/Ciscoの暗号化パスワードのマスク
        # Junos: encrypted-password "$1$Ec7...."; -> "$1$********";
        text = re.sub(r'(encrypted-password\s+)"[^"]+"', r'\1"********"', text)
        
        # Cisco: password 7 0822455D0A16 or secret 5 $1$....
        text = re.sub(r'(password|secret)\s+(\d)\s+\S+', r'\1 \2 ********', text)
        text = re.sub(r'(username\s+\S+\s+secret)\s+\d\s+\S+', r'\1 5 ********', text)

        # 2. SNMP Communityのマスク
        text = re.sub(r'(snmp-server community)\s+\S+', r'\1 ********', text)
        
        # 3. 必要であればグローバルIPなどもここでマスク可能だが、
        #    トポロジー解析にIPが必要な場合があるため、今回は認証情報のみとする。
        
        return text

    def analyze_redundancy_depth(self, device_id: str, alerts: List[str]) -> Dict[str, Any]:
        """
        LLMを使用して、Configとアラートから冗長性の深度を判定する。
        """
        # アラートがない場合は正常として即返す
        if not alerts:
            return {
                "status": HealthStatus.NORMAL,
                "reason": "No active alerts detected.",
                "impact_type": "NONE"
            }

        # 必要なコンテキスト情報の収集
        device_info = self.topology.get(device_id, {})
        raw_config = self._read_config(device_id)
        metadata = device_info.get('metadata', {})

        # --- サニタイズ処理 (Security Consideration) ---
        # AIへ送信する前にConfigとAlertsから機密情報を除去する
        safe_config = self._sanitize_text(raw_config)
        safe_alerts = [self._sanitize_text(a) for a in alerts]

        # --- プロンプトエンジニアリング ---
        prompt = f"""
あなたはネットワーク運用のエキスパートAIです。
以下の情報に基づき、現在発生しているアラートが「サービス停止(CRITICAL)」を引き起こしているか、
それとも「冗長機能によりサービスは維持されている(WARNING)」状態かを判定してください。

### 対象デバイス
- **Device ID**: {device_id}
- **Metadata**: {json.dumps(metadata)}
  (メタデータに 'hw_inventory' がある場合は、PSU数などを参考にしてください。無い場合はモデル名から推測してください)

### 設定ファイル (Config - Sanitized)
この設定内容から、LAG (LACP/Port-Channel) 構成や、インターフェースの役割を読み解いてください。
機密情報はマスクされています。

```text
{config_content}
発生中のアラートリスト
{json.dumps(alerts)}

判定ルール (Thinking Process)
１．電源(PSU)障害の判定:
  ・デバイスが複数のPSUを持っていると推測され、かつ「全て」ではなく「一部」のPSUのみがFailしている場合。
  ・判定: WARNING (理由: Redundancy Lost - 片系運転中)
  ・全てのPSUがFail、または単一PSUデバイスのFailの場合。
  ・判定: CRITICAL (理由: Power Outage)
２．インターフェース/LAG障害の判定:
　・Configを確認し、Downしている物理インターフェースが LAG (Port-Channel / ae / Bond) のメンバーか確認してください。
　・親となる論理インターフェース(Port-Channel Xなど)自体のアラートが出ていなければ、親はUpしているとみなします。
　・メンバーのみのDownの場合。
　・判定: WARNING (理由: Degraded - 帯域縮退)
　・LAG非構成ポートのDown、または親LAG自体のDownの場合。
　・判定: CRITICAL (理由: Link Down - Service Impacting)
３．その他:
　・上記に当てはまらない不明なエラーや、CPU高負荷などは内容に応じて判断してください。

出力フォーマット
以下のJSON形式のみを出力してください。Markdownのコードブロック(json ...)は含めないでください。
{{
"status": "STATUS_STRING", // "NORMAL", "WARNING", "CRITICAL" のいずれか
"reason": "判定理由を簡潔に記述",
"impact_type": "IMPACT_STRING" // "NONE", "DEGRADED", "REDUNDANCY_LOST", "OUTAGE", "UNKNOWN" のいずれか
}}
"""
try:
        # LLMへの問い合わせ
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # レスポンスの解析
        response_text = response.text.strip()
        # 万が一Markdown記法が含まれていた場合の除去
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        result_json = json.loads(response_text)
        
        # 文字列ステータスをEnumに変換
        status_str = result_json.get("status", "CRITICAL").upper()
        
        if status_str in ["GREEN", "NORMAL"]:
            health_status = HealthStatus.NORMAL
        elif status_str in ["YELLOW", "WARNING"]:
            health_status = HealthStatus.WARNING
        else:
            health_status = HealthStatus.CRITICAL

        return {
            "status": health_status,
            "reason": result_json.get("reason", "AI provided no reason"),
            "impact_type": result_json.get("impact_type", "UNKNOWN")
        }

    except Exception as e:
        # AI推論エラー時は安全側に倒してCRITICAL扱い、またはエラー情報を返す
        print(f"[!] AI Inference Error for {device_id}: {e}")
        return {
            "status": HealthStatus.CRITICAL,
            "reason": f"AI Analysis Failed: {str(e)}",
            "impact_type": "AI_ERROR"
        }


if name == "main":
# テスト用設定
TEST_TOPOLOGY = "topology.json"
TEST_CONFIG_DIR = "./configs"

# 簡易的なファイル生成（ディレクトリが存在しない場合のみ作成）
if not os.path.exists(TEST_CONFIG_DIR):
    os.makedirs(TEST_CONFIG_DIR)

# テスト実行
try:
    engine = InferenceEngine(TEST_TOPOLOGY, TEST_CONFIG_DIR)
    
    print("--- AI Redundancy Analysis Test ---\n")

    # ケース1: PSU片系障害
    test_device = "WAN_ROUTER_01"
    test_alerts = ["Environment: PSU 1 Status Failed", "Environment: PSU 2 Status OK"]
    print(f"Testing Device: {test_device}")
    print(f"Alerts: {test_alerts}")
    result = engine.analyze_redundancy_depth(test_device, test_alerts)
    print(f"Result: {result['status'].value} ({result['impact_type']})")
    print(f"Reason: {result['reason']}\n")

except ValueError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")
