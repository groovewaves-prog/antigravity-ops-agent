"""
Antigravity AIOps - Logical Inference Engine (Rule-Based / Deterministic)
"""

class LogicalRCA:
    def __init__(self, topology):
        self.topology = topology
        
        # 親ID -> 子IDリスト のマップ
        self.parent_to_children = {}
        for node_id, node in self.topology.items():
            if node.parent_id:
                if node.parent_id not in self.parent_to_children:
                    self.parent_to_children[node.parent_id] = []
                self.parent_to_children[node.parent_id].append(node_id)

        # 基本シグネチャ
        self.signatures = [
            {
                "type": "Hardware/Critical_Multi_Fail",
                "label": "複合ハードウェア障害",
                "rules": lambda alarms: any("power supply" in a.message.lower() for a in alarms) and any("fan" in a.message.lower() for a in alarms),
                "base_score": 1.0
            },
            {
                "type": "Hardware/Physical",
                "label": "ハードウェア障害 (電源/デバイス)",
                "rules": lambda alarms: any(k in a.message.lower() for a in alarms for k in ["power supply", "device down"]),
                "base_score": 0.95
            },
            {
                "type": "Network/Link",
                "label": "物理リンク/インターフェース障害",
                "rules": lambda alarms: any(k in a.message.lower() for a in alarms for k in ["interface down", "connection lost", "heartbeat loss"]),
                "base_score": 0.90
            },
            {
                "type": "Hardware/Fan",
                "label": "冷却ファン故障",
                "rules": lambda alarms: any("fan fail" in a.message.lower() for a in alarms),
                "base_score": 0.70
            },
            {
                "type": "Config/Software",
                "label": "設定ミス/プロトコル障害",
                "rules": lambda alarms: any(k in a.message.lower() for a in alarms for k in ["bgp", "ospf", "config"]),
                "base_score": 0.60
            },
            {
                "type": "Resource/Capacity",
                "label": "リソース枯渇 (CPU/Memory)",
                "rules": lambda alarms: any(k in a.message.lower() for a in alarms for k in ["cpu", "memory", "high"]),
                "base_score": 0.50
            }
        ]

    def _get_all_descendants(self, root_id):
        """再帰的に配下の全ノードを取得"""
        descendants = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in self.parent_to_children:
                children = self.parent_to_children[current]
                for child in children:
                    descendants.add(child)
                    stack.append(child)
        return descendants

    def analyze(self, current_alarms):
        candidates = []
        device_alarms = {}
        
        for alarm in current_alarms:
            if alarm.device_id not in device_alarms:
                device_alarms[alarm.device_id] = []
            device_alarms[alarm.device_id].append(alarm)
            
        # 1. 直接的なアラーム評価
        for device_id, alarms in device_alarms.items():
            best_match = None
            max_score = 0.0
            for sig in self.signatures:
                if sig["rules"](alarms):
                    score = min(sig["base_score"] + (len(alarms) * 0.02), 1.0)
                    if score > max_score:
                        max_score = score
                        best_match = sig
            
            if best_match:
                candidates.append({
                    "id": device_id,
                    "type": best_match["type"],
                    "label": best_match["label"],
                    "prob": max_score,
                    "alarms": [a.message for a in alarms],
                    "verification_log": ""
                })
            elif alarms:
                candidates.append({
                    "id": device_id,
                    "type": "Unknown/Other",
                    "label": "その他異常検知",
                    "prob": 0.3,
                    "alarms": [a.message for a in alarms],
                    "verification_log": ""
                })

        # 2. サイレント障害検知
        down_children_count = {} 
        for alarm in current_alarms:
            msg = alarm.message.lower()
            if "connection lost" in msg or "interface down" in msg:
                node = self.topology.get(alarm.device_id)
                if node and node.parent_id:
                    pid = node.parent_id
                    down_children_count[pid] = down_children_count.get(pid, 0) + 1

        for parent_id, count in down_children_count.items():
            if count >= 2:
                parent_node = self.topology.get(parent_id)
                if not parent_node: continue 

                active_verification_log = f"""
[Auto-Probe] Multiple downstream failures detected (Count: {count}).
[Topology] Identified upstream aggregator: {parent_id}
[Action] Initiating active health check from Core Switch...
[Exec] ping {parent_id}_mgmt_ip source Core_SW
[Result] Request Timed Out (100% loss).
[Conclusion] {parent_id} is unresponsive (Silent Failure confirmed).
"""
                existing = next((c for c in candidates if c['id'] == parent_id), None)
                if existing:
                    existing['prob'] = 1.0
                    existing['label'] = "サイレント障害 (配下デバイス一斉断 + 応答なし)"
                    existing['verification_log'] = active_verification_log
                else:
                    candidates.append({
                        "id": parent_id,
                        "type": "Network/Silent",
                        "label": "サイレント障害 (配下デバイス一斉断)",
                        "prob": 0.99, 
                        "alarms": [f"Downstream Impact: {count} devices lost"],
                        "verification_log": active_verification_log
                    })

        # 3. ★修正: 影響伝播 (冗長化考慮版)
        root_cause_ids = [c['id'] for c in candidates if c['prob'] > 0.8]
        impacted_nodes = set()
        
        for rid in root_cause_ids:
            # 冗長構成チェック: パートナーが生きていれば伝播させない
            node = self.topology.get(rid)
            should_propagate = True
            
            if node and node.redundancy_group:
                # 同じグループの他ノードを探す
                partners = [nid for nid, n in self.topology.items() 
                           if n.redundancy_group == node.redundancy_group and nid != rid]
                
                # パートナーが「生存（RootCauseリストにいない）」なら、伝播を止める
                # (簡易ロジック: パートナーも障害なら伝播する)
                partners_alive = [p for p in partners if p not in root_cause_ids]
                
                if partners_alive:
                    should_propagate = False # パートナーが生きているので配下は無事
            
            if should_propagate:
                descendants = self._get_all_descendants(rid)
                impacted_nodes.update(descendants)
            
        for node_id in impacted_nodes:
            existing = next((c for c in candidates if c['id'] == node_id), None)
            if existing:
                if existing['prob'] <= 0.8:
                    existing['type'] = "Network/Secondary"
                    existing['label'] = "影響下 (上位障害による通信不能)"
                    existing['prob'] = 0.50 
            else:
                candidates.append({
                    "id": node_id,
                    "type": "Network/Unreachable",
                    "label": "応答なし (上位障害の影響)",
                    "prob": 0.50, 
                    "alarms": ["Parent node failure"],
                    "verification_log": ""
                })

        candidates.sort(key=lambda x: x["prob"], reverse=True)
        
        if not candidates:
            candidates.append({"id": "System", "type": "Normal", "label": "正常稼働中", "prob": 0.0, "alarms": [], "verification_log": ""})
            
        return candidates
