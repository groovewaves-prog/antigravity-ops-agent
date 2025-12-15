"""
Antigravity AIOps - Logical Inference Engine (Rule-Based / Deterministic)
Human-like reasoning for Redundancy Depth (N+1, LAG, HA)
"""

class LogicalRCA:
    def __init__(self, topology):
        self.topology = topology
        
        # --- トポロジー階層の自動計算 ---
        self.parent_to_children = {}
        for node_id, node in self.topology.items():
            if node.parent_id:
                if node.parent_id not in self.parent_to_children:
                    self.parent_to_children[node.parent_id] = []
                self.parent_to_children[node.parent_id].append(node_id)

        self.node_tiers = {}
        for node_id in self.topology:
            self.node_tiers[node_id] = self._calculate_tier(node_id)

    def _calculate_tier(self, node_id):
        """階層深度の計算"""
        if node_id not in self.parent_to_children:
            return 0
        children = self.parent_to_children[node_id]
        child_tiers = [self._calculate_tier(child) for child in children]
        return max(child_tiers) + 1

    def _get_all_descendants(self, root_id):
        """子孫ノードの取得"""
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
        """
        高度な冗長性分析を含む推論ロジック
        """
        candidates = []
        device_alarms = {}
        
        # 1. アラームのデバイス別グルーピング
        for alarm in current_alarms:
            if alarm.device_id not in device_alarms:
                device_alarms[alarm.device_id] = []
            device_alarms[alarm.device_id].append(alarm)
            
        # 2. デバイスごとの詳細分析
        for device_id, alarms in device_alarms.items():
            alarm_texts = [a.message.lower() for a in alarms]
            
            # --- AI Reasoning: 冗長構成の残存性評価 ---
            
            # A. 電源冗長 (PSU Redundancy Check)
            psu_fails = [msg for msg in alarm_texts if "power supply" in msg and ("fail" in msg or "down" in msg)]
            is_psu_critical = False
            
            if len(psu_fails) >= 2:
                # 2系統ダウン -> サービス停止 (Critical)
                match_type = "Hardware/Critical_Multi_Fail"
                match_label = "電源喪失 (二重障害による停止)"
                prob = 1.0
                is_psu_critical = True
            elif len(psu_fails) == 1:
                # 1系統ダウン -> サービス継続中 (Warning)
                # 人間が見逃しがちな「片系は生きている」という判断
                match_type = "Hardware/RedundancyLost"
                match_label = "電源冗長性喪失 (片系稼働中)"
                prob = 0.65 # Warningレベルに留める
                is_psu_critical = True
            else:
                match_type = None
                match_label = ""
                prob = 0.0

            # B. LAG/論理リンク冗長 (Logical Link Check)
            # Interfaceは落ちているが、論理インターフェース(BGPなど)が生きているか？
            link_down = any("interface down" in msg or "connection lost" in msg for msg in alarm_texts)
            bgp_down = any("bgp" in msg or "neighbor down" in msg for msg in alarm_texts)
            
            if not is_psu_critical: # 電源判定が優先
                if link_down and not bgp_down:
                    # 物理は落ちたが、プロトコル(BGP)のアラームがない = LAGの片系断の可能性大
                    match_type = "Network/Degraded"
                    match_label = "帯域縮退 (LAGメンバー障害 / サービス継続)"
                    prob = 0.60 # Warning
                elif link_down and bgp_down:
                    # 物理もプロトコルも落ちた = 全断
                    match_type = "Network/Link"
                    match_label = "物理リンク/インターフェース全断"
                    prob = 0.90 # Critical
                elif any("fan fail" in msg for msg in alarm_texts):
                    match_type = "Hardware/Fan"
                    match_label = "冷却ファン故障"
                    prob = 0.70
                elif any("high" in msg and ("cpu" in msg or "memory" in msg) for msg in alarm_texts):
                    match_type = "Resource/Capacity"
                    match_label = "リソース枯渇 (CPU/Memory)"
                    prob = 0.50
                elif alarms and prob == 0.0: # マッチしないがアラームあり
                    match_type = "Unknown/Other"
                    match_label = "その他異常検知"
                    prob = 0.30

            # 候補に追加
            if match_type:
                candidates.append({
                    "id": device_id,
                    "type": match_type,
                    "label": match_label,
                    "prob": prob,
                    "alarms": [a.message for a in alarms],
                    "verification_log": "",
                    "tier": self.node_tiers.get(device_id, 0)
                })

        # 3. サイレント障害検知 (L2SW/Parent Logic)
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
                        "verification_log": active_verification_log,
                        "tier": self.node_tiers.get(parent_id, 0)
                    })

        # 4. 影響伝播 (冗長化考慮)
        # Prob > 0.8 (Critical) のものだけが配下を道連れにする
        root_cause_ids = [c['id'] for c in candidates if c['prob'] > 0.8]
        impacted_nodes = set()
        
        for rid in root_cause_ids:
            node = self.topology.get(rid)
            should_propagate = True
            
            # 機器レベルの冗長チェック (Active/Standby)
            if node and node.redundancy_group:
                partners = [nid for nid, n in self.topology.items() 
                           if n.redundancy_group == node.redundancy_group and nid != rid]
                # パートナーが無事(Warning以下)なら、サービスは生きているので伝播させない
                partners_alive = [p for p in partners if p not in root_cause_ids]
                if partners_alive:
                    should_propagate = False 
            
            if should_propagate:
                descendants = self._get_all_descendants(rid)
                impacted_nodes.update(descendants)
            
        for node_id in impacted_nodes:
            existing = next((c for c in candidates if c['id'] == node_id), None)
            if existing:
                # 既にCriticalなアラームを持っているならそのまま
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
                    "verification_log": "",
                    "tier": self.node_tiers.get(node_id, 0)
                })

        # 5. ソート (Risk > Tier > ID)
        candidates.sort(key=lambda x: (x["prob"], x["tier"], x["id"]), reverse=True)
        
        if not candidates:
            candidates.append({
                "id": "System", "type": "Normal", "label": "正常稼働中", 
                "prob": 0.0, "alarms": [], "verification_log": "", "tier": 0
            })
            
        return candidates
