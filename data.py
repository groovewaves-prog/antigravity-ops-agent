# -*- coding: utf-8 -*-
"""
Google Antigravity AIOps Agent - Data Module (Optimized Final)
"""

import json
import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

# =====================================================
# ロギング設定
# =====================================================
logger = logging.getLogger(__name__)

# =====================================================
# 定数定義
# =====================================================
class TopologyConstants:
    DEFAULT_TOPOLOGY_FILE = "topology.json"
    DEFAULT_LAYER = 99
    DEFAULT_TYPE = "UNKNOWN"
    MAX_LAYER = 100

# =====================================================
# データクラス定義
# =====================================================
@dataclass
class NetworkNode:
    """ネットワークノードを表現するデータクラス"""
    id: str
    layer: int
    type: str
    parent_id: Optional[str] = None
    redundancy_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """データ検証"""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Invalid node id: {self.id}")
        
        # Layer検証
        if not isinstance(self.layer, int):
            try:
                self.layer = int(self.layer)
            except (ValueError, TypeError):
                logger.warning(f"Node {self.id}: invalid layer, using default")
                self.layer = TopologyConstants.DEFAULT_LAYER
        
        # Metadata検証
        if not isinstance(self.metadata, dict):
            logger.warning(f"Node {self.id}: metadata must be dict, resetting")
            self.metadata = {}

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

# =====================================================
# デフォルトデータ (JSONがない場合のバックアップ)
# =====================================================
DEFAULT_RAW_DATA = {
  "WAN_ROUTER_01": {
    "layer": 1, "type": "ROUTER", 
    "metadata": { "redundancy_type": "PSU", "model": "Cisco ISR" }
  },
  "FW_01_PRIMARY": {
    "layer": 2, "type": "FIREWALL", "parent_id": "WAN_ROUTER_01",
    "redundancy_group": "FW_HA_GROUP",
    "metadata": { "redundancy_type": "PSU", "role": "Active" }
  },
  "FW_01_SECONDARY": {
    "layer": 2, "type": "FIREWALL", "parent_id": "WAN_ROUTER_01",
    "redundancy_group": "FW_HA_GROUP",
    "metadata": { "redundancy_type": "PSU", "role": "Standby" }
  },
  "CORE_SW_01": {
    "layer": 3, "type": "SWITCH", "parent_id": "FW_01_PRIMARY",
    "metadata": { "redundancy_type": "PSU" }
  },
  "L2_SW_01": {
    "layer": 4, "type": "SWITCH", "parent_id": "CORE_SW_01",
    "metadata": { "redundancy_type": "PSU", "location": "Floor 1" }
  },
  "L2_SW_02": {
    "layer": 4, "type": "SWITCH", "parent_id": "CORE_SW_01",
    "metadata": { "redundancy_type": "PSU", "location": "Floor 2" }
  },
  "AP_01": { "layer": 5, "type": "ACCESS_POINT", "parent_id": "L2_SW_01" },
  "AP_02": { "layer": 5, "type": "ACCESS_POINT", "parent_id": "L2_SW_01" },
  "AP_03": { "layer": 5, "type": "ACCESS_POINT", "parent_id": "L2_SW_02" },
  "AP_04": { "layer": 5, "type": "ACCESS_POINT", "parent_id": "L2_SW_02" }
}

# =====================================================
# トポロジー読み込み関数
# =====================================================
def load_topology_from_json(filename: str = TopologyConstants.DEFAULT_TOPOLOGY_FILE) -> Dict[str, NetworkNode]:
    """JSONファイルからトポロジーを読み込み"""
    topology = {}
    raw_data = {}

    # ファイル読み込み試行
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            logger.info(f"Loaded topology from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}. Using default data.")
            raw_data = DEFAULT_RAW_DATA
    else:
        logger.info(f"{filename} not found. Using default data.")
        raw_data = DEFAULT_RAW_DATA

    # オブジェクト変換
    for key, value in raw_data.items():
        try:
            node = NetworkNode(
                id=key,
                layer=value.get("layer", TopologyConstants.DEFAULT_LAYER),
                type=value.get("type", TopologyConstants.DEFAULT_TYPE),
                parent_id=value.get("parent_id"),
                redundancy_group=value.get("redundancy_group"),
                metadata=value.get("metadata", {})
            )
            # 互換性維持
            if value.get("internal_redundancy"):
                node.metadata["redundancy_type"] = value.get("internal_redundancy")
            
            topology[key] = node
        except Exception as e:
            logger.error(f"Error parsing node {key}: {e}")
            continue
            
    # バリデーション実行
    if topology:
        validate_topology(topology)

    return topology

# =====================================================
# トポロジー検証関数
# =====================================================
def validate_topology(topology: Dict[str, NetworkNode]) -> bool:
    """整合性チェック"""
    issues = []
    
    for node_id, node in topology.items():
        # ID不一致
        if node.id != node_id:
            issues.append(f"Node ID mismatch: {node_id}")
        
        # 親存在チェック
        if node.parent_id and node.parent_id not in topology:
            issues.append(f"Node {node_id} has invalid parent: {node.parent_id}")
        
        # 循環参照チェック
        if _has_circular_reference(node, topology):
            issues.append(f"Circular reference detected: {node_id}")

    if issues:
        for i in issues: logger.warning(i)
        return False
    return True

def _has_circular_reference(node: NetworkNode, topology: Dict[str, NetworkNode], visited=None) -> bool:
    if visited is None: visited = set()
    if node.id in visited: return True
    if not node.parent_id: return False
    
    visited.add(node.id)
    parent = topology.get(node.parent_id)
    if parent:
        return _has_circular_reference(parent, topology, visited)
    return False

# =====================================================
# グローバル変数
# =====================================================
TOPOLOGY = load_topology_from_json()
