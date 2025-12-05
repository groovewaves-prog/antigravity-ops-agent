"""
Google Antigravity AIOps Agent - データモジュール
将来的な拡張性（LAG, 3重化など）に対応するため、属性を「metadata」辞書で管理します。
"""

import json
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class NetworkNode:
    id: str
    layer: int
    type: str
    parent_id: Optional[str] = None
    redundancy_group: Optional[str] = None
    # ★変更: 特定のフィールドではなく、汎用的な辞書にする
    metadata: Dict[str, Any] = field(default_factory=dict)

def load_topology_from_json(filename: str = "topology.json") -> Dict[str, NetworkNode]:
    topology = {}
    
    if not os.path.exists(filename):
        return {}

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for key, value in raw_data.items():
            node = NetworkNode(
                id=key,
                layer=value.get("layer", 99),
                type=value.get("type", "UNKNOWN"),
                parent_id=value.get("parent_id"),
                redundancy_group=value.get("redundancy_group"),
                # JSON内の "metadata" フィールド、または旧 "internal_redundancy" をここに統合
                metadata=value.get("metadata", {})
            )
            
            # (互換性維持) もしJSONに古い internal_redundancy があれば metadata に入れる
            if value.get("internal_redundancy"):
                node.metadata["redundancy_type"] = value.get("internal_redundancy")
                
            topology[key] = node
            
    except Exception as e:
        print(f"Error loading topology: {e}")
        return {}

    return topology

TOPOLOGY = load_topology_from_json()
