"""
Google Antigravity AIOps Agent - データモジュール
外部の JSON ファイルからネットワークトポロジーを読み込みます。
"""

import json
import os
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class NetworkNode:
    id: str
    layer: int
    type: str
    parent_id: Optional[str] = None
    redundancy_group: Optional[str] = None

def load_topology_from_json(filename: str = "topology.json") -> Dict[str, NetworkNode]:
    """
    JSONファイルを読み込み、NetworkNodeオブジェクトの辞書として返す
    """
    topology = {}
    
    # ファイルが存在しない場合のフォールバック（デモ用）
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using empty topology.")
        return {}

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for key, value in raw_data.items():
            # JSONデータをデータクラスにマッピング
            node = NetworkNode(
                id=key,
                layer=value.get("layer", 99),
                type=value.get("type", "UNKNOWN"),
                parent_id=value.get("parent_id"),
                redundancy_group=value.get("redundancy_group")
            )
            topology[key] = node
            
    except Exception as e:
        print(f"Error loading topology: {e}")
        return {}

    return topology

# グローバル変数として公開（他モジュールからは data.TOPOLOGY としてアクセス）
TOPOLOGY = load_topology_from_json()
