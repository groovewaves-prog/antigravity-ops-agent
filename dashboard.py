import streamlit as st
import pandas as pd
import random

def render_intelligent_alarm_viewer(bayes_engine, selected_scenario, current_alarms):
    """
    AIOps時代のインシデント管理ビューアー
    行クリックによる選択（マスター・ディテール方式）を採用
    """
    st.markdown("### 🛡️ AIOps インシデント・コックピット")
    
    # --- アラーム数計算 ---
    actual_alarm_count = len(current_alarms)
    
    if selected_scenario == "正常稼働":
        raw_alarm_count = 0
        suppressed_count = 0
        incident_count = 0
        noise_reduction_rate = "100%"
    else:
        noise_factor = random.randint(12, 25) 
        if actual_alarm_count == 0: actual_alarm_count = 1
        
        raw_alarm_count = actual_alarm_count * noise_factor
        suppressed_count = raw_alarm_count - 1 # 基本は集約
        incident_count = 1 # 本来は推論結果の件数
        
        reduction = (suppressed_count / raw_alarm_count) * 100
        noise_reduction_rate = f"{reduction:.1f}%"

    # KPI表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📉 ノイズ削減率", value=noise_reduction_rate, delta="高効率稼働中")
    with col2:
        st.metric(label="📨 処理したアラーム総数", value=f"{raw_alarm_count}件", delta=f"-{suppressed_count}件 (抑制済)", delta_color="inverse")
    with col3:
        st.metric(label="🚨 要対応インシデント", value=f"{incident_count}件", delta="対処が必要")

    st.markdown("---")
    
    # ランキング取得
    ranking = bayes_engine.get_ranking()
    
    # データ整形
    data = []
    for rank, candidate in enumerate(ranking[:5], 1): # 上位5件
        prob = candidate["prob"]
        
        if prob > 0.8:
            status = "🔴 危険 (根本原因)"
            action = "🚀 自動修復が可能"
            impact = "大"
        elif prob > 0.4:
            status = "🟡 警告 (被疑箇所)"
            action = "🔍 詳細調査を推奨"
            impact = "中"
        else:
            status = "⚪ 監視中"
            action = "👁️ 静観"
            impact = "小"

        data.append({
            "順位": rank,
            "ID": candidate['id'], 
            "AI診断": status,
            "根本原因分析": f"デバイス: {candidate['id']}\n原因種別: {candidate['type']}",
            "確信度": prob,
            "影響範囲": impact,
            "推奨アクション": action,
            "Type": candidate['type']
        })

    df = pd.DataFrame(data)

    # ★ここがポイント: on_select="rerun" でクリック時に再描画させる
    st.info("💡 ヒント: インシデントの行をクリックすると、右側に詳細分析と復旧プランが表示されます。")
    
    event = st.dataframe(
        df,
        column_order=["順位", "AI診断", "根本原因分析", "確信度", "影響範囲", "推奨アクション"],
        column_config={
            "順位": st.column_config.NumberColumn("#", format="%d", width="small"),
            "AI診断": st.column_config.TextColumn("ステータス", width="medium"),
            "根本原因分析": st.column_config.TextColumn("📌 根本原因候補", width="large"),
            "確信度": st.column_config.ProgressColumn("AI確信度", format="%.1f", min_value=0, max_value=1),
            "推奨アクション": st.column_config.TextColumn("🤖 Next Action"),
            "影響範囲": st.column_config.TextColumn("影響度", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        height=250,
        on_select="rerun",          
        selection_mode="single-row" 
    )
    
    selected_candidate = None
    
    # 選択された行があればそれを返す。なければトップ(1位)を返す
    if len(event.selection.rows) > 0:
        idx = event.selection.rows[0]
        selected_row = df.iloc[idx]
        target_id = selected_row["ID"]
        target_type = selected_row["Type"]
        
        # IDとTypeが一致する候補をrankingから探す
        for cand in ranking:
            if cand['id'] == target_id and cand['type'] == target_type:
                selected_candidate = cand
                break
    else:
        # デフォルト選択（1位）
        selected_candidate = ranking[0]
        
    return selected_candidate
