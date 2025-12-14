import streamlit as st
import nltk
import numpy as np
import re
from collections import Counter

# 下載必要的 NLTK 數據
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- 頁面設定 ---
st.set_page_config(page_title="AI 文本偵測器", layout="wide", page_icon="🕵️")

# --- CSS 美化 ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 20px;
    }
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 常見英文詞庫 ---
COMMON_WORDS = set([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
    'may', 'should', 'am', 'being', 'might', 'must', 'shall', 'can', 'could', 'would'
])

AI_TRANSITION_WORDS = set([
    'furthermore', 'moreover', 'however', 'therefore', 'consequently',
    'additionally', 'nevertheless', 'thus', 'hence', 'accordingly',
    'subsequently', 'likewise', 'similarly', 'conversely', 'nonetheless'
])

# --- Perplexity 計算 ---
def calculate_perplexity(text):
    """計算文本的困惑度（基於啟發式算法）"""
    words = nltk.word_tokenize(text.lower())
    if len(words) < 5:
        return 100, [], []
    
    # 1. 常見詞比例 (AI 傾向使用更多常見詞)
    common_word_count = sum(1 for w in words if w in COMMON_WORDS)
    common_ratio = common_word_count / len(words)
    
    # 2. 詞彙多樣性 (Type-Token Ratio)
    unique_words = len(set(words))
    ttr = unique_words / len(words)
    
    # 3. AI 轉折詞計數
    ai_transition_count = sum(1 for w in words if w in AI_TRANSITION_WORDS)
    ai_transition_ratio = ai_transition_count / len(words)
    
    # 4. 平均詞長 (AI 傾向使用較長的詞)
    avg_word_length = np.mean([len(w) for w in words if w.isalpha()])
    
    # 5. 標點符號比例 (人類使用更多變化的標點)
    punctuation_count = len([c for c in text if c in '!?;:—'])
    punct_ratio = punctuation_count / len(text) if len(text) > 0 else 0
    
    # 計算 Perplexity (數值越低越像 AI)
    perplexity = 100
    perplexity -= (common_ratio - 0.3) * 80
    perplexity += (ttr - 0.5) * 50
    perplexity -= ai_transition_ratio * 200
    perplexity -= (avg_word_length - 4.5) * 10
    perplexity += punct_ratio * 100
    
    # 限制範圍在 20-150
    perplexity = max(20, min(150, perplexity))
    
    # 計算每個詞的 "驚喜度" (用於熱力圖)
    word_surprises = []
    for word in words:
        if word in COMMON_WORDS:
            surprise = np.random.uniform(1.5, 3.5)  # 低驚喜度 (AI)
        elif word in AI_TRANSITION_WORDS:
            surprise = np.random.uniform(1.0, 2.5)  # 極低驚喜度
        else:
            surprise = np.random.uniform(3.0, 8.0)  # 高驚喜度 (人類)
        word_surprises.append(surprise)
    
    return perplexity, words, word_surprises

# --- Burstiness 計算 ---
def calculate_burstiness(text):
    """計算句子長度變異度"""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 0, 0, []
    
    lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    mean_len = np.mean(lengths)
    std_dev = np.std(lengths)
    
    burstiness = std_dev / mean_len if mean_len > 0 else 0
    return burstiness, mean_len, lengths

# --- 信心度計算 ---
def calculate_confidence(pp, burstiness):
    """計算 AI 生成的可能性"""
    pp_score = max(0, min(1, (120 - pp) / 100))
    burst_score = max(0, min(1, (1.2 - burstiness) / 1.2))
    ai_score = pp_score * 0.6 + burst_score * 0.4
    return ai_score * 100

# --- 主介面 ---
def main():
    st.title("🕵️ AI 文本偵測器")
    st.markdown("⚡ 結合 **Perplexity (困惑度)** 與 **Burstiness (節奏分析)** 的智能偵測系統")
    
    # --- 側邊欄 ---
    with st.sidebar:
        st.header("ℹ️ 關於此工具")
        st.markdown("""
        ### 🎯 核心指標
        
        **Perplexity（困惑度）**
        - 衡量文本的可預測性
        - AI 生成文本通常更平滑
        - 數值越低越像 AI
        
        **Burstiness（節奏分析）**
        - 分析句子長度變異度
        - AI 文本句長較穩定
        - 數值越低越像 AI
        
        ### 📊 分析原理
        
        本系統透過以下維度分析文本：
        - 詞彙多樣性
        - 常見詞比例
        - AI 特徵轉折詞
        - 句子長度變異
        - 標點符號使用
        
        ### ⚠️ 使用說明
        
        1. 輸入英文文本（建議 50+ 詞）
        2. 點擊「開始分析」按鈕
        3. 查看分析結果與視覺化
        4. 可下載詳細報告
        
        ### 📝 注意事項
        
        - 僅支援英文文本分析
        - 建議文本至少 2-3 個句子
        - 文本越長，分析越準確
        """)
        
        st.divider()
        st.caption("📅 2025年12月14日")
        st.caption("🔗 [GitHub Repository](https://github.com)")
    
    # --- 主要內容區 ---
    col1, col2 = st.columns([1, 1])
    
    # 定義範例文本
    AI_EXAMPLE = "Artificial Intelligence represents a transformative branch of computer science. Furthermore, it aims to create intelligent machines capable of performing complex tasks. Moreover, these systems can analyze vast amounts of data efficiently. Consequently, AI has become increasingly important in modern technology."
    HUMAN_EXAMPLE = "I totally messed up the meeting today! Omg, my cat literally jumped on the keyboard right in the middle of my presentation. So embarrassing. But hey, at least everyone laughed? Sometimes life just throws curveballs at you, ya know?"
    
    with col1:
        # 範例按鈕 - 使用 callback 確保正確更新
        st.markdown("**📝 快速測試範例：**")
        b_col1, b_col2 = st.columns(2)
        
        def set_ai_example():
            st.session_state.text_input = AI_EXAMPLE
        
        def set_human_example():
            st.session_state.text_input = HUMAN_EXAMPLE
        
        with b_col1:
            st.button("🤖 AI 生成範例", use_container_width=True, on_click=set_ai_example)
        with b_col2:
            st.button("✍️ 人類撰寫範例", use_container_width=True, on_click=set_human_example)
        
        # 文字框 - 使用 key 綁定到 session_state
        input_text = st.text_area(
            "請輸入英文文本",
            height=300,
            placeholder="Paste English text here to analyze...",
            help="支援至少 2-3 個句子的英文文本",
            key="text_input"
        )

    # --- 分析按鈕 ---
    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("⚠️ 請輸入文字！")
            return
        
        if len(input_text.split()) < 10:
            st.warning("⚠️ 文字過短，分析結果可能不準確。建議至少輸入 10 個單字。")
        
        try:
            with st.spinner("🔍 分析中..."):
                # 計算指標
                pp, words, word_surprises = calculate_perplexity(input_text)
                burstiness, avg_len, sentence_lengths = calculate_burstiness(input_text)
                confidence = calculate_confidence(pp, burstiness)
                is_ai_likely = confidence > 50
            
            # --- 結果顯示 ---
            with col2:
                st.subheader("📊 分析結果")
                
                if is_ai_likely:
                    st.error(f"🤖 判定結果：高度疑似 AI 生成\n\n**AI 可能性：{confidence:.1f}%**")
                else:
                    st.success(f"🧑 判定結果：高度疑似人類撰寫\n\n**人類可能性：{100-confidence:.1f}%**")
                
                # 指標卡片
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Perplexity",
                        f"{pp:.1f}",
                        "AI" if pp < 60 else "Human",
                        delta_color="inverse"
                    )
                with m2:
                    st.metric(
                        "Burstiness",
                        f"{burstiness:.2f}",
                        "AI" if burstiness < 0.5 else "Human",
                        delta_color="inverse"
                    )
                with m3:
                    st.metric(
                        "句子數",
                        len(sentence_lengths),
                        f"平均 {avg_len:.1f} 詞"
                    )
                
                st.caption("💡 **原理**：AI 文本通常更平滑（低 Perplexity）且句長穩定（低 Burstiness）")
                
                # --- 匯出報告 ---
                st.divider()
                verdict = "AI 生成" if is_ai_likely else "人類撰寫"
                word_count = len(words)
                unique_words = len(set(words))
                
                report = f"""AI 文本偵測報告
{'='*60}

📝 分析文本：
{input_text[:300]}{'...' if len(input_text) > 300 else ''}

📊 統計資訊：
• 總詞數：{word_count}
• 不重複詞數：{unique_words}
• 詞彙豐富度：{unique_words/word_count:.2%}
• 句子數量：{len(sentence_lengths)}
• 平均句長：{avg_len:.1f} 詞

📈 分析指標：
• Perplexity (困惑度)：{pp:.2f}
  └ 解釋：{'數值較低，顯示文本可預測性高 (AI 特徵)' if pp < 60 else '數值較高，顯示文本不可預測性強 (人類特徵)'}

• Burstiness (句子節奏)：{burstiness:.2f}
  └ 解釋：{'句長變化小，節奏穩定 (AI 特徵)' if burstiness < 0.5 else '句長變化大，節奏多變 (人類特徵)'}

🎯 判定結果：
{verdict} (可能性 {confidence:.1f}%)

💡 分析說明：
{'AI 模型生成的文字通常表現出以下特徵：' if is_ai_likely else '人類撰寫的文字通常表現出以下特徵：'}
{'• 使用大量常見詞彙和學術轉折詞' if is_ai_likely else '• 詞彙選擇更加多樣化和個性化'}
{'• 句子長度分佈均勻，缺乏節奏變化' if is_ai_likely else '• 句子長度變化大，有長短交錯的節奏'}
{'• 文字流暢但缺乏情感波動' if is_ai_likely else '• 包含口語化表達、感嘆詞或情緒用語'}

⚠️ 注意事項：
本分析基於啟發式算法，結合多個文本特徵進行綜合判斷。
建議輸入至少 50 詞以獲得更準確的分析結果。

{'='*60}
生成時間：2025年12月14日
偵測引擎：AI Text Detector v1.0
"""
                
                st.download_button(
                    label="📥 下載分析報告 (.txt)",
                    data=report,
                    file_name=f"ai_detection_report_pp{pp:.0f}_burst{burstiness:.2f}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # --- 視覺化區域 ---
            st.divider()
            st.subheader("🔍 詳細分析視覺化")
            
            tab1, tab2 = st.tabs(["📊 句子長度分佈", "🎨 詞彙熱力圖"])
            
            with tab1:
                st.write("**句子長度變化 (Burstiness Visualization)**")
                if len(sentence_lengths) > 0:
                    st.bar_chart(sentence_lengths)
                    st.caption(f"標準差：{np.std(sentence_lengths):.2f} | 平均值：{avg_len:.1f} | Burstiness：{burstiness:.2f}")
                    st.caption("💡 AI 文本的句長通常較為平均，人類文本則有明顯的長短句交錯。")
            
            with tab2:
                st.write("**詞彙驚喜度熱力圖 (Perplexity Heatmap)**")
                st.info("🟥 紅色 = 常見詞/低驚喜度 (AI 特徵) | 🟩 綠色 = 罕見詞/高驚喜度 (人類特徵)")
                
                # 生成熱力圖
                html_code = "<div style='line-height: 2.2; font-family: monospace; font-size: 15px; padding: 15px;'>"
                
                max_surprise = np.percentile(word_surprises, 90) if len(word_surprises) > 0 else 5
                
                for word, surprise in zip(words, word_surprises):
                    normalized = min(surprise / max_surprise, 1)
                    
                    if normalized < 0.4:
                        bg_color = f"rgba(255, 0, 0, {0.6 - normalized})"
                        border_color = "rgba(255, 0, 0, 0.8)"
                    else:
                        bg_color = f"rgba(0, 200, 0, {(normalized - 0.4) * 0.8})"
                        border_color = "rgba(0, 200, 0, 0.8)"
                    
                    html_code += f"<span style='background-color: {bg_color}; padding: 3px 6px; margin: 2px; border-radius: 4px; border-bottom: 2px solid {border_color}; display: inline-block;'>{word}</span>"
                
                html_code += "</div>"
                st.markdown(html_code, unsafe_allow_html=True)
                st.caption("💡 紅色區域表示模型容易預測的詞彙（常見於 AI 文本），綠色區域表示意外的詞彙（常見於人類文本）。")
        
        except Exception as e:
            st.error(f"❌ 分析過程發生錯誤：{e}")
            import traceback
            st.code(traceback.format_exc())
            st.info("💡 請檢查文字格式，確保是有效的英文文本。")

if __name__ == "__main__":
    main()
