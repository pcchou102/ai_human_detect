# AI 文本偵測器

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> 🚀 **立即體驗線上 Demo：** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

基於 Streamlit 的 AI 生成文本偵測工具，結合 Perplexity（困惑度）與 Burstiness（節奏分析）進行智能判斷。

## 🎯 功能特色

### 核心指標
- **Perplexity（困惑度）**：衡量文本的可預測性
- **Burstiness（節奏分析）**：分析句子長度變異度

### 視覺化
- 📊 句子長度分佈圖
- 🎨 詞彙熱力圖（紅色=AI特徵，綠色=人類特徵）

### 匯出功能
- 📥 一鍵下載詳細分析報告（.txt）
- 包含完整指標、判斷結果與解釋說明

---

## 🚀 快速開始

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 啟動應用
```bash
streamlit run ai_detector.py
```

### 開啟瀏覽器
訪問 http://localhost:8501

---

## 🧮 技術原理

### Perplexity 計算
本系統使用啟發式算法，基於以下特徵：
- 詞彙多樣性 (Type-Token Ratio)
- 常見詞比例分析
- AI 轉折詞偵測（furthermore, moreover 等）
- 平均詞長計算
- 標點符號使用模式

### Burstiness 計算
- 句子長度標準差 / 平均值
- 反映寫作節奏的變化程度

### 判斷邏輯
```python
AI 可能性 = Perplexity 分數 × 60% + Burstiness 分數 × 40%

if AI 可能性 > 50%:
    判定為「AI 生成」
else:
    判定為「人類撰寫」
```

---

## 📝 使用範例

### AI 生成文本特徵
```
Artificial Intelligence represents a transformative 
branch of computer science. Furthermore, it aims to 
create intelligent machines. Moreover, these systems 
can analyze vast amounts of data efficiently.
```
- Perplexity: **低** (< 60)
- Burstiness: **低** (< 0.5)
- 判定：**AI 生成**

### 人類撰寫文本特徵
```
I totally messed up the meeting today! Omg, my cat 
literally jumped on the keyboard. So embarrassing. 
But hey, at least everyone laughed?
```
- Perplexity: **高** (> 80)
- Burstiness: **高** (> 0.7)
- 判定：**人類撰寫**

---

## 🌐 Streamlit Cloud 部署

### 1. 建立 GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. 連結 Streamlit Cloud
1. 訪問 [share.streamlit.io](https://share.streamlit.io)
2. 點擊 "New app"
3. 選擇您的 GitHub repo
4. **Main file path:** `ai_detector.py`
5. **Python version:** 3.9+
6. 點擊 "Deploy"

### 3. 等待部署（通常 < 2 分鐘）

---

## 📦 專案結構

```
HW5/
├── ai_detector.py        # 主程式
├── requirements.txt      # 依賴套件
└── README.md            # 說明文檔
```

---

## 📚 依賴套件

```txt
streamlit>=1.28.0
nltk>=3.8
numpy>=1.21.0,<2.0.0
```

---

## ⚠️ 注意事項

1. **語言限制**：目前僅支援英文文本
2. **文本長度**：建議至少 50 個單字以獲得準確結果
3. **準確度**：本工具使用啟發式算法，僅供參考

---

## 🛠️ 故障排除

### 問題：NLTK 資料下載失敗
```python
# 手動下載
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 問題：Streamlit 啟動失敗
```bash
# 確認已安裝依賴
pip install -r requirements.txt

# 使用完整路徑啟動
python -m streamlit run ai_detector.py
```

---

## 🎨 介面預覽

### 主畫面
- 左側：文本輸入區與範例按鈕
- 右側：分析結果與指標卡片
- 底部：視覺化圖表（句長分佈 + 詞彙熱力圖）

### 側邊欄
- 工具介紹
- 核心指標說明
- 分析原理
- 使用說明

---

## 📧 聯絡資訊

- GitHub Issues: [回報問題](https://github.com)
- Email: your-email@example.com

---

## 📄 授權

MIT License

---

**開發日期：** 2025年12月14日  
**框架版本：** Streamlit 1.28+  
**偵測引擎：** AI Text Detector v1.0
