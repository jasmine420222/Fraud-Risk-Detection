
# 跨境支付反欺诈风控
这是一套可直接运行、可交互展示的端到端风控项目 Demo：
- 模拟风控数据集（synthetic_fraud_data.csv）
- 交互式 Dashboard

> 数据为**模拟生成**

---

## 1. 项目结构

```
fraud_streamlit_demo/
  app.py
  requirements.txt
  data/
    synthetic_fraud_data.csv
    sample_rows.csv
```

---

## 2. 本地运行

### 2.1 安装依赖
建议用虚拟环境：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2.2 启动 Streamlit
在项目根目录执行：

```bash
streamlit run app.py
```
