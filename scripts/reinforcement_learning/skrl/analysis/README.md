# Analysis — 訓練事後分析

訓練完成後的數據分析與報告生成工具。

| 檔案 | 功能 |
|------|------|
| `analyze_wandb_run.py` | WandB Run 分析器。支援三種模式: (1) monitor-log — 解析即時 log 健康度 (2) report — 生成單 run 摘要與 plots (3) compare — 多 run 排名比較。理解 rl/*, aux/*, charge/*, curriculum/* metric layout |
