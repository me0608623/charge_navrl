# 論文架構圖 (RNN-as-MOT)

對應配置：**79D obs (act_hist off) + velocity aux (predict_dim 13) + hybrid skip ON + 非對稱 critic**。
維度來源：`scripts/.../modular_rnn_models.py`、`train_rnn_car_wdclip.py`（已核對代碼）。

## 檔案

| 檔 | 說明 | 來源格式 |
|----|------|----------|
| `fig1_compact.drawio` / `.png` / `.svg` | **主架構（推薦）** 手刻緊湊 2D 佈局：編碼鏈 → MOT 解碼 → Actor-Critic heads | draw.io XML |
| `fig1_rnn_mot_architecture.mmd` / `.png` | 主架構（Mermaid 自動佈局版，較高，備份用） | Mermaid |
| `fig2_feature_extractor.mmd` / `.png` / `.svg` | 2-branch 特徵抽取細節（circular Conv1d + state MLP） | Mermaid |
| `fig3_rnn_vs_explicit_mot.mmd` / `.png` / `.svg` | RNN-as-MOT vs 傳統 explicit MOT 對照（motivation 圖） | Mermaid |
| `fig4_baseline_aux_only.mmd` / `.png` / `.svg` | baseline（hybrid OFF，rl_input 91D）對照 | Mermaid |

## 重現

```bash
# Mermaid → PNG/SVG（系統 Chrome，免抓 chromium）
export PUPPETEER_SKIP_DOWNLOAD=true PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
npx -y -p @mermaid-js/mermaid-cli mmdc -i figX.mmd -o figX.png -b white -s 3 \
  -c mermaid-config.json -p puppeteer-config.json
npx -y -p @mermaid-js/mermaid-cli mmdc -i figX.mmd -o figX.svg -b transparent \
  -c mermaid-config.json -p puppeteer-config.json

# draw.io XML → PNG/SVG（headless，需 xvfb + drawio AppImage）
xvfb-run -a /path/to/drawio -x -f png -s 3 --crop -b 12 \
  -o fig1_compact.png fig1_compact.drawio --no-sandbox --disable-gpu
```

論文匯出：用 `.svg`（向量，LaTeX `\includegraphics` 可直接吃）或在 draw.io 開 `.drawio` 後 `Export as → PDF`（勾 Transparent + Crop）。
