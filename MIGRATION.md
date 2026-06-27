# 遷移到第二台機 (PC-B) — 給對面 Claude Code 的指南

> 目的：把本訓練 repo 在另一台「路徑 / 版本 / 位置都相同」的電腦上跑起來，
> 與 PC-A 平行訓練。**結論：代碼走 git，大資料 (logs/wandb) 不搬。**

## 0. 在 PC-B 第一步 (取得 repo)

PC-B 還沒有 repo 時，clone 一行(需先有 GitHub SSH key)：

```bash
git clone git@github.com:me0608623/charge_navrl.git /home/aa/IsaacLab
cd /home/aa/IsaacLab && git checkout wdclean-repro-20260429
```

## 1. 一鍵設定 (clone 後)

```bash
bash /home/aa/IsaacLab/scripts/setup_machine_b.sh          # checkout + pull + 驗證
bash /home/aa/IsaacLab/scripts/setup_machine_b.sh --smoke  # 多跑 10-step 煙霧測試
```

腳本會：fetch+checkout 分支 → 啟用 `env_isaaclab` → (僅在 import 失敗時) `./isaaclab.sh -i`
→ 驗證 USD → (可選) 煙霧測試。腳本可獨立執行：若 repo 不存在會自動 clone，所以
PC-B 全新機時把這支腳本單獨 scp/貼過去 `bash setup_machine_b.sh` 即可。

## 2. USD 機器人模型放在哪裡？← 不用手動搬

USD 已被 git **強制追蹤**，clone 一完成就在正確位置，policy 自動找到：

| 檔案 | 大小 | 誰在用 |
|------|------|--------|
| `assets/usd/charge/charge.usd` | 12 K | **active 任務 (charge_skrl / rnn_car)** |
| `source/.../charge_skrl/charge.usd` | 67 M | 同 charge_skrl，亦隨 git 走 |

解析邏輯 `source/isaaclab_tasks/.../charge_skrl/cfg/charge_cfg.py:20-26`，依序：
1. 環境變數 `CHARGE_USD_PATH` (最高優先)
2. `{REPO_ROOT}/assets/usd/charge/charge.usd` ← **clone 後即命中，免設定**
3. `/home/aa/usd/charge/charge.usd` (本機 fallback)

⚠️ 注意：舊任務 `charge` / `charge_sb3` / `.charge` 在 `charge_cfg.py:34` **硬寫**
`/home/aa/usd/charge/charge.usd`。現在的 rnn_car 訓練不碰它們；若 PC-B 要跑那些舊任務，
該絕對路徑須存在(複製 12K 的 USD 過去即可)。

## 3. 大資料怎麼辦

- **全新平行訓練** → 什麼都不用搬。logs/ 與 wandb/ 各機獨立產生，wandb 各自 sync 雲端。
- **要續訓某個 checkpoint** → 只 rsync 那一個 run 資料夾(幾百 MB)，別搬整包 70 G：
  ```bash
  rsync -avhP PC-A:/home/aa/IsaacLab/logs/rnn_car/<run>/ /home/aa/IsaacLab/logs/rnn_car/<run>/
  ```

## 4. ⚠️ 平行訓練 — 兩台機別共用同一 branch

PC-B 若要 commit 訓練改動：

```bash
git checkout -b wdclean-repro-20260429-pcB   # 開自己的 branch
# 改動 → commit → push 到自己的 branch；push 前一律先 git fetch
```

代碼從主 branch (`wdclean-repro-20260429`) 拉、結果推到自己的 branch，避免互相 force-push 覆蓋。

## 5. 正式訓練指令 (從頭訓 SA1 vaux)

```bash
PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 ./isaaclab.sh -p \
  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa1_v3f_vaux --headless --feat_norm --aux_epochs 8 \
  --run_name sa1_v3f_vaux_ne1024_s42
```

> vaux lineage 因 GRU + feat_norm 架構變更，**不相容舊 v3f checkpoint，必須從頭訓**。
