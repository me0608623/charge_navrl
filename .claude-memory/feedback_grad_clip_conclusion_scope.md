---
name: grad_clip_conclusion_scope
description: Experimental conclusions about grad clipping must be scoped to specific regime, not generalized
type: feedback
---

結論必須收斂到具體條件，不能寫成一般性定理。

**Why:** 用戶指出我把 "merged clip 在此 run 不適合" 錯誤推廣為 "merged clip 錯" 或 "physical necessity"。

**How to apply:** 
- 實驗結論必須註明具體條件（architecture, vf_coeff, max_grad_norm, init, reward scale）
- 不寫「X 方法錯」，寫「在 A/B/C 條件下，X 會導致 Y」
- 邊界要明確：「不保證換了 vf_coeff / reward scale / value init / critic warmup 後仍然成立」
- 區分「merged@1.0 不適合這個 regime」vs「所有 merged clip 都不行」

例：此次 A/B 實驗結論的精確寫法：
- 對目前 train_marl_rnn.py 架構、初始化階段、vf_coeff=0.025、max_grad_norm=1.0 而言，separate clip 有實證必要性
- 理由不是「理論上必然正確」，而是「在目前 loss scale 下，merged clip 會使 actor early learning 幾乎停滯」（gn_v/gn_p ≈ 1629x at iter 3）
