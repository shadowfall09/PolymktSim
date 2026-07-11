# DCV Prototype — 进度日志

最后更新:2026-05-10

这份文档记录 **DCV (Decomposer + Verifier)** 方法在本仓库的当前进度,供后续 Claude Code 接续工作。

---

## 是什么

DCV 是相对 S0/S1/S2 多智能体协议之外的一条新路线。核心 idea:

1. **Decomposer (1 个 LLM 调用)**:把一道二元预测题分解成 `K` 条**原子可验证子声明 (sub-claim)**,每条带:
   - `supports`: 该子声明为真时把答案推向 YES 还是 NO
   - `weight`: 该子声明对最终答案的决定性 (0–1,K 条权重归一化和为 1)
2. **Verifier (K 个 LLM 调用,各 1 条 sub-claim)**:每个 verifier 看**完整证据池**,只判断单条 sub-claim 的真伪,输出 `p_true`、`confidence (0/1/2)` 和 `rationale`。
3. **Combiner**:对 K 个 verifier 结果做**置信度加权 log-pool**,得到最终 `p_yes`。

设计动机:S0/S1/S2 让 agent 同时做"检索 + 推理 + 概率"三件事,假阳/假阴混在一起;DCV 把"找事实"和"做判断"切开,逻辑上更可控、更可解释。

---

## 代码位置

| 文件 | 作用 |
|---|---|
| `src/agents/dcv_agent.py` | `DCVAgent`,实现 `decompose()` 和 `verify()`,支持 `prompt_version=v1/v2` |
| `src/aggregation/dcv_combiner.py` | `combine_dcv()`,置信度加权 logit pool |
| `run_dcv_prototype.py` | CLI 跑测脚本,支持 resume / baseline 对比 / 增量保存 |
| `data/dcv_full_qids.txt` | 完整评测集 375 个 qid |
| `data/dcv_test_qids.txt` | 50 题原型子集 |

Prompt 版本:
- **v1**:基础版,只要求"原子事实 + 权重 + supports"。
- **v2**:更严格,显式禁止"逻辑互补型 sub-claim"和"meta 解释规则",verifier 端要求**先引用证据原句**再下数,自查 `rationale` 与 `p_true` 是否同向。

---

## 实验结果(全部用 `openai/gpt-5.4-mini` via OpenRouter,K=3)

### 50 题原型 (`dcv_test_qids.txt`)

| 文件 | 配置 | Brier | Acc |
|---|---|---|---|
| `dcv_proto_50.jsonl`    | v1 prompts, verifier_temp=0.7 | 0.4725 | 36.0% |
| `dcv_proto_50_v2.jsonl` | v2 prompts, verifier_temp=0.0 | 0.4801 | 32.0% |

注意:50 题子集是**偏难的子集**(很多在 baseline 上也错),不要拿这两个数字对比 dev_small 上的 Brier ≈ 0.03。

### 完整 375 题 (`dcv_full_qids.txt`)

| 文件 | 配置 | Brier | Acc |
|---|---|---|---|
| `dcv_full_v1.jsonl` | v1 prompts, verifier_temp=0.7 | 0.2613 | 60.27% |
| `dcv_full_v2.jsonl` | v2 prompts, verifier_temp=0.0 | 0.2685 | 62.40% |
| **Baseline: CW+BM25 S2** (同 375 qids,来自 `20260413_215746.jsonl`) | — | **0.1783** | **77.87%** |

**结论:DCV 目前明显劣于 CW+BM25 S2 baseline**(Brier 高 +0.08,Acc 低 ~17 pp)。v2 比 v1 略好 (Acc +2pp) 但仍远不及 baseline。

---

## 下一步可以做什么

按优先级排:

1. **诊断 DCV 为什么差**(必做):
   - 抽样 30 道 DCV 错而 baseline 对的题,人工/再用一个 LLM 判 sub-claim 是不是合理。重点查:
     - 分解质量:sub-claim 是不是真原子、真独立?
     - 验证质量:`rationale` 里引的事实是否真的存在于证据里?`p_true` 方向是否和 `rationale` 一致?
     - 组合质量:logit pool 是否被一两条极端 sub-claim 拖偏?
   - 直接的 detail jsonl:`dcv_full_v2_detail.jsonl`,每行一条 sub-claim 验证结果。
2. **修分解失败模式**:看到的典型坏 sub-claim 是"X 高于 Y"和"X 低于等于 Y"两条互补,即使 v2 prompt 已经明令禁止。可能要做后处理去重、或改成结构化输出 schema。
3. **可调实验**:
   - K 扫(2 / 3 / 5)。
   - verifier_temp(0 vs 0.7,目前结论不一致)。
   - 权重 strategy:目前用 LLM 自报权重 × `confidence+1`;可以试只用 confidence 或只用 weight。
   - 把 verifier 从 "全证据" 改成 BM25 召回的 top-N 证据(和 baseline 对齐)。
4. **DCV + 多 agent 杂交**:K 个 verifier 当成 S1 风格的独立 agent 跑;或把 DCV 的 `p_yes` 作为另一路并入 S2 deliberation。
5. **写进 README**:确认方向后,把 DCV 当作第 4 种 protocol 写进主 README 的方法学一节。

---

## 怎么复现

```bash
# 50-题原型 (v2 prompts)
python run_dcv_prototype.py \
  --qid-file data/dcv_test_qids.txt \
  --output data/results/dcv_proto_50_v2.jsonl \
  --prompt-version v2 --k 3 --temperature 0.0 \
  --api-key-env OPENROUTER_API_KEY

# 完整 375 题 (v2)
python run_dcv_prototype.py \
  --qid-file data/dcv_full_qids.txt \
  --output data/results/dcv_full_v2.jsonl \
  --prompt-version v2 --k 3 --temperature 0.0 \
  --api-key-env OPENROUTER_API_KEY
```

脚本会增量写 `--output`,中断重跑会自动 resume 已完成的 qid。Baseline 对比默认指向 `data/results/20260413_215746.jsonl` 的 CW+BM25 S2 行。

---

## 相关提交

```
9b4851f data: add DCV full 375-question runs (v1 + v2)
e2e1d24 data: add DCV 50-question prototype results (v1 + v2 prompts)
751125c data: add DCV smoke-test outputs
3baa248 data: add DCV qid selections (full 375 + test 50)
89b7dfb feat: add run_dcv_prototype.py runner with resume + baseline compare
75a826c feat: add DCV (Decomposer + Verifier) agent and log-pool combiner
```
