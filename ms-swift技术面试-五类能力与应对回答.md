# ms-swift 技术面试：五类能力与应对回答（LLM 后训练 & Agent）

> 配套文档：`ms-swift面试准备-LLM与Agent后训练.md`（原理速通 + 代码索引）。
> 本文目标：把同一个框架，按面试官考察的 **5 类能力** 重新组织成"可开口的回答"，重点不是复述概念，而是体现：
> ① 懂原理+局限+改进 ② 会验证 ③ 会排查 ④ 能落地 ⑤ 懂业务。
> 所有论点尽量回扣 ms-swift 真实代码（`文件:行号`），用"我在源码里看过"替代空谈。

---

## 总纲：面试官怎么用这 5 类能力筛选实习生

| 能力 | 面试官真正在听什么 | 实习生最容易翻车的地方 |
|---|---|---|
| 1. 底层原理 | 这个方法**解决什么问题**、**局限在哪**、**怎么改进** | 只背公式，说不清"为什么这么设计" |
| 2. 实验验证 | 你怎么**证明**它有效、实验细节是否经得起追问 | 只报结果，说不清 baseline/消融/指标 |
| 3. 问题定位 | 结果/性能不符预期时**怎么排查** | 只讲流程，不讲优化思路与排错路径 |
| 4. 工程落地 | 理论可行但工程**不可行**时怎么办、部署/稳定/回滚 | 只有 notebook 实验，没碰过生产链路 |
| 5. 业务场景 | 方案**适合什么场景**、成本/优先级/用户关心什么 | 脱离业务空谈 SOTA |

下面逐类给"面试题 → 回答骨架 → 追问应对"。

---

## 能力 1：底层原理深入理解（方法为何这么设计 + 局限 + 改进）

> 核心话术模板：**"这个方法解决的是 X 问题；它的假设/代价是 Y（局限）；因此有了 Z 改进（结合 ms-swift 实现）。"**

### 1.1 面试题：GRPO 为什么要这么设计？相比 PPO 解决了什么、又引入了什么新问题？

**回答骨架（建议 90 秒）：**
- **解决什么**：PPO 需要训练一个 value/critic 网络估计优势，参数量翻倍、训练不稳、对 value 的 GAE 超参敏感。GRPO 对同一 prompt 采样 G 个回答，用**组内相对优势** $\hat A=(R-\mu_G)/\sigma_G$ 当基线，直接省掉 critic（`docs/.../GRPO/GetStarted/GRPO.md` 目标函数）。实现见 `swift/rl_core/advantage.py:54-60`：`grouped = rewards.view(-1,K); group_mean = grouped.mean(dim=1)`。
- **引入的局限**：① 必须每个 prompt 采样 G 份，rollout 算力线性增长；② 组内 std=0 时优势全 0、梯度消失（DAPO 动态采样解决）；③ 句子级 loss 归一化带来长度偏差（DAPO/BNPO 的 token 级归一解决）；④ 依赖 on-policy，但用 vLLM 加速采样会破坏该假设（见能力 3/4 的 mismatch）。
- **改进脉络**：DAPO（Clip-Higher+动态采样+token 级 loss+overlong 过滤）、GSPO（序列级 IS 降方差，`grpo_trainer.py:994-1002`）、CISPO（裁剪 IS 权重本体保留梯度，`:1019-1021`）、SAPO（软门控替硬裁剪，`:1022-1028`）、RLOO（leave-one-out 无偏基线，`advantage.py:57`）。

**追问应对**：
- Q：组内 std=0 到底会发生什么、怎么发现？
  A：优势分子为 0 → `advantages` 全 0 → `per_token_loss` 恒 0 → 该 prompt 不产生梯度。监控靠 `compute_reward_metrics` 的 `frac_reward_zero_std`（`advantage.py:320`），比例高说明 reward 设计太简单/数据太易，需要动态采样或换更难数据。
- Q：GRPO 真的不需要 value model 吗？那 KL 怎么算？
  A：KL 用 K3 估计 `exp(ref-cur)-(ref-cur)-1`（`grpo_trainer.py:958-959`），只需一次额外 forward 拿 `ref_per_token_logps`，不训练 ref 模型，代价是多一份 ref 推理显存（可 offload/`beta=0` 关掉）。

### 1.2 面试题：GRPO 家族里 CISPO / GSPO / SAPO 各自针对什么痛点？

**回答骨架（逐点对应"问题→方法"）：**
- **GSPO（序列级 IS）**：token 级 IS 每 token 只采样一次，无法真正修正分布、方差高易崩（MoE 路由异质更明显）。改为序列级 $w_i=\exp(\frac1{|y|}\sum_t\log\frac{\pi_\theta}{\pi_{\text{old}}})$，与目标（序列级 reward）单位一致（`GSPO.md`、`grpo_trainer.py:994-1002`）。
- **CISPO（保留关键 token 梯度）**：GRPO 硬裁剪 `min(r, clip)·A` 会把低概率但关键（"Wait/Aha"）token 的梯度直接丢掉的。CISPO 改为 `detach(min(r, ε_high)) · A · log π_θ`，裁剪的是权重、梯度始终来自 $\log\pi_\theta$（`CISPO.md`、`:1019-1021`）。`epsilon_high` 取大（5.0）。
- **SAPO（平滑离群）**：硬裁剪两难——太严丢有效样本、太松引入噪声梯度。改用温度 sigmoid 软门控 $g_t=\sigma(\tau(r_t-1))·4/\tau$ 平滑衰减（`:1022-1028`）。

**追问应对**：
- Q：GSPO-token 和 GSPO 什么关系？
  A：当所有 token 优势相同（GRPO 现状），二者梯度等价；GSPO-token 只是把序列权重 stop-gradient 后乘回逐 token 概率，预留未来细粒度 token 优势空间（`GSPO.md` 注记 + `grpo_trainer.py:1000-1002`）。

### 1.3 面试题：DPO 为什么会有长度偏好？SimPO/ORPO/KTO 怎么解？

**回答骨架**：
- DPO 隐式奖励是 $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，长回复累积 logp 更大 → 被偏好。
- **SimPO**：用序列平均 logp 作隐式奖励 + length 归一化 + target margin γ，省 reference（ms-swift `dpo_trainer.py` 支持 `loss_type`）。
- **ORPO**：在 SFT loss 上加 odds-ratio 偏好项，单一阶段、无 reference。
- **KTO**：只需 $(x,y,\text{label}\in\{0,1\})$，人类"痛苦/快乐"不对称损失，适配只有好坏标注的数据（`RLHF.md`）。
- **局限**：reference-free 方法对数据噪声更敏感；DPO 仍需先 SFT 再训以防分布偏移。

### 1.4 能力 1 的反向展示话术
> "我不只会写 GRPO 公式。我能讲清它相比 PPO 省掉 critic 的代价是 rollout 算力翻倍和 on-policy 脆弱性；并能顺着 ms-swift 的 `advantage.py` / `grpo_trainer.py` 说清 DAPO/GSPO/CISPO/SAPO 各自补的是哪块短板。"

---

## 能力 2：实验和方案验证能力（怎么证明有效）

> 核心话术：**"我的验证设计是：明确任务+指标 → 选 baseline/消融 → 控制变量 → 看哪些监控量 → 得出结论。"** 面试官追问细节时，用 ms-swift 的**日志指标**和**参数**作答最稳。

### 2.1 面试题：你怎么验证"换成 GRPO 比 SFT 好 / DAPO 比原始 GRPO 好"？

**回答骨架（以 GRPO 后训练数学推理为例）：**
- **任务与指标**：用 GSM8K/MATH 做 `num_generations=8` 的 GRPO；主指标是评测集准确率（ms-swift 接 EvalScope，`docs/.../Evaluation.md`）；RL 过程指标看 `reward_mean`、`reward_std`、`frac_reward_zero_std`、`clip_ratio`、`kl`。
- **Baseline**：同数据同 seed 的 SFT 模型作对照；DAPO 对比时固定 `num_generations`、学习率、rollout 引擎，只改 `loss_type / epsilon_high / dynamic_sample / overlong_filter`。
- **消融**：① 关掉动态采样看 `frac_reward_zero_std` 上升、有效步数下降；② 关掉 Clip-Higher 看长尾探索 token 的 `clip_ratio` 分布；③ `beta=0` vs `beta=0.0x` 看 `kl` 与泛化。
- **结论判据**：不只看终局 accuracy，还要看 reward 曲线是否稳定上升、KL 是否可控、有无 reward hacking（reward 涨但评测不涨）。

**追问应对**：
- Q：你怎么知道提升不是随机种子/数据顺序带来的？
  A：固定 seed + 多次复现；或至少 reports 区间；并做消融隔离单变量。
- Q：reward 一直在涨但评测 accuracy 不涨，说明什么？
  A：典型 **reward hacking / 过拟合奖励函数**。要检查奖励是否只测格式、是否可被捷径骗分；ms-swift 支持多奖励函数 + `reward_weights`，应加入"真实正确性"奖励而非仅格式奖励。
- Q：`kl` 突然飙升说明什么？
  A：策略严重偏离 reference，可能 `beta` 太小 / 学习率太大 / 某批 reward 噪声大；应降 lr 或升 `beta`，或开 off-policy sequence masking（`grpo_trainer.py:1052`）。

### 2.2 面试题：你做过多奖励函数的实验吗？尺度不一怎么验证？

**回答骨架**：
- 多奖励（如"正确性 + 格式 + 长度惩罚"）用 `reward_weights` 加权；问题在于各奖励量纲不同。
- ms-swift 提供 **GDPO**（`advantage.py:73-84`）：对每个奖励函数先组内标准化再加权求和、整体再标准化，缓解尺度不一致。
- 验证：对比 GDPO vs 直接加权，看训练曲线是否更稳、`per_func_mean/std`（`compute_reward_metrics` 输出）是否各自有意义而非被某一函数主导。

### 2.3 能力 2 的反向展示话术
> "验证一个 RL 后训练改动，我不只看终局 benchmark。我会同时盯 ms-swift 日志里的 `reward_mean/std`、`frac_reward_zero_std`、`clip_ratio`、`kl`、以及 mismatch 的 `ESS/χ²`，用这些过程量判断改动是真有效还是 reward hacking。"

---

## 能力 3：问题定位能力（结果/性能不符预期怎么排查）

> 核心话术：**"先定位现象属于哪一类（结果错 / 慢 / 崩），再按'数据→奖励→训练配置→系统'分层排查，最后给可验证的假设。"**

### 3.1 面试题：GRPO 训练后模型能力反而下降了，怎么排查？

**分层排查清单（开口就能列）：**
1. **数据/奖励层**：`frac_reward_zero_std` 是否过高（多数 prompt 组内无区分 → 几乎无梯度）；reward 函数是否只判格式导致 **reward hacking**（`reward_mean` 涨但评测跌）。→ 查 `swift/rl_core/grpo_algorithm.py:77` 的 NaN 告警、`:320` 的 zero_std 比例。
2. **训练配置层**：`beta` 是否过小导致偏离 reference（`kl` 飙升）；`epsilon` 是否过大致更新过猛；`loss_type` 是否引发长度偏差（默认 `grpo` 句子级归一让模型变短变傻）→ 换 `bnpo/dapo`。
3. **数值/稳定性层**：`clip_ratio` 长期 100% 说明裁剪全程触发 → 学习率/eps 不合理；IS 层级 token 级方差大 → 试 GSPO（`grpo_trainer.py:992`）。
4. **系统层**：训练-推理 mismatch 使样本 off-policy（见 3.3）→ 看 `rollout_correction/` 的 `ESS`、`χ²`、`k3_kl`。

### 3.2 面试题：GRPO 训练突然变得非常慢，怎么定位？

**回答骨架（系统向）：**
- **Rollout 是瓶颈**：GRPO 速度主要受采样限制。检查是否用了 vLLM/SGLang 加速（`--infer_backend vllm`）；colocate 模式推理被训练占显存拖累 → 换 server 模式 + async engine（`multi_turn.md` 图示，减少多轮气泡）。
- **动态采样死循环**：`dynamic_sample=true` 且数据太难/奖励太稀疏 → `max_resample_times` 内凑不满 batch，疯狂重采样（`grpo_algorithm.py:124` `compute_std_for_dynamic_sampling`）。→ 调 `max_resample_times` 或放宽奖励。
- **序列过长**：未开 `overlong_filter` 导致大量截断样本仍参与计算 + 长序列 attention 爆炸。→ 开 `overlong_filter` + `soft_overlong` 惩罚。
- **并行/显存**：未用序列并行训长文本 → 单卡 OOM 触发频繁 reclaim；未用 packing 导致短样本 padding 浪费。

### 3.3 面试题：结果和预期不一致——训练-推理不一致（off-policy）怎么发现与修？

**回答骨架（硬核点，直接套 ms-swift）：**
- **现象**：loss 不降 / 训着训着崩 / `kl` 异常。根因：vLLM 与训练模型算子差异使 $\pi_{\text{vLLM}}\ne\pi_\theta$，违反 on-policy（`training_inference_mismatch.md`）。
- **定位**：开 `--log_rollout_offpolicy_metrics true`，看 `rollout_correction/` 下 `k3_kl`、`chi2_token/seq`、`is_weight_mean`（理想 1.0）、`ESS`（越近 1 越 on-policy）。ESS 远低于 1 即严重 off-policy。
- **修复**：① IS 修正 `--rollout_importance_sampling_mode token_truncate/sequence_mask` 等（代码 `grpo_trainer.py:1049` 把 `rollout_is_weights` 乘进 loss）；② Off-Policy Sequence Masking `--off_policy_sequence_mask_delta`（`:1052`，负优势+策略偏移大时直接 mask）；③ 提高权重同步频率。

### 3.4 能力 3 的反向展示话术
> "排查我不靠拍脑袋。我会先按数据→奖励→训练配置→系统四层定位，并用 ms-swift 的 `frac_reward_zero_std` / `clip_ratio` / `kl` / `ESS` / `χ²` 这些过程指标把'感觉不对'变成可观测、可验证的假设。"

---

## 能力 4：工程落地能力（理论可行≠工程可行）

> 核心话术：**"落地时要考虑显存/算力/稳定性/回滚监控；很多论文 trick 在工业框架里要配合 colocate/server、量化、序列并行才跑得动。"**

### 4.1 面试题：论文里的 GRPO 很美好，工业界落地有什么坑？

**回答骨架：**
- **显存**：GRPO 需同时驻留 训练模型 + ref 模型 +（server 模式）rollout 引擎。7B 可用 QLoRA + 把 ref 卸载；72B 需 4×80G + vLLM 张量并行 + LoRA（`examples/train/grpo/internal`）。
- **吞吐**：rollout 占大头 → 必须 vLLM/SGLang + async engine；多轮 agent 用 `AsyncEngine` 降气泡（`multi_turn.md`）。
- **稳定性**：训练-推理 mismatch（3.3）、reward hacking、KL 爆炸 → 用 mismatch 监控 + off-policy masking + `beta` 兜底。
- **可复现/回滚**：训练用固定 seed、定期存 checkpoint；评测接 EvalScope 自动回归，能力回退能定位到哪次迭代。
- **部署**：训完 `swift export` 合并 LoRA / 量化（AWQ/GPTQ/FP8）再用 vLLM 部署（`Export-and-push.md`）。

### 4.2 面试题：模型上线后能力突然下降，怎么保证可回滚与监控？

**回答骨架（结合生产链路）：**
- **监控**：线上持续用 EvalScope 跑核心探针集 + 业务指标（如工具调用成功率、回答准确率），波动超阈值告警。
- **回滚**：保留多版本 checkpoint 与评测报告，能力下降立即回退上一稳定版；ms-swift 训练产出可直接 `swift deploy` 多版本灰度。
- **根因**：下降可能来自数据漂移（用户问题分布变）、奖励函数过期、或新模型引入 regression；用 A/B + 离线复现定位。

### 4.3 面试题：要把一个 Agent 能力从实验落到生产，工程上要注意什么？

**回答骨架：**
- **工具返回不进 loss**：`tool_response` mask 掉（`Agent-support.md:18`），否则模型学去"背"环境输出。
- **多轮稳定性**：scheduler 的 `check_finished`/`step` hook（`multi_turn.py`）管理终止与上下文；长轨迹需考虑上下文压缩（自定义 scheduler 改 history）。
- **异步与并发**：server 模式 + async engine 支撑高 QPS 多轮交互。
- **失败兜底**：工具调用 JSON 非法要有格式奖励 + 解析兜底；环境异常（timeout）要在 `rollout_infos` 记 reward 并隔离。

### 4.4 能力 4 的反向展示话术
> "我认为落地第一优先是'跑得动且稳'：显存靠 QLoRA/序列并行，吞吐靠 vLLM+async，稳定靠 mismatch 监控和 off-policy masking，可回滚靠 checkpoint+EvalScope 自动回归。论文 trick 必须嵌进这条工程链才有价值。"

---

## 能力 5：业务与实际场景理解（方案为谁、成本、优先级）

> 核心话术：**"先定义用户/业务目标与可量化价值，再谈适用场景与成本，最后说资源有限时先优化哪块。"**

### 5.1 面试题：GRPO 后训练适合什么业务场景？不适合什么？

**回答骨架（分场景）：**
- **适合**：有明确**可验证奖励**的任务——数学/代码（单测通过）、SQL/工具调用（执行成功）、格式合规、Agent 多轮（环境给终局分）。这类规则奖励便宜、无奖励模型成本。
- **不适合 / 谨慎**：开放式创作、主观问答——没有可靠奖励，只能上 reward model / GenRM（`rm_plugin.py:41`），但奖励模型本身有偏差且贵，易 reward hacking。
- **成本**：GRPO rollout 算力是同数据 SFT 的 G 倍（G=`num_generations`）；需 vLLM 集群。小团队应先评估"奖励获取成本 vs 收益"。

### 5.2 面试题：如果资源有限，应该先优化哪部分？

**回答骨架（优先级排序，体现取舍）：**
1. **先保数据/奖励质量**（最高杠杆）：垃圾奖励 → 一切白做。先确保 reward 真测"对不对"而非"像不像"。
2. **再保训练稳定**（防崩）：开 `dynamic_sample` 消灭零梯度样本、合理 `beta`、监控 `kl`/`clip_ratio`。
3. **最后才堆算力/算法花活**：DAPO/CISPO 等是在前两者稳了之后再榨收益。
4. **部署侧**：QLoRA + 量化部署先把"能上线"跑通，再谈更大模型。

### 5.3 面试题：用户/业务方真正关心什么？上线成本有多高？

**回答骨架：**
- 业务方关心：**效果（任务成功率↑）、成本（每次推理￥）、延迟（响应秒数）、稳定（不崩不退化）**，而非用了哪个 RL 算法。
- 上线成本构成：① 训练算力（G 倍 rollout）；② 奖励工程（规则便宜、RM/GenRM 贵）；③ 推理部署（vLLM 显存）；④ 持续评测与监控人力。
- 建议：MVP 用规则奖励 + 小模型 GRPO 验证业务指标提升，再决定是否上 RM/大模型。

### 5.4 能力 5 的反向展示话术
> "我不会一上来就堆 GRPO。先问业务有没有可验证的奖励信号、用户最在乎成功率还是延迟还是成本；资源有限时先保奖励质量和训练稳定这两个最高杠杆点，算法花活放最后。实验环境跑通≠业务有用——要用业务指标（调用成功率/成本/满意度）而非仅 benchmark 来证明价值。"

---

## 综合演练：一个"端到端"项目讲述模板（能力 1~5 全覆盖）

面试常让你讲一个项目。用下面骨架，把 ms-swift 当你的"工具箱"：

> **背景**（能力5）：业务要提升客服 Agent 的工具调用成功率。
> **方案**（能力1）：用 ms-swift 的 Agent 模板（`Agent-support.md`）统一数据格式，GRPO 做多轮 RL；奖励 = 工具调用格式正确 + 最终回答解决用户问题（规则奖励，零 RM 成本）。
> **验证**（能力2）：GSM8K 式内部集 + 线上探针集双指标；与 SFT baseline 对比；消融 `dynamic_sample`/`overlong_filter`；盯 `reward_mean`、`frac_reward_zero_std`、`kl`。
> **踩坑**（能力3）：训练初期 `frac_reward_zero_std` 高 → 开动态采样；上线 vLLM 后 `ESS` 低 → 开 `rollout_importance_sampling_mode` 修正。
> **落地**（能力4）：QLoRA 训 7B + 量化部署；EvalScope 持续监控；保留 checkpoint 可回滚。
> **取舍**（能力5）：资源有限先保奖励质量与稳定，暂不上 GenRM；用业务成功率而非仅准确率证明价值。

---

## 附：被追问时最该脱口而出的"监控指标 → 含义 → 动作"速查

| 指标（ms-swift 日志） | 含义 | 异常时的动作 |
|---|---|---|
| `reward_mean` / `reward_std` | 奖励水平与区分度 | 涨但评测不涨=reward hacking |
| `frac_reward_zero_std` | 组内无区分样本比例 | 高→动态采样/换难数据 |
| `clip_ratio` | 策略更新被裁剪比例 | 长期 100%→lr/eps 不合理 |
| `kl` | 偏离 reference 程度 | 飙升→升 `beta`/降 lr |
| `rollout_correction/k3_kl` | 训练-推理分布差 | 大→IS 修正/on-policy mask |
| `rollout_correction/ESS` | 有效样本率（近1=on-policy） | 低→off-policy 修正 |
| `rollout_correction/chi2_*` | IS 权重方差 | 大→权重不稳，需截断/掩码 |

> 一句话收尾：面试不是考你背了多少算法，而是考你**能不能把"方法—问题—证据—落地"串成一条可验证的链路**。ms-swift 的代码与日志就是这条链上最硬的证据。
