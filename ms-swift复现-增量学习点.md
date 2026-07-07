# ms-swift 复现：相对前两轮的新学习点（8×4090 全流程视角）

> 本文记录：在把 ms-swift **真正落到 8×4090 上跑通全流程**这一轮（plan 文档 `ms-swift复现-Plan-8x4090.md`）中，
> 相比 **第一轮（原理速通 + 代码索引）** 和 **第二轮（五类面试能力应对）** 增量学到的东西。
> 一句话总结增量：**前两轮是"读懂"和"能讲"，这一轮是"跑得动"和"算得清账"——把算法翻译成硬件约束与可复跑命令。**

---

## 0. 三轮各自补了什么（定位）

| 轮次 | 关注 | 产出 | 缺什么 |
|---|---|---|---|
| 第一轮 | 底层原理 + 代码索引 | 原理文档、文件:行号 | 只在"读"，没"跑"；不知道在自己机器上能不能跑 |
| 第二轮 | 面试五类能力 | 回答骨架、排查/落地话术 | 仍是纸面应对，没碰过真实显存/吞吐/报错 |
| **第三轮** | **资源约束下的全流程复现** | **plan + 本增量文档** | —— 把理论变成"24GB 单卡上限 / 192GB 总量 / 无 NVLink"下的具体决策 |

**本质增量 = 工程可行性翻译**：把"GRPO/DAPO/序列并行"从论文概念，翻译成"8×4090 上该用几个卡、什么模式、哪个模型、哪些参数"。

---

## 1. 新增认知 1：硬件拓扑直接决定算法/并行选型（前两轮完全没提）

- **新学**：8×4090 是 **PCIe 互联、无 NVLink**。这意味着：
  - **TP（张量并行）跨卡很慢** → 实战优先 **DP / ZeRO2/3 / PP / 序列并行**，不要开大 TP。
  - 这是"能跑"和"跑得动"的分水岭；前两轮讲 Megatron TP/PP/EP 时没强调"消费卡无 NVLink 下 TP 不划算"。
- **新学**：Megatron 在 4090 上**基本不实用**（无 NVLink，TP 通信成瓶颈）→ 改为"读 `swift/megatron` 代码 + 看文档"即可，不强求跑。这是第一轮列 Megatron 支持时未做的"落地取舍"。
- **对照**：前两轮把"序列并行/Ulysses/Ring"当特性列出；这一轮明确它= **长文本在 4090 上不 OOM 的必需手段**（`--sequence_parallel_size N`）。

## 2. 新增认知 2：显存账要算到"单卡 24GB"这个硬上限

- **新学**：模型可行性矩阵（bf16）：
  - 0.5B/1.5B/3B：单卡或 2 卡轻松；
  - **7B：单卡放不下全参训练余量**，需 2–4 卡 ZeRO 或 **QLoRA 单卡**；
  - 14B：4–8 卡 ZeRO3（紧）；
  - **32B/72B：192GB 总显存也跑不了全参训练**，只能 QLoRA 推理级 → 官方 `vllm_72b_4gpu.sh`（4×80G）**在 4090 上不能直接等价复现**。
- **新学**：前两轮讲 QLoRA 只是"4-bit 省显存"；这一轮它是**让 7B GRPO 在 24GB 单卡/少卡上可行的关键杠杆**（`--tuner_type qlora`）。
- **新学**：colocate vs server 的选择被单卡 24GB 钉死——7B 全参 colocate 易 OOM，**必须用 server 模式把卡拆成"训练卡 + rollout 卡"**。

## 3. 新增认知 3：server 模式拆卡是 GRPO 在 4090 上的标准拓扑

- **新学**：官方 `examples/train/grpo/external/grpo_7b.sh` 直接给出 **6 卡训练 + 2 卡 vLLM rollout server** 的拓扑（注释写明 `8 * 80G，6 for training, 2 for rollout`）。这一轮把它**平移到 8×4090**：
  - 终端 A：`CUDA_VISIBLE_DEVICES=6,7 swift rollout --model ... --data_parallel_size 2 --vllm_server_port 8000`
  - 终端 B：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC_PER_NODE=6 swift rlhf --rlhf_type grpo ... --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port 8000`
- **新学**：多轮 Agent GRPO 也需要**两个终端**（rollout server + trainer）——前两轮讲 `MultiTurnScheduler` 的 hook 逻辑，但没说"实操上它是跨进程 server 调用"。
- **对照**：第一轮讲"colocate/server 两种模式"只是概念；这一轮变成了"我的卡怎么分"的具体决策。

## 4. 新增认知 4：真正的 CLI  fluency（前两轮没给过一条可运行命令）

- **新学**：ms-swift 的统一入口族（都在 `swift/cli/`）：
  `swift sft` / `swift rlhf`（`--rlhf_type grpo|dpo|kto|ppo|...`）/ `swift infer` / `swift eval` / `swift export` / `swift deploy` / `swift rollout` / `swift sample` / `swift pt`。
- **新学**：多卡用 `NPROC_PER_NODE=N` + `CUDA_VISIBLE_DEVICES` 驱动 DeepSpeed，不是 `torchrun` 手写。
- **新学**：GRPO 变体不是改代码，而是**改参数即可热切换**（这把第一轮"家族对比"变成可操作实验）：
  - DAPO：`--loss_type dapo --epsilon_high 0.28 --dynamic_sample true --overlong_filter true`
  - GSPO：`--importance_sampling_level sequence`
  - CISPO：`--loss_type cispo --epsilon_high 5.0`
  - SAPO：`--loss_type sapo`
  - RLOO：`--advantage_estimator rloo --ki_in_reward true`
  - mismatch 修正：`--rollout_importance_sampling_mode token_truncate`
- **对照**：前两轮这些只作为"概念/代码位置"出现；这一轮是"我下哪条命令就能看到效果差异"。

## 5. 新增认知 5：内置数据集让"复现"跳过数据工程

- **新学**：ms-swift 内置 150+ 数据集，可直接 `#采样数` 取子集，无需自己造数据：
  - 数学 GRPO：`AI-MO/NuminaMath-TIR#1000`（自带 `solution` 列供 accuracy 奖励）
  - Agent：`AI-ModelScope/function-calling-chatml`、`AI-ModelScope/alpaca-gpt4-data-zh`
  - DPO：`hjh0119/shareAI-Llama3-DPO-zh-en-emoji`
- **新学**：第一轮讲 Agent 数据格式（tools/tool_call/tool_response JSONL）是"原理"；这一轮发现**直接喂现成 function-calling 数据集 + `--agent_template hermes` 就能训**，数据格式由模板自动映射——理论里的"解耦"变成了"开箱即用"。

## 6. 新增认知 6：监控指标从"名词"变成"我要盯的曲线"

- **新学**：前两轮把 `frac_reward_zero_std`/`clip_ratio`/`kl`/`ESS`/`χ²`/`k3_kl` 当"面试能列出来加分"；这一轮换成了**跑训练时 TensorBoard 里真正要盯的过程量**：
  - `reward_mean` 不涨但评测不涨 → reward hacking；
  - `frac_reward_zero_std` 高 → 开动态采样/换难数据；
  - 开 `--rollout_importance_sampling_mode` 后看 `rollout_correction/ESS`（越近 1 越 on-policy）、`k3_kl`、`chi2_*`。
- **新学**：这把第二轮"能力3 问题定位"的排查清单，落成了**每天看日志的具体动作**。

## 7. 新增认知 7：工程杠杆（flash-attn/liger-kernel/packing）从"特性列表"变"必开项"

- **新学**：在 24GB 单卡上，`--use_liger_kernel true --attn_impl flash_attn --packing true` 不是可选项，而是**让 3B 长上下文 SFT/7B 训练跑得动、跑得快的必开项**（见 `examples/train/agent/qwen2_5.sh` 实测 ~35GB 单卡 3B）。
- **新学**：`--packing true` 对短样本多的 SFT 提升利用率显著——前两轮提过 packing 但没说"4090 这种小显存卡上它是吞吐关键"。

## 8. 新增认知 8：资源约束逼你做"优先级"，恰好呼应第二轮能力5

- **新学**：8×4090 不能啥都跑（72B 放弃、Megatron 不跑、TP 不开），所以复现必须**先跑通最小闭环（0.5B SFT → 3B GRPO → 7B DPO），再上变体**。
- **新学**：这和第二轮"能力5：资源有限先优化哪块"完全一致——**先保奖励质量与训练稳定（小模型跑通+动态采样+合理 beta），算法花活（DAPO/CISPO）放最后**。理论（第二轮）和实操（第三轮）在此闭环印证。
- **新学**："完整流水线"价值 > 单个算法 SOTA：只有 train→eval→export→deploy 全跑通，才算真复现；这也回答了第二轮"实验环境跑通≠业务有用"——至少先证明工程链路可端到端闭环。

## 9. 新增认知 9：踩坑类型是前两轮没覆盖的"环境/部署级"

- **新学**：会遇到的真问题（plan 文档第 10 节）：
  - colocate 7B OOM → 改 server 拆分 / QLoRA / 降 `max_completion_length`；
  - rollout server 连不上 → 核对 `host/port`、先单卡自测；
  - 4090 上 vLLM 与 transformers 版本冲突 → 锁 `requirements/` 版本；
  - 多卡 ZeRO 因无 NVLink 偏慢 → 用 ZeRO2+DP+packing 而非大 TP。
- **对照**：前两轮"问题定位"聚焦算法/指标层；这一轮补了**环境/显存/版本/进程通信**这一整层工程排错。

## 10. 新增认知 10：学习方法本身也升级了

- **新学**：第一轮是"先读源码再理解"；这一轮是"**先跑通最小命令 → 出错 → 回读对应源码/文档 → 记一条增量点**"的闭环，效率更高。
- **新学**：官方 `examples/` 目录是**最快 on-ramp**——`grpo_7b.sh` 直接给了 6+2 拆卡模板，比从零写命令可靠得多。
- **新学**：可编辑安装 `pip install -e .` 让你边跑边改边读 `swift/` 源码，是把"读过的代码"和"跑过的实验"缝合的关键动作。

---

## 附：三轮能力递进一句话

- **第一轮**：我能讲清 ms-swift 里 GRPO/DAPO/GSPO/CISPO/Agent 训练的原理和代码位置。
- **第二轮**：我能在面试里，按"原理/验证/排查/落地/业务"五类能力，把上面那些讲成有深度的回答。
- **第三轮**：我能在自己的 8×4090 上，把上面那些**真正跑成一条可复现、可监控、可部署的全流程**，并算清显存账与卡分配。

> 增量核心：**从"懂"到"会跑"，从"讲得清"到"算得清（显存/算力/拓扑）"**。前两轮是认知资产，这一轮是把资产变成可演示的工程能力——而这正是实习面试里"有没有真做过项目"的分水岭。
