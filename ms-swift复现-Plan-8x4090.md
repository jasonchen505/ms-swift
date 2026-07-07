# ms-swift 全流程复现 Plan（基于 8×RTX 4090 实际算力）

> 目标：在 **8 张 RTX 4090（24GB/卡，共 192GB 显存，无 NVLink）** 上，把 ms-swift 的
> "预训练/微调 → 偏好对齐 → GRPO 强化学习 → Agent 多轮 → 推理/评测/部署"
> 完整跑通一遍，并以此为主线**边复现边学透这个项目**。
> 配套文档：
> - `ms-swift面试准备-LLM与Agent后训练.md`（原理 + 代码索引）
> - `ms-swift技术面试-五类能力与应对回答.md`（五类能力应对）
> - `ms-swift复现-增量学习点.md`（本 plan 执行中相对前两轮的新学点）

---

## 0. 算力评估：8×4090 能做什么、不能做什么

### 0.1 硬件画像（决定所有工程决策）

| 项 | 规格 | 对训练的影响 |
|---|---|---|
| 单卡显存 | 24 GB GDDR6X | 单卡放不下 7B 全参训练的余量，需要多卡/量化 |
| 总显存 | 8×24 = 192 GB | 远大于 7B 需求，可做 7B 全参+rollout 拆分 |
| 互联 | PCIe 4.0 x16（无 NVLink） | **TP（张量并行）跨卡很慢**，优先 DP / ZeRO / PP / 序列并行 |
| 算力 | ~165 TFLOPS bf16（Ada Lovelace） | 足够小模型全流水线；支持 FP8（推理量化可选） |
| 卡数 | 8（单机） | 无需多机通信，规避 IB/网络复杂度 |

### 0.2 模型尺寸可行性矩阵（bf16，含优化器/梯度余量）

| 模型 | 参数显存(bf16) | 全参 SFT | GRPO(需 rollout) | 推荐方案 |
|---|---|---|---|---|
| 0.5B（Qwen2.5-0.5B） | ~1 GB | 1 卡轻松 | 1 卡 colocate | 入门练手 |
| 1.5B | ~3 GB | 1 卡 | 1–2 卡 | 快速迭代 |
| 3B | ~6 GB | 1–2 卡 | 6 训+2 rollout | **GRPO 主力** |
| 7B | ~14 GB | 2–4 卡 ZeRO2/3 | QLoRA 或 6+2 拆分 | **SFT/DPO 主力，GRPO 用 QLoRA** |
| 14B | ~28 GB | 4–8 卡 ZeRO3 | QLoRA + rollout | 进阶（慢） |
| 32B/72B | 64/144 GB | 不可全参 | QLoRA 仅推理级 | **不建议做"完整流水线"复现**，列为 stretch |

> 结论：**完整流水线复现以 0.5B/3B 做 GRPO 全参练手、7B 做 SFT/DPO 与 QLoRA-GRPO** 最稳；
> 72B 那种 `examples/train/grpo/internal/vllm_72b_4gpu.sh`（4×80G）在 4090 上**直接等价复现不了**，需用 QLoRA 降配或放弃。

### 0.3 关键工程约束（来自源码/示例）

- ms-swift GRPO 两种 rollout 模式（见 `docs/.../GRPO/GetStarted/GRPO.md`）：
  - **colocate**：训练/推理同卡，省权重同步但推理被训练挤占 → 4090 上 7B 全参 colocate 易 OOM，**小模型（≤3B）或 QLoRA 才适合**。
  - **server**：vLLM 独立 server，训练卡远程采样 → 适合 8 卡**拆分成"训练卡 + rollout 卡"**（官方 `grpo_7b.sh` 即 6 训 + 2 rollout）。
- 4090 上 **server 模式拆分更稳**（避免 colocate 显存打架），下文默认 server 模式。
- 无 NVLink → 多卡用 **DeepSpeed ZeRO2/3** 或 **DP**，不要开大 TP。

---

## 1. 总览：六阶段里程碑

```
阶段0  环境          装 ms-swift + vLLM + DeepSpeed + flash-attn + liger-kernel（1 天）
阶段1  SFT           0.5B/3B 全参 + 7B LoRA，打通数据/模板/损失掩码（1–2 天）
阶段2  偏好对齐      7B DPO / KTO LoRA，理解 reward-model-free 路线（1 天）
阶段3  GRPO          3B 全参 / 7B QLoRA 数学 RL，server 模式拆分（2–3 天）★核心
阶段4  Agent 多轮    3B/7B 工具调用 SFT + 多轮 GRPO（2 天）
阶段5  推理/评测/部署 infer + EvalScope + vLLM deploy + export 量化（1 天）
阶段6  进阶(可选)    序列并行长文本 / Qwen3 小 MoE / Megatron 浅尝（时间允许）
```

每个阶段都对应之前文档里的原理点，边跑边回扣代码（`rl_core/advantage.py`、`grpo_trainer.py`、`Agent-support.md` 等）。

---

## 2. 阶段 0：环境与 sanity check

```bash
# 1) 安装（建议可编辑安装，方便读源码）
cd /data/home/yizhou/ms-swift
pip install -e . -U
pip install vllm deepspeed flash-attn liger-kernel tensorboard

# 2) 单卡 smoke test：确认算力可见、flash-attn 可用
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"

# 3) 跑通最小 SFT，验证整条链路（0.5B，1 卡，几分钟）
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --max_length 1024 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 --output_dir output/sft_05b
```
**学习目标**：确认 `swift sft` CLI、`--dataset` 内置数据集机制、`tuner_type`、输出目录结构。

---

## 3. 阶段 1：SFT（打通数据/模板/损失掩码）

### 3.1 3B 全参 SFT（2 卡 ZeRO2，验证多卡）

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift sft \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh#2000 \
  --tuner_type full --torch_dtype bfloat16 \
  --deepspeed zero2 --use_liger_kernel true --attn_impl flash_attn \
  --packing true --max_length 4096 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --num_train_epochs 1 --output_dir output/sft_3b
```
> 7B 全参把卡数提到 4（`CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4`），其余同上。

### 3.2 Agent 模板 SFT（呼应面试文档第 4 节）

直接复用官方 `examples/train/agent/qwen2_5.sh` 思路（实测 ~35GB，单卡 4090 可跑 3B）：
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen2.5-3B \
  --tuner_type full --agent_template hermes \
  --dataset AI-ModelScope/function-calling-chatml \
  --torch_dtype bfloat16 --max_length 8192 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --num_train_epochs 2 --packing true --use_liger_kernel true --attn_impl flash_attn \
  --output_dir output/agent_sft_3b
```
**学习目标**：
- `agent_template hermes` 如何把 `tools` + `tool_call`/`tool_response` 转成标准 messages（`Agent-support.md`）；
- `tool_response` 不进 loss（labels=-100）；
- `packing` 提升短样本利用率、`use_liger_kernel` 省显存。

---

## 4. 阶段 2：偏好对齐（DPO / KTO，7B LoRA 单卡）

复用官方 `examples/train/rlhf/dpo/lora.sh`（标注 24GiB，正好 1 张 4090）：
```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
  --rlhf_type dpo --model Qwen/Qwen2.5-7B-Instruct \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
  --torch_dtype bfloat16 --max_length 2048 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_train_epochs 1 --rpo_alpha 0.1 \
  --output_dir output/dpo_7b
```
**学习目标**：`swift rlhf --rlhf_type dpo` 入口；对比 SFT 多了 chosen/rejected 数据要求；`rpo_alpha` 混入 SFT loss 提稳（面试文档 2.1）。
可选做 KTO（`--rlhf_type kto`，只需 $(x,y,\text{label})$）体会 reference-free 路线。

---

## 5. 阶段 3：GRPO 强化学习（核心，server 模式拆分）

### 5.1 3B 全参 GRPO（6 训 + 2 rollout，对齐官方 7B 示例的拆分思路）

**终端 A — 启动 vLLM rollout server（2 卡）：**
```bash
CUDA_VISIBLE_DEVICES=6,7 swift rollout \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data_parallel_size 2 --vllm_server_port 8000
```
**终端 B — GRPO 训练（6 卡 ZeRO2）：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC_PER_NODE=6 swift rlhf \
  --rlhf_type grpo --model Qwen/Qwen2.5-3B-Instruct \
  --reward_funcs accuracy \
  --use_vllm true --vllm_mode server \
  --vllm_server_host 127.0.0.1 --vllm_server_port 8000 \
  --tuner_type full --torch_dtype bfloat16 \
  --dataset AI-MO/NuminaMath-TIR#1000 \
  --max_completion_length 1024 --num_generations 8 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 --num_train_epochs 3 \
  --deepspeed zero2 --beta 0.04 --log_completions true \
  --output_dir output/grpo_3b
```
> 想练 **7B GRPO** 把 `--model` 改 7B、`--tuner_type qlora`（QLoRA）以适配 24GB；rollout 侧 2 卡放 7B 足够。

### 5.2 进阶变体（对照面试文档第 3 节，逐个换参数体验）

| 想体验 | 改动（在 5.1 基础上） | 对应文档点 |
|---|---|---|
| DAPO | `--loss_type dapo --epsilon_high 0.28 --dynamic_sample true --overlong_filter true --reward_funcs soft_overlong` | DAPO.md / loss_types.md |
| GSPO | `--importance_sampling_level sequence` | GSPO.md |
| CISPO | `--loss_type cispo --epsilon_high 5.0` | CISPO.md |
| SAPO | `--loss_type sapo` | SAPO.md |
| RLOO | `--advantage_estimator rloo --kl_in_reward true` | RLOO.md |
| 训练-推理修正 | `--rollout_importance_sampling_mode token_truncate --rollout_importance_sampling_threshold 2` | training_inference_mismatch.md |

**学习目标（边跑边看日志）**：
- `reward_mean`/`reward_std` 是否随步数上升；
- `frac_reward_zero_std`（动态采样要消灭的量）；
- `clip_ratio`、`kl`；
- 打开 `--rollout_importance_sampling_mode` 后看 `rollout_correction/ESS`、`k3_kl`、`chi2_*`（mismatch 监控）。

---

## 6. 阶段 4：Agent 多轮 GRPO

复用 `examples/train/grpo/external/vllm_multi_turn.sh` 与 `agent.sh` 思路（注意 4090 改小模型/QLoRA）：
```bash
# 多轮 server 模式（同理：A 起 rollout，B 起训练）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC_PER_NODE=6 swift rlhf \
  --rlhf_type grpo --model Qwen/Qwen2.5-3B-Instruct \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port 8000 \
  --multi_turn_scheduler tool_call_scheduler --max_turns 4 \
  --reward_funcs tool_call_format ... \
  --dataset <工具调用数据集> --num_generations 8 --max_completion_length 2048
```
**学习目标**：
- `MultiTurnScheduler` 的 `check_finished`/`step`/`on_turn_end` hook（`multi_turn.md`）；
- 工具返回 `response_loss_mask=0` 不进 loss；
- async engine 减多轮气泡。

---

## 7. 阶段 5：推理 / 评测 / 部署 / 导出

```bash
# 推理（训练产物）
CUDA_VISIBLE_DEVICES=0 swift infer --model output/grpo_3b --stream true

# 评测（EvalScope 后端）
swift eval --model output/grpo_3b --dataset gsm8k --infer_backend pt

# 导出（合并 LoRA / 量化）：QLoRA 模型先 merge
swift export --model Qwen/Qwen2.5-7B-Instruct --adapters output/dpo_7b --output_dir output/dpo_7b_merged
# 量化（可选，4090 上 vLLM 部署更省显存）
swift export --model output/dpo_7b_merged --quant_method awq --quant_bits 4

# 部署（OpenAI 接口）
CUDA_VISIBLE_DEVICES=0 swift deploy --model output/dpo_7b_merged --infer_backend vllm
```
**学习目标**：全链路闭环；量化/合并在 4090 上的显存收益；deploy 用 vLLM 加速。

---

## 8. 阶段 6（可选）：长文本 / 小 MoE / Megatron 浅尝

- **序列并行长文本**：`--sequence_parallel_size N` + 长 `max_length`，验证 Ulysses/Ring（`swift/sequence_parallel`）。4090 上长上下文 SFT 必备。
- **小 MoE**：Qwen3-0.6B-MoE 类小模型做 MoE GRPO，体会 EP（专家并行）。
- **Megatron**：8×4090 无 NVLink 下 Megatron TP 很慢，**仅建议读 `swift/megatron` 代码 + 看文档**，不强制跑。

---

## 9. 学习节奏建议（把"复现"变成"学透"）

每天固定动作：
1. **跑一条命令** → 2. **读对应源码/文档** → 3. **记一条"增量学习点"**到 `ms-swift复现-增量学习点.md`。
建议源码阅读顺序（与阶段对应）：
- 阶段1：`swift/template`、`swift/tuners/lora.py`、`swift/trainers`
- 阶段2：`swift/rlhf_trainers/dpo_trainer.py`、`reward_trainer.py`
- 阶段3：`swift/rl_core/advantage.py` → `swift/rlhf_trainers/grpo_trainer.py:917-1176` → `swift/rewards/*` → `docs/.../GRPO/AdvancedResearch/*`
- 阶段4：`swift/rollout/multi_turn.py`、`swift/agent_template/*`、`Agent-support.md`
- 阶段5：`swift/cli/{infer,eval,export,deploy}.py`、`swift/infer_engine`

---

## 10. 风险与回退

| 风险 | 现象 | 回退/对策 |
|---|---|---|
| colocate 7B OOM | CUDA OOM | 改 server 拆分 / 改 QLoRA / 降 `max_completion_length` |
| rollout server 连不上 | 训练卡等不到采样 | 确认 `vllm_server_host/port` 一致；先单卡 `swift rollout` 自测 |
| GRPO 不收敛 | reward 不涨 | 查 `frac_reward_zero_std`、奖励函数是否只判格式（reward hacking） |
| 4090 上 vLLM 版本冲突 | 启动报错 | 锁定 vLLM 与 transformers 兼容版本（看 `requirements/`） |
| 多卡 ZeRO 慢 | 吞吐低 | 4090 无 NVLink，避免大 TP；用 ZeRO2 + DP + packing |

---

## 11. 交付物（复现完成时应有的产出）

1. 每个阶段各一份可复跑的 shell 脚本（放进 `examples/my_repro/`）。
2. TensorBoard/日志截图：SFT loss 曲线、GRPO `reward_mean` 上升、`kl` 可控。
3. 一份"我在 8×4090 上把 ms-swift 全流程跑通"的复盘（含踩坑与显存账）。
4. 持续更新的 `ms-swift复现-增量学习点.md`。

> 一句话：用 8×4090 跑通 ms-swift 全流水线，**核心约束是"无 NVLink→别开大 TP、显存够但单卡小→用 server 拆分/QLoRA"**；
> 模型以 3B 做 GRPO 全参、7B 做 SFT/DPO 与 QLoRA-GRPO 为主，既不浪费 192GB 总显存，又避开 24GB 单卡上限。
