# ms-swift 面试准备：LLM 后训练 & Agent 应用（算法实习向）

> 适用对象：正在找 **LLM 算法实习** 的硕士在读同学（MS）。
> 目标：通过精读 **ms-swift** 这个工业级训练框架的真实代码，把"后训练（Post-training）/ RLHF / GRPO / Agent 训练"这条面试高频主线的**底层原理 + 工程细节 + 可被深挖的代码级证据**串起来，让你能"讲得出来、答得下去、被追问不慌"。
> 用法建议：先通读"原理速通"，再看"代码级深挖"，最后用"面试实战"自检。每条都尽量给了 **文件:行号**，面试时能说"我在 ms-swift 里看过它的实现"是极大加分项。

---

## 0. 先认识 ms-swift：它是什么、为什么值得作为面试素材

ms-swift（Scalable lightWeight Infrastructure for Fine-Tuning）是 ModelScope 社区出品的大模型/多模态大模型**全链路**框架，覆盖：预训练、SFT、人类对齐（DPO/KTO/RM/GRPO…）、推理、评测、量化、部署。它目前（v4.0+）同时支持：

- **轻量微调**：LoRA / QLoRA / DoRA / LoRA+ / LLaMA-Pro / LongLoRA / ReFT / RS-LoRA 等；
- **人类对齐 / 偏好学习**：DPO、KTO、RM、CPO、SimPO、ORPO、GKD；
- **强化学习后训练**：内置完整的 **GRPO 算法家族**——GRPO、DAPO、GSPO、SAPO、CISPO、CHORD、RLOO、Reinforce++ 等；
- **Agent 训练**：Agent 模板（与具体模型解耦）+ 多轮工具调用 RL；
- **分布式 / 系统**：DDP、DeepSpeed ZeRO、FSDP、Megatron（TP/PP/CP/EP/VPP）、Ulysses / Ring 序列并行、vLLM/SGLang/LMDeploy 推理加速。

**面试价值**：ms-swift 把学术界最新的后训练算法（GRPO 家族、off-policy 修正、训练-推理 mismatch）都落到了**真实可运行的代码**里。面试官问"GRPO 和 PPO 区别""GRPO 的 loss 怎么写""为什么会有训练-推理不一致"，你都能用这个仓库里的具体实现来回答，比只会背公式强一个档次。

关键目录（面试前至少都点开看一眼）：

```
swift/rlhf_trainers/   # dpo/grpo/ppo/kto/orpo/gkd 等 Trainer 实现
swift/rl_core/         # 算法核心：advantage 计算、grpo_algorithm、resample、data
swift/rollout/         # 多轮 rollout、agent loop、gym env、multi_turn scheduler
swift/rewards/         # 奖励函数 / 奖励模型插件（ORM、GenRM）
swift/agent_template/  # ReAct / Hermes / Qwen 等 agent 模板
swift/tuners/          # LoRA 及各类高效微调
swift/template/        # 通用对话/训练模板（SFT 的 label/损失掩码在此产生）
docs/source_en/Instruction/GRPO/  # GRPO 家族每个算法的官方文档（必读）
```

---

## 1. 后训练主线一：SFT 与高效微调（LoRA）

### 1.1 原理速通（面试必会）

- **SFT（Supervised Fine-Tuning）**：在 `(prompt, response)` 上做 next-token 交叉熵，但只对 **response 部分** 算 loss（prompt 用 `-100` mask 掉）。ms-swift 在 `swift/template` 里把 `labels` 中 prompt 与 `tool_response` 等位置置为 `-100`。
- **LoRA**：冻结预训练权重 $W_0$，在旁路加低秩分解 $\Delta W = BA$（$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}, r\ll d$）。前向 $h = W_0x + \frac{\alpha}{r}BAx$。只训练 $A,B$，显存与参数量大幅下降。
- **常见变体（面试可聊）**：
  - **QLoRA**：4-bit NF4 量化底座 + LoRA，7B 仅需 ~9GB；
  - **LoRA+**：给 $A,B$ 不同学习率（$lora_lr_ratio$），加速收敛；
  - **DoRA**：把权重分解为"方向+幅度"分别训练；
  - **RS-LoRA**：对 LoRA 缩放因子做秩相关的重新缩放，稳定训练；
  - **LLaMA-Pro**：新增"扩展块"而非只加旁路，缓解灾难性遗忘；
  - **LongLoRA**：配合稀疏局部注意力做长上下文。

### 1.2 代码级证据

- `swift/tuners/lora.py:18` `LoRAConfig`（含 `lorap_lr_ratio` 即 LoRA+）、`:74` `class LoRA`、`:171` 注释说明 "LoRA constructs an additional layer with low-rank decomposition matrices"。
- `swift/tuners/__init__.py` / `mapping.py` 看支持的全部 tuner 列表。

### 1.3 面试实战

**让面试者介绍**（可主动甩给面试官展示）：
> "请介绍 LoRA 的原理，以及相比全参微调它解决了什么问题？"

**深挖追问（准备要点）**：
1. LoRA 为什么通常在 $W$ 上加而不是在 LayerNorm/embedding 上加？（秩的假设：权重更新是低秩的；LN/emb 参数量小但影响大，常全参训练）
2. $\alpha$ 和 $r$ 的作用分别是什么？$\alpha/r$ 作为缩放，太大易不稳定、太小学不动。
3. LoRA 的推理部署怎么做的？（合并权重 `merge` 或推理时旁路相加；ms-swift 支持 `swift export` 合并）
4. QLoRA 的 NF4 + 双重量化具体省了哪里的显存？（底座 4-bit + 优化器状态用 8-bit）
5. 为什么 MoE 模型微调常配合 LoRA？（专家参数多，全参贵；且 LoRA 对路由影响小）

---

## 2. 后训练主线二：偏好学习（DPO / ORPO / SimPO / KTO / RM）

### 2.1 原理速通

- **RM（Reward Model）**：在 SFT 模型上加 value head（输出维度 1），用 Bradley-Terry 偏好对 $(x, y_w, y_l)$ 训练。ms-swift 的 RM loss（见 `docs/source_en/Instruction/RLHF.md`）：
  $$\text{loss}=-\log\sigma\big(r^{(c)}-r^{(r)}-m\big)+\lambda(r^{(c)}+r^{(r)})^2$$
  其中 $\lambda=$ `center_rewards_coefficient`（鼓励分数靠近 0），$m=$ margin（数据集 `margin` 列）。
- **DPO（Direct Preference Optimization）**：不用显式奖励模型，直接对策略做偏好优化。核心是把奖励参数化为 $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}+c$，代入 BT 目标得到：
  $$\mathcal{L}_{\text{DPO}}=-\mathbb{E}\left[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big)\right]$$
  `beta` 是 KL 强度（默认 0.1）。ms-swift 还支持 `loss_type`（sigmoid/IPO/…）、`ld_alpha`（LD-DPO 缓解长度偏好）、`discopop_tau`、`rpo_alpha`（混入 SFT loss 提升稳定）。
- **ORPO / SimPO / KTO（无（显式）参考模型的变体）**：
  - **ORPO**：在 SFT loss 上直接加一个 odds-ratio 偏好项，不需要单独的 reference / 两阶段；
  - **SimPO**：用序列平均 log 概率作为隐式奖励，并引入 **target reward margin $\gamma$** 与长度归一化，省去 reference 模型；
  - **KTO**：只需要 $(x, y, \text{label}\in\{0,1\})$（是否合意），用人类快乐/痛苦的不对称损失，适合只有"好/坏"标注的数据。

### 2.2 代码级证据

- `swift/rlhf_trainers/dpo_trainer.py`、`orpo_trainer.py`、`kto_trainer.py`、`reward_trainer.py`、`cpo_trainer.py`、`gkd_trainer.py` 都存在，说明工业框架已把整套偏好算法工程化。
- RM loss 公式见 `docs/source_en/Instruction/RLHF.md` 的 RM 小节；DPO 超参见同文档 DPO 小节（`beta` 默认 0.1、`loss_type` 默认 sigmoid）。

### 2.3 面试实战

**让面试者介绍**：
> "请推导 DPO 是怎么从 RLHF + Bradley-Terry 得到的，并对比 PPO/GRPO 的 reward model 路线。"

**深挖追问**：
1. DPO 为什么会出现 **length bias**（偏好长回复）？如何缓解？（SimPO 长度归一化、LD-DPO 对共享前缀外 token 降权、`beta` 调节）
2. DPO 与 RM+PPO 相比的优缺点？（DPO 简单稳定、无需在线采样；但无法用过程奖励、对数据质量/分布更敏感）
3. KTO 相比 DPO 的数据要求有什么变化？为什么工业界有时更喜欢 KTO？（只需要二元标签，不需成对偏好）
4. `beta` 太大/太小分别会怎样？（大→贴近 reference 难学；小→偏离 reference 易崩溃）

---

## 3. 后训练主线三：强化学习（GRPO 家族）—— 重中之重

这是当前 LLM 后训练（尤其推理模型、数学/代码/agent）最热的方向，也是面试深挖重灾区。**务必把 3.1~3.6 吃透。**

### 3.1 GRPO 基础：为什么不用 PPO 的 value model

GRPO（Group Relative Policy Optimization）用**同一 prompt 的组内相对优势**替代 PPO 的 value 网络，并把 KL 直接写进 loss。目标（见 `docs/source_en/Instruction/GRPO/GetStarted/GRPO.md`）：

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{q,\{o_i\}_{i=1}^G}\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big[\rho_{i,t}\hat A_{i,t},\ \text{clip}(\rho_{i,t},1\!-\!\varepsilon,1\!+\!\varepsilon)\hat A_{i,t}\big]-\beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]\Big\}
$$

其中重要性采样比 $\rho_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$，组内优势：

$$
\hat A_{i,t}=\frac{R_i-\text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

**关键直觉（面试必须能说）**：
- 每个 prompt 采样 $G$ 个回答（`num_generations`），用"组内均值/标准差"做基线 → 不需要训练 critic/value model，省一半显存、训练更稳。
- KL 项用 **K3 估计**（见 `grpo_trainer.py:958-959`：$\text{KL}\approx e^{\Delta}-\Delta-1$），无需额外 forward。
- 三大阶段：**Rollout 采样 → Reward 打分 → Policy 优化**（伪代码见 GRPO.md）。

### 3.2 优势（Advantage）计算：代码即标准答案

ms-swift 的优势计算是纯 tensor 函数，适配 HF / Megatron / Ray 多后端（见 `swift/rl_core/advantage.py`）：

- `compute_advantages`（`:10`）：先按 `reward_weights` 加权求和各奖励函数 → `rewards`；可选把 `kl_in_reward` 的 KL 从 reward 里减掉（RLOO 风格，`:50`）；然后按 `num_generations` 分组求 `group_mean`。
  - **RLOO 估计器**（`:57`）：`advantages = rewards * K/(K-1) - group_mean * K/(K-1)`（leave-one-out，无偏基线）。
  - **GRPO / Reinforce++**：`advantages = rewards - group_mean`，再按 `scale_rewards`（`batch`/`group`/`gdpo`/`none`）做归一化（`:62-87`）。
  - **GDPO**（`:73-84`）：对每个奖励函数分别组内标准化再加权求和，再做一次整体标准化——**缓解多奖励函数尺度不一致**的问题。
- `compute_advantages_dynamic`（`:92`）：面向**动态采样 / 变长生成数**的 request-aware 版本，按 `prompt_id`/`request_id` 去重分组，支持每个 prompt 不同数量的回答（多轮场景）。

### 3.3 Loss 类型：归一化维度与梯度偏差（高频深挖）

这是区分"背过公式"和"真做过实验"的关键。文档 `docs/source_en/Instruction/GRPO/DeveloperGuide/loss_types.md` + 代码 `grpo_trainer.py:1065-1114`：

| loss_type | 归一化方式 | 要点 |
|---|---|---|
| `grpo` | 先对每个样本按 token 平均，再对样本平均 | **句子级归一化，引入长度偏差**（长样本每个 token 贡献被稀释） |
| `bnpo` | 所有 token 求和 / 总 completion token 数 | **token 级归一化**，消除长度偏差（DAPO 用） |
| `dr_grpo` | 求和 / `(batch_size × L_max)` | 用固定长度归一，避免 batch 长度分布带来的偏差 |
| `dapo` / `cispo` / `fipo` | 全局（跨进程）completion token 求和归一 | 多卡下更一致 |

**核心代码**（`:1065`）：
```python
if self.loss_type in ['grpo', 'sapo']:
    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()
elif self.loss_type == 'bnpo':
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
elif self.loss_type == 'dr_grpo':
    loss = (...) / (batch_size * self.max_completion_length)
elif self.loss_type in ['cispo', 'dapo', 'fipo']:
    normalizer = grpo_batch.num_items_in_batch / self.accelerator.num_processes
    loss = (per_token_loss * completion_mask).sum() / normalizer
```

**为什么重要**：GRPO 默认句子级归一化会**偏向短答案**——长 CoT 样本的 token loss 被平均稀释，模型可能学出"少说话"。DAPO/BNPO 改 token 级归一化即为此。

### 3.4 重要性采样层级：token vs sequence（GSPO）

`grpo_trainer.py:991-1006` 实现三种 IS 层级（`--importance_sampling_level`）：

- **token**（默认，GRPO）：$\log w=\log\pi_\theta-\log\pi_{\theta_{\text{old}}}$（逐 token）；
- **sequence**（GSPO）：$w_i=\exp\big(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}\big)$，序列级 IS 比；
- **sequence_token**（GSPO-token）：$\text{sg}[w_i^{\text{seq}}]\cdot\frac{\pi_\theta}{\text{sg}[\pi_\theta]}$。

**为什么这能成为深挖点**（见 `docs/.../GSPO.md`）：token 级 IS 每 token 只采样一次，无法真正做分布修正，方差高、易训练崩溃；序列级与目标（序列级 reward）单位一致，更稳。当所有 token 优势相同（GRPO 现状）时 GSPO-token ≡ GSPO（梯度等价），但 GSPO-token 预留了未来细粒度 token 优势的空间。

### 3.5 GRPO 家族算法：每个都要能讲清"它解决了什么"

面试常让你对比下列算法。ms-swift 在 `docs/source_en/Instruction/GRPO/AdvancedResearch/` 下每个都有专文 + 代码参数：

#### (a) DAPO —— 工程 tricks 集大成
见 `DAPO.md`。在 GRPO 基础上：
- **Clip-Higher（非对称裁剪）**：上界放松到 `epsilon_high=0.28`（鼓励探索），下界保持 `epsilon=0.2`（抑制）。解决对称裁剪限制低概率关键 token 被提升的问题。
- **Dynamic Sampling（动态采样）**：跳过"组内 reward 标准差为 0"的样本（此时优势全 0、梯度消失），持续重采样直到填满 batch（见 `grpo_algorithm.py:124` `compute_std_for_dynamic_sampling`）。
- **Token-level Loss**：即 `loss_type=bnpo/dapo`，消除长度偏差。
- **Overlong Filtering**：过滤被截断（超 `max_completion_length`）的样本，不参与 loss（代码 `grpo_trainer.py:948-953` `overlong_filter`）。
- **Soft Overlong Punishment**：三段式长度惩罚函数（见 DAPO.md 公式），`reward_funcs=soft_overlong`。

#### (b) CISPO —— 裁剪 IS 权重本身，保留梯度
见 `CISPO.md`（MiniMax-M1）。GRPO 裁剪的是"ratio×advantage"整体，会丢弃关键低概率 token 的梯度。CISPO 改为**裁剪 IS 权重 + detach**，梯度始终来自 $\log\pi_\theta$：
$$\mathcal{L}_{\text{CISPO}}=-\mathbb{E}\big[\text{detach}(\min(r_t,\epsilon_{\text{high}}))\cdot\hat A_t\cdot\log\pi_\theta\big]$$
代码（`grpo_trainer.py:1019-1021`）：
```python
clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
per_token_loss = -clamped_ratios * advantages * per_token_logps
```
`epsilon_high` 一般设较大（如 5.0，来自 ScaleRL）。

#### (c) SAPO —— 软门控替代硬裁剪
见 `SAPO.md`。用温度 sigmoid 软门控 $g_t=\sigma(\tau(r_t-1))\cdot\frac{4}{\tau}$ 替换 PPO 式硬 clip，平滑衰减离群（off-policy）更新、保留学习信号。代码 `grpo_trainer.py:1022-1028`（正/负优势分别用 $\tau_{\text{pos}}/\tau_{\text{neg}}$）。适合长文本、MoE 路由异质导致 IS 方差大的场景。

#### (d) RLOO —— Leave-One-Out 无偏基线
见 `RLOO.md`。优势 $\hat A_i=R_i-\frac{1}{K-1}\sum_{j\ne i}R_j=\frac{K}{K-1}(R_i-\bar R)$，**无偏**；KL 以 `kl_in_reward=true` 直接并入 reward（而非 loss 项）。参数：`--advantage_estimator rloo --kl_in_reward true`。

#### (e) GSPO / Reinforce++ / CHORD / FIPO / REAL ……
- **Reinforce++**：用 batch/group 级 std 归一化优势（见 `advantage.py:63-67`），可配合 `scale_rewards`。
- **CHORD / FIPO / REAL**：更进阶的变体（FIPO 在 DAPO 基础上加"未来 KL 影响权重" `fipo_weight`，见 `grpo_trainer.py:1032-1043`；REAL 用正负组对比的 listwise 损失，`:1029-1108`）。面试若被问到，能说出"这些是 GRPO 在方差/归一化/对比形式上的变体"即可。

### 3.6 奖励（Reward）：RL 的灵魂

ms-swift 的奖励支持多种形态（见 `swift/rl_core/grpo_algorithm.py:17` `compute_rewards_per_func`）：
- **规则奖励函数**（sync / async 均可，`_is_async_reward` 检测 coroutine，`:12`、`:66` asyncio 并发）：如格式正确、答案匹配、代码可运行。
- **判别式奖励模型（ORM）**：`DefaultRMPlugin`（`swift/rewards/rm_plugin.py:20`），取 value head 第一个 logit 作为分数。
- **生成式奖励模型（GenRM / LLM-as-judge）**：`GenRMPlugin`（`:41`），把对话拼成 query，让 judge 模型输出 `Reward: 0.85` 再用正则抽取（`:122` `extract_reward`），多 choice 取平均。
- **过程奖励（PRM）** 与 **ORM**：`swift/rewards/orm.py`、`prm.py`。
- **Gym 环境奖励**：`use_gym_env` 时把 `rollout_infos['total_reward']` 作为额外奖励列（`:100-113`）。

**重点细节**：
- 多奖励函数用 `reward_weights` 加权（`nanstd`/`nansum` 处理 `None` 奖励，`:48`、`:133`）。
- **`frac_reward_zero_std`**（`:320` `compute_reward_metrics`）：监控"组内 std 为 0"的比例——这正是 DAPO 动态采样要消灭的无效样本信号。

### 3.7 训练-推理不一致（Training-Inference Mismatch）：高阶系统工程点

见 `docs/.../training_inference_mismatch.md`，这是 **能拉开差距的硬核点**。

- **问题根源**：GRPO 用 vLLM 加速采样，理想假设 $\pi_{\text{vLLM}}\equiv\pi_\theta$。但即使权重同步，算子实现差异导致 $\pi_{\text{vLLM}}(y|x)\ne\pi_\theta(y|x)$ → **违反 on-policy 假设**，训练可能崩。
- **IS 修正**：引入权重 $w=\frac{\pi_\theta}{\pi_{\text{vLLM}}}$，可在 **token 级 / 序列级**，配合 **truncate / mask** 控制（`--rollout_importance_sampling_mode` 四选一：`token_truncate/token_mask/sequence_truncate/sequence_mask`）。代码 `grpo_trainer.py:1049-1050` 把 `rollout_is_weights` 乘进 loss。
- **监控指标**（面试能列出来很加分）：KL 散度（direct / K3）、PPL、`χ²` 散度（IS 权重方差）、**有效样本数 ESS**（`grpo_trainer.py` 的 `rollout_correction/` 日志）。ESS 越接近 1 越 on-policy，越小越 off-policy。
- **Off-Policy Sequence Masking**（来自 DeepSeek-V3.2）：当 $\delta_i=\frac{1}{|y_i|}\sum_t(\log\pi_{\text{old}}-\log\pi_\theta)>\tau$ **且** $\hat A_i<0$ 时，直接 mask 掉该序列（`:1052-1063`）。精准针对"负优势 + 策略偏移大"的不稳定样本。

### 3.8 面试实战（GRPO 专区）

**让面试者介绍（任选一个甩出去）**：
1. "请完整推导 GRPO 的目标函数，并解释它相比 PPO 省掉了什么、为什么更稳定。"
2. "GRPO 默认的 loss 归一化有什么问题？DAPO/BNPO 怎么解决？"
3. "讲讲 GRPO 家族里 DAPO / GSPO / CISPO / SAPO 各自针对什么痛点。"

**深挖追问清单（逐条准备 1 分钟口语答案）**：
- 组内 std=0 会发生什么？怎么处理？（梯度消失 → 动态采样跳过）
- `num_generations` 越大越好吗？（更多样本=更准基线，但算力线性增长；compute 与 memory 权衡）
- `beta`（KL 系数）设 0 会怎样？（模型可能偏离 reference 导致语言退化/崩；GSPO 论文也常设 0 但配小 epsilon）
- `epsilon` 与 `epsilon_high` 的作用与取值？（裁剪幅度；DAPO 上界 0.28）
- 为什么 GRPO 不需要 value model 但 PPO 需要？（组内相对优势替代 critic）
- reward 设计不好（全 0/全 1、奖励 hacking）怎么发现？（看 `frac_reward_zero_std`、reward_mean/std、clip_ratio 日志）
- 多奖励函数尺度不一怎么办？（GDPO 或 `reward_weights`）
- 训练-推理 mismatch 的表现与修正手段？（见 3.7）

---

## 4. 后训练主线四：Agent 训练与多轮 RL（应用向热点）

LLM & Agent 应用岗极可能问这块。ms-swift 把 Agent 训练做成**与模型解耦的模板 + 多轮 scheduler + RL 闭环**。

### 4.1 Agent 模板：数据格式与模型解耦

见 `docs/source_en/Instruction/Agent-support.md`。核心思想：`tools` 字段 + `tool_call`/`tool_response` 角色的 JSONL 数据，**一套数据可训练任意模型**。

- `tools` 是函数 schema 的 JSON 字符串；`tool_call`/`tool_response`(=`tool`) 的 `content` 也是 JSON 字符串。
- 训练时 `agent_template` 负责把：
  - `tools` + `system` 拼成完整 system；
  - 连续 `tool_call` 合并成 `assistant` 段；
  - `tool_response` 转换为 `user` 段，**且其 token 不计入 loss**（同 `user` 一样被 mask，`:18`）。
- 支持并行工具调用、多模态（`<image>` 数量与 `images` 对齐）。
- 模板示例：`hermes`（Qwen/GLM 常用）、`react_en`（ReAct 的 Thought/Action/Action Input/Observation 格式，`:113` 起）。

### 4.2 训练时的损失掩码（loss_scale）

`--loss_scale` 可对不同输出段加权/屏蔽（`:196` 起）：
- `react`：Action/Action Input 权重 2，Observation 本身 2 但**其后的工具结果内容权重 0**（不学环境返回）；
- `ignore_empty_think`：正则匹配 `<think>\s*</think>` 置 0，避免空思考块贡献 loss。
- 原理：工具返回/环境反馈是**外部生成**，模型不该被惩罚去"预测"它们。

### 4.3 多轮 RL：MultiTurnScheduler

见 `docs/.../GRPO/DeveloperGuide/multi_turn.md`，核心类 `MultiTurnScheduler`（`swift/rollout/multi_turn.py`）：

- `check_finished(req, response_choice, turn)`（`:73`）：终止条件 = 回复被截断（`finish_reason=='length'`）或到达 `max_turns`。
- `step(...)`（`:53`）：构造下一轮请求，可返回 `infer_request / response_token_ids / response_loss_mask / rollout_logprobs / rollout_infos`。
- **Hook 协议**（`on_trajectory_start` / `on_turn_end`）：默认 `GYMScheduler` 用 `env.reset` / `env.step`，用户只需实现 `Env` 接口即可接入标准 gym 环境（FrozenLake 例子开箱即用）。
- **AsyncEngine**：多轮 rollout 用异步批处理 engine 减少计算气泡（`multi_turn.md` 图示），`use_async_engine`（server 模式）。

### 4.4 Agent RL 的关键坑（面试加分项）

1. **Credit assignment（信用分配）**：整条多轮轨迹通常当作一个样本算 loss（一个 reward 赋给整条轨迹，见 `multi_turn.md:228`）。若中途改了 history（如压缩上下文），则每轮单独成样本、同一 trajectory 的 record 必须共享同一 reward。
2. **Loss mask 工具返回**：工具/环境返回内容要 `response_loss_mask=0`，否则模型被强迫"背诵"外部结果（`:255-273`）。
3. **rollout logprobs 回流**：多轮需把每轮的 `rollout_logprobs` 回传 trainer 做 off-policy 修正（`rollout_importance_sampling_mode`，`:304`）。
4. **reward 函数读取多轮信息**：scheduler 把 `rollout_infos` 写入，reward 函数从 `kwargs['rollout_infos']` 读取（`:278`）。

### 4.5 面试实战（Agent 专区）

**让面试者介绍**：
> "如果要训练一个会调用工具的多轮 Agent，训练数据和 loss 设计上要注意什么？"

**深挖追问**：
- 为什么 `tool_response` 不参与 loss？（外部生成，不应作为预测目标；否则模型学去模仿环境输出）
- 多轮 RL 里 reward 怎么分配给每一步？（整条轨迹共享终局 reward；若分步奖励需 scheduler 在 `rollout_infos` 记录）
- 多轮采样为什么用 async engine？（减少轮间等待气泡）
- Agent 模板解耦的好处？（一套数据训多模型、易切换）
- 工具调用格式错误（JSON 不合法）如何奖励/惩罚？（规则奖励函数检测 JSON 合法性 + 格式奖励）

---

## 5. 系统与工程：能体现"工程水位"的点

面试官看实习生态度，常顺带问系统。ms-swift 覆盖很全，挑高频的讲：

### 5.1 Rollout 两种部署模式（GRPO.md）
- **Colocate（内部）模式**：训练与推理**共享 GPU**，inference 在 Trainer 内启动（`--vllm_mode colocate`）。省显存切换开销小，但推理吞吐受训练占用影响。
- **Server 模式**：vLLM 独立 server，Trainer 远程调用采样。吞吐高，支持 async engine，但需权重同步（引入 3.7 的 mismatch）。
- 还有 Megatron GRPO 的 colocate 模式，适合超大 MoE。

### 5.2 序列并行与 Megatron（长文本 / 大模型）
- **Ulysses / Ring-Attention 序列并行**（`--sequence_parallel_size N`）：把长序列切到多卡，降低单卡显存，长文本训练必备（`swift/sequence_parallel`）。
- **Megatron**：TP/PP/CP/EP/VPP 并行，`swift/megatron`，显著加速 **MoE 训练**（专家并行 EP）。面试能说"MoE 用 EP + vLLM 张量并行做 GRPO"就是加分。

### 5.3 显存优化全家桶
GaLore / Q-GaLore、Unsloth、Liger-Kernel、Flash-Attention 2/3、量化训练（BNB/AWQ/GPTQ）。面试问"7B 怎么在单卡训"→ QLoRA + 这些优化，9GB 可训。

### 5.4 面试实战
- "GRPO 训练里 vLLM 和训练模型怎么协同？为什么要权重同步？不一致会怎样？"（→ 3.7）
- "训练一个 72B 模型做 GRPO，显存/算力不够怎么办？"（→ 4×80G 用 vLLM 张量并行 + LoRA/QLoRA + 序列并行 + 异步 engine）
- "长文本（32k+）SFT/GRPO 怎么不 OOM？"（→ 序列并行 + packing + 梯度检查点）

---

## 6. 一页速查：被追问时的"标准回答骨架"

**GRPO 一句话总结**：
> 对每个 prompt 采样 G 个回答，用组内相对优势（去均值除标准差）替代 PPO 的 value 网络做基线，配合 PPO 式裁剪和 KL 惩罚更新策略；实现见 `rl_core/advantage.py` + `rlhf_trainers/grpo_trainer.py`。

**GRPO vs PPO**：
> 省掉 value/critic 模型（省显存、更稳），用组内 baseline 替代 GAE；仍保留 importance sampling 裁剪与 ref 模型 KL。

**为什么 GRPO 要动态采样**：
> 组内 reward 全相同 → std=0 → 优势全 0 → 梯度消失、训练低效；DAPO 跳过这类样本并重采样（监控 `frac_reward_zero_std`）。

**长度偏差从哪来、怎么解**：
> 默认句子级 loss 归一化稀释长样本 token 贡献 → 模型偏好短答；改 `bnpo/dapo`（token 级全局归一）或 `dr_grpo`（固定长度归一）。

**训练-推理 mismatch**：
> vLLM 与训练模型算子差异导致分布不一致、违反 on-policy；用 IS 修正（token/seq × truncate/mask）并监控 ESS/χ²/KL，或 off-policy sequence masking。

**Agent 训练要点**：
> 模板解耦 + tool_response 不进 loss + 多轮 scheduler 管理终止/环境交互 + 整条轨迹共享 reward + 异步 engine 减少气泡。

---

## 7. 给面试官的反向展示清单（你能主动抛的"我会"）

面试结尾常被问"你还有什么想展示的 / 你最熟哪块"。建议挑 1~2 个，结合 ms-swift 代码说：

1. "我精读过 ms-swift 的 GRPO 实现，能讲清 `advantage.py` 里 GRPO/RLOO/Reinforce++/GDPO 四种优势估计，以及 `grpo_trainer.py` 里 token/sequence IS 层级和 grpo/bnpo/cispo 的归一化差异。"
2. "我理解训练-推理 mismatch 的成因与 IS 修正，能解释 ESS、χ² 这些监控指标。"
3. "我做过/研究过多轮 Agent RL，知道 scheduler 的 check_finished/step hook、loss mask 工具返回、异步 engine 减少气泡。"
4. "我能对比 DPO 与 GRPO 两条人类对齐路线，以及 SimPO/ORPO/KTO 为什么不需要 reference 模型。"

> 小贴士：说"我在 ms-swift 源码里看到……"比"论文里说……"更有说服力。每个论点都尽量回扣到第 1~5 节给的 **文件:行号**。

---

## 附：推荐精读路径（按面试优先级）

1. `docs/source_en/Instruction/GRPO/GetStarted/GRPO.md` —— 算法总览与伪代码
2. `swift/rl_core/advantage.py` —— 优势计算（GRPO/RLOO/Reinforce++/GDPO/动态）
3. `swift/rlhf_trainers/grpo_trainer.py:917-1176` —— 真实 loss（IS 层级、clip、归一化、KL、mismatch）
4. `docs/source_en/Instruction/GRPO/AdvancedResearch/{DAPO,GSPO,CISPO,SAPO,RLOO}.md` —— 家族对比
5. `docs/source_en/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md` —— off-policy 硬核
6. `docs/source_en/Instruction/Agent-support.md` + `docs/.../GRPO/DeveloperGuide/multi_turn.md` —— Agent 训练
7. `docs/source_en/Instruction/RLHF.md` —— DPO/RM/KTO/PPO 偏好学习
8. `swift/tuners/lora.py` —— 高效微调

祝面试顺利。把"原理 → 代码 → 对比 → 坑"这条链练熟，实习生水位完全够用，且能在一众只背公式的候选人里脱颖而出。
