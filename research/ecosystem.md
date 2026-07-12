# OPD 生态、论文谱系与证据核验底稿

> 核验日期：2026-07-13（Asia/Shanghai）  
> 范围：以本地 `asset/AwesomeOPD/README.md`（自述最后更新 2026-06-23）为索引，并重点核验用户点名的论文。技术结论只引用 arXiv、作者/机构仓库、作者项目页或厂商技术报告；AwesomeOPD 只用来确定“需要查哪些条目”，不作为算法事实的最终证据。

覆盖检查：本地 README 共解析出 139 个主表资源行（含交叉重复；134 个唯一“名称+首链接”、132 个唯一首链接），其首个一手资源链接均已出现在第 8 节导航中。

## 1. 先给结论

1. 用户点名的 2026 年论文都能在 arXiv 上核验到，不是虚构条目；但多个版本号已经更新。特别是 Entropy-Aware OPD 已由用户给出的 v1 更新到 v3，IW-OPD 已到 v3，Prune-OPD 已到 v3，TrOPD 已到 v3。
2. `G-OPD / ExOPD` 是同一篇论文里的框架与特例，不是两篇论文；`Adaptive Target Reformulation / Veto` 也是同一篇论文与方法名；`AOPD` 的 A 是 **Asymmetric**，不要与 Adaptive Target Reformulation 混为一谈。
3. 用户原文里的 `TrOPDGKD / On-Policy Distillation...` 很像两项内容粘连：**TrOPD** 是 2026 年的 *Trust Region On-Policy Distillation*（arXiv:2606.01249），**GKD** 是 2023 年 Agarwal 等人的 *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*（arXiv:2306.13649）。两者并非同名方法。
4. *The Many Faces of OPD* 是诊断研究，不是一套可以直接替换 vanilla OPD 的单一训练算法。它最重要的价值是给出三类失败机制和对应稳定化手段。
5. 公式驱动综述中的 GAE-OPD、CR-OPD 是作者提出的研究假设/议程。论文存在的证据很强，但不能把这些尚未充分实证的提案写成“已经验证优于 OPD 的新算法”。
6. AwesomeOPD 的 2026-06-23 快照并不完整：用户点名的 AOPD、Prune-OPD、IW-OPD、TrOPD、公式驱动综述均未出现在该快照主表里。因此网站应把 AwesomeOPD 当作导航索引，而不是封闭且完备的 taxonomy。

## 2. 证据等级与“开源”的定义

| 等级 | 含义 | 可以在网站中怎么写 |
|---|---|---|
| E3 | arXiv/官方报告 + 作者或机构的公开源码仓库，且本次看到了训练入口、核心文件或复现实例 | “论文与实现均已公开”；仍不等于我们已经复现结果 |
| E2 | arXiv/官方报告可核验，摘要或正文直接支持方法描述，但没有核验到官方源码 | “论文已公开；未核验到官方实现” |
| E1 | 只有官方项目页、模型页或仓库说明，或只是对条目做了链接级盘点 | 只写存在性和项目自述，不替作者做更强结论 |
| E0 | 链接失效、命名含混、只有二手列表声称存在 | 明确写“未核验/待核验”，不能进入核心教程事实层 |

这里的“未核验到官方源码”是保守表述：表示 arXiv 页面没有给出代码链接、且本轮没有找到可明确归属于作者的仓库；它不等于数学意义上的“互联网上绝对不存在代码”。公开 GitHub 仓库也不自动等于完整可复现实现，可能只有 README、评测脚本或占位文件。

## 3. 基础 OPD：统一参照系

对提示词 `x`，学生生成自己的轨迹 `y ~ pi_theta(.|x)`；在每个学生实际访问的前缀 `s_t=(x,y_<t)` 上，教师给出分布或评分。最常见的 sampled-token 形式把

`a_t = log pi_T(y_t|s_t) - log pi_theta(y_t|s_t)`

当作停止梯度的稠密 token 信号，再更新 `log pi_theta(y_t|s_t)`。直接计算全词表 KL、top-k 截断 KL、JSD，以及把 `a_t` 当 RL advantage 的实现，其梯度、偏差与方差并不相同；教程里必须把“目标散度”和“实际估计器”分开讲。

- [GKD / On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) 给出 student-generated outputs、teacher feedback、可切换散度与 on/off-policy 混合的基础框架。
- [Thinking Machines Lab 的 OPD 文章](https://thinkingmachines.ai/blog/on-policy-distillation/) 采用逐 token reverse KL，把它解释为 on-policy 的状态覆盖 + distillation 的稠密监督，并给出可运行的 [tinker-cookbook recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)。
- [PPO](https://arxiv.org/abs/1707.06347) 是理解“采样—代理目标—多轮小批更新—trust region”语言的背景材料，但 PPO 本身不是 OPD，也没有教师在学生 token 上给稠密分布监督。

## 4. 方法谱系：各分支到底改了 vanilla OPD 的哪一环

| 分支 | vanilla OPD 中被改造的环节 | 代表方法 | 核心问题 |
|---|---|---|---|
| 奠基与通用目标 | rollout 来源、散度、on/off-policy 混合 | GKD、MiniLLM、DistiLLM | 如何在学生自己的状态上做蒸馏，并控制 mode-seeking / mode-covering |
| 目标分布几何 | 教师目标不再直接照搬 | Veto、Entropy-Aware OPD | 教师太远或高熵时，直接 FKL/RKL 会产生病态梯度或多样性塌缩 |
| advantage / reward 设计 | log-ratio 的符号、缩放、参考策略 | G-OPD/ExOPD、AOPD、OPD+ | OPD 与 KL-constrained RL 的关系；负 advantage 是否真的应继续做负强化 |
| 支持集与 token 选择 | 全词表/单采样 token 改为可靠局部支持 | Revisiting OPD、TIP、FiRe-OPD | 单 token 信号不平衡，top-k 估计偏差，高损失 token 主导训练 |
| 状态兼容性与冷启动 | 先让学生进入教师“会教”的状态 | Rethinking OPD、AdaSwitch、TrOPD | 学生前缀偏离教师分布后，教师局部建议可能不可靠 |
| 轨迹与位置信用分配 | token 等权改为按位置/漂移/轨迹质量加权 | Prune-OPD、IW-OPD、TRD、Fast OPD | 长链后段监督衰减、prefix drift、无效长 rollout 浪费算力 |
| verifier 双路径 | 先按正确性路由，再决定监督 | SCOPE | 正确轨迹与错误轨迹不该吃同一种 KL 信号 |
| 自蒸馏 | 外部大教师改为同一模型 + 特权上下文 | OPSD、OPCD、Skill-SD | 把 ground-truth trace、提示、技能或记忆内化到无特权策略 |
| 失败诊断 | 不提出单一替代算法，分析“何时不工作” | Revisiting OPD、Rethinking OPD、Many Faces | 教师选择、分布失配、biased top-k、instance-specific PI 缺失 |
| 黑盒与 outcome 教师 | logits 改为 discriminator、verbal feedback、rubric | GAD、OVD、ROPD | 无法访问教师 logits 时如何保留 on-policy 训练 |
| RL 混合 | token 教师信号与序列结果奖励融合 | KDRL、RLSD、TGPO、HDPO | 稠密模仿与稀疏探索如何兼得 |
| 系统与应用扩展 | teacher serving、跨 tokenizer、多模态、agent、draft model | verl/KDFlow、VLM/agent OPD、Draft-OPD | 通信、显存、异步陈旧、多轮状态、不同动作空间 |

一个更稳妥的教学主线是：**先解释状态分布，再解释散度，再解释估计器，最后才讲各种加权/裁剪。** 如果只按论文时间线罗列，初学者很容易误以为这些论文都只是在“换一个 KL”。

## 5. 用户点名论文逐项核验

| 资源 | 截至 2026-07-13 的版本与准确标题 | 相对基础 OPD 的关键改动 | 官方代码状态 | 证据与注意事项 |
|---|---|---|---|---|
| [Revisiting OPD](https://arxiv.org/abs/2603.25562v2) | arXiv:2603.25562 **v2**，2026-04-27；*Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes* | 识别 token 信号不平衡、学生前缀上的教师指导不可靠、tokenizer/特殊 token 不匹配；采用 teacher top-K local support、截断 reverse-KL、top-p rollout、特殊 token mask | [官方仓库](https://github.com/hhh675597/revisiting_opd) 有 verl fork、训练脚本与 Teacher-TopK 配置 | E3。论文摘要报告相对 sampled-token OPD 的稳定性与性能提升；网站应说“作者报告”，不要写成跨设置定律 |
| [G-OPD / ExOPD](https://arxiv.org/abs/2602.12125) | arXiv:2602.12125 **v2**，2026-02-26；*Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation* | 把 OPD 写成 dense KL-constrained RL，引入灵活 reference model 与 reward scale；scale > 1 的设置称 ExOPD，另讨论用教师的 pre-RL base 做 reward correction | [官方仓库](https://github.com/RUCBM/G-OPD) 已公开训练/评测代码与数据链接，基于 verl 0.6.1 | E3。ExOPD 是 G-OPD 的参数区间/具体策略，不是第二篇论文；“超越教师”是特定任务与组合实验结论 |
| [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) | arXiv:2603.07079 最新 **v3**，2026-06-12；*Entropy-Aware On-Policy Distillation of Language Models*，论文页标注 ICML 2026 | 教师低熵位置保留 mode-seeking 的 RKL；教师高熵位置加入 FKL，以覆盖多个合理模式、维持生成熵 | arXiv 页未链接官方仓库；本轮未核验到作者代码 | E2。用户给的是 v1，教程链接应指无版本 URL或 v3；不要把它粗写成“全局在 FKL/RKL 二选一”，其关键是 token-level teacher entropy |
| [SCOPE](https://arxiv.org/abs/2604.10688) | arXiv:2604.10688 最新 **v2**，2026-05-30；*SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting* | 按 verifier correctness 路由：错误轨迹做 teacher-PPL 加权 KL，正确轨迹做 student-PPL 加权 MLE；再做 group normalization | [官方仓库](https://github.com/machine981/SCOPE) 有 data、examples、recipe、verl 与运行脚本 | E3。它不是单纯“一个 token 权重公式”，而是 correctness routing + 两条目标路径 |
| [Formula-Driven Survey](https://arxiv.org/abs/2606.22793) | arXiv:2606.22793 **v1**，2026-06-22；*A Formula-Driven Survey and Research Agenda for On-Policy Distillation* | 以“feedback-to-update”拆分直接分布损失与 policy-gradient log-ratio；区分 temporal credit 与 vocabulary routing；提出 GAE-OPD、CR-OPD 研究方向 | 无需期待训练仓库；arXiv 页未给官方代码 | E2（存在性/综述内容）；对 GAE-OPD、CR-OPD 只能标“研究假设/议程”，不能标成已充分实证的 SOTA 方法 |
| [Rethinking OPD](https://arxiv.org/abs/2604.13016) | arXiv:2604.13016 **v2**，2026-04-15；*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* | 给出两个成功条件：thinking pattern 兼容、教师确有学生未见的新能力；观察高概率共享 token 集的渐进对齐；建议 off-policy cold start 与 teacher-aligned prompt selection | [THUNLP 官方仓库](https://github.com/thunlp/OPD) 有 LLaMA-Factory、verl、脚本与 `on_policy_distillation.sh` | E3。它是机制与 recipe 研究，不应简化成“progressive top-k 是新 loss” |
| [IW-OPD / Position Bias](https://arxiv.org/abs/2606.22600) | arXiv:2606.22600 最新 **v3**，2026-06-26；*On the Position Bias of On-Policy Distillation* | 发现后段 prefix drift 使监督质量下降；权重依赖累计 teacher-student discrepancy，自然前高后低 | arXiv 页未链接官方仓库；本轮未核验到作者代码 | E2。论文报告只用前 30% 可接近全长、只用后 30% 几乎学不到；这是其实验观察，不是所有模型的固定 30% 规则 |
| [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734) | arXiv:2601.18734 最新 **v3**，2026-03-20；*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* | 同一模型作 teacher/student；teacher 额外看到 verified reasoning trace 等 privileged information，student 只看问题；在 student rollout 上做逐 token 分布匹配 | [官方仓库](https://github.com/siyan-zhao/OPSD) 含 `opsd_train.py`、`opsd_trainer.py`、脚本与评测；基于 TRL GOLD | E3。仓库 2026-03 更新还加入逐 token KL clipping；“同一个模型”不等于 teacher 与 student 输入相同 |
| [AOPD](https://arxiv.org/abs/2605.06387) | arXiv:2605.06387 **v3**，2026-05-13；*Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level* | 正 advantage 区域保留正向强化；非正 advantage 区域不用低效负强化，而改做 localized divergence minimization | arXiv 页未链接官方仓库；本轮未核验到作者代码 | E2。A = Asymmetric。与 Veto 的 Adaptive Target Reformulation 是两条不同思路 |
| [Prune-OPD](https://arxiv.org/abs/2605.07804) | arXiv:2605.07804 最新 **v3**，2026-06-01；*Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning* | 用 top-k overlap 等局部兼容性信号在线检测 prefix drift；漂移后单调下调 reward 并动态截断 rollout，把算力移到可利用监督上 | arXiv 页未链接官方仓库；本轮未核验到作者代码 | E2。不是固定截短；兼容性高时会保留/扩展长上下文窗口 |
| [The Many Faces of OPD](https://arxiv.org/abs/2605.11182) | arXiv:2605.11182 **v2**，2026-05-24；*The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes* | 诊断三类失败：student-prefix 下的分布失配、biased TopK RKL 梯度导致优化不稳、OPSD 在 instance-specific PI 上只能学到 PI-free 聚合策略；考察 stop-gradient TopK、RLVR-adapted teacher、SFT-stabilized student | arXiv 页未链接官方代码 | E2。它不是一个名为 “Many Faces” 的新 trainer；对 OPSD 的负结论限定在论文测试的 PI 类型与设置 |
| [Veto / Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) | arXiv:2601.07155 **v2**，2026-04-20；*Stable On-Policy Distillation through Adaptive Target Reformulation*；论文页标注 ACL 2026 Findings | 在 logit 空间构造 teacher 与 student 之间的几何中间目标；参数 beta 同时充当 harmful-gradient veto 与 decisiveness knob，缓和 FKL 病态梯度/RKL 多样性塌缩 | arXiv 页未链接官方仓库 | E2。“Veto”是方法名，“Adaptive Target Reformulation”是论文题目里的机制描述，不是两个独立条目 |
| [TrOPD](https://arxiv.org/abs/2606.01249) | arXiv:2606.01249 最新 **v3**，2026-06-17；*Trust Region On-Policy Distillation* | 只在 teacher supervision 可靠的 trust region 做 OPD；outlier 区域比较 clipping、mask、FKL；再从 teacher prefix 继续生成并用 FKL 做 off-policy guidance | arXiv 页未链接官方仓库；本轮未核验到作者代码 | E2。TrOPD 与 GKD 是不同年代、不同目标的论文；TrOPD 也不是 PPO/TRPO 的简单改名 |

## 6. 这些论文其实在回答四个不同问题

### 6.1 教师在学生状态上“会不会教”

Rethinking OPD、Revisiting OPD、Many Faces、Prune-OPD、IW-OPD 和 TrOPD 的共同母题是：学生前缀越走越偏，教师即使总体更强，也不代表在这个局部状态上仍给出可利用、低方差的下一 token 信号。

- Rethinking OPD 从 teacher/student thinking-pattern compatibility 与 prompt/cold-start 入手；
- Revisiting OPD 从 teacher-supported local vocabulary 与 special-token/tokenizer 实现细节入手；
- Prune-OPD 与 IW-OPD 沿序列位置处理监督衰减；
- TrOPD 直接把可靠区域定义成 trust region，并给 outlier 单独处理；
- Many Faces 说明同一个“分布失配”还会与估计器偏差、学生初始化共同作用。

因此不要把这些方法画成互斥选项；它们分别作用于 prompt/初始化、词表支持、位置权重、轨迹长度和优化区域，可以组合，但组合是否稳定需要新实验。

### 6.2 教师分布本身该如何进入目标

- Veto：先构造学生与教师之间的中间目标，避免一步跨太远。
- Entropy-Aware OPD：教师不确定时更多 mode covering，教师确定时更多 mode seeking。
- AOPD：按 advantage 正负选择“强化”还是“局部模仿”。
- G-OPD/ExOPD：把 teacher log-ratio 看作 reward，并解耦 reward scale 与 KL reference。

这四者都在“改目标”，但控制变量不同：距离、熵、advantage 符号、reward/reference。教程中应该并列比较，而不是放进一个模糊的“adaptive KL”篮子里。

### 6.3 正确与错误轨迹是否应同等处理

SCOPE 的答案是否定的：错误 rollout 需要教师纠错，但只有教师真的比学生更会时才应高权重；正确 rollout 更适合用学生置信度挑出能力边界样本做 MLE。这与单纯把 verifier reward 乘在 KL 上不同。

### 6.4 privileged information 能否被无损内化

OPSD 假设同一模型在额外解答/提示上下文下形成更强 teacher policy，然后把它蒸馏到不看 PI 的 student policy。Many Faces 提醒：若 PI 是每题不同的 instance-specific information，测试时消失后，学生只能学 PI-conditioned teachers 的聚合，未必能重建每道题的缺失信息；若 PI 是共享规则、system prompt 或稳定偏好，则更有希望被内化。这是 OPSD 章节必须讲清的边界。

## 7. 推荐选读路径

### 路线 A：第一次入门（先建立直觉）

1. [Thinking Machines Lab：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
2. [GKD](https://arxiv.org/abs/2306.13649) 的摘要、方法图与 on/off-policy mixture
3. [tinker-cookbook distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
4. [verl OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html) 与本地 `asset/verl/recipe/on_policy_distill/`

### 路线 B：准备第一次实验（少踩坑）

1. [Revisiting OPD](https://arxiv.org/abs/2603.25562v2)：top-p、special-token mask、teacher top-k support
2. [Rethinking OPD](https://arxiv.org/abs/2604.13016)：先测 teacher/student compatibility，再做 cold start 与 prompt selection
3. [Many Faces](https://arxiv.org/abs/2605.11182)：检查 teacher、loss estimator、初始化是否处在已知失败区
4. [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) 与 [Veto](https://arxiv.org/abs/2601.07155)：理解“稳定”与“多样性”不是同一指标

### 路线 C：长链推理与算力优化

1. [IW-OPD](https://arxiv.org/abs/2606.22600)
2. [Prune-OPD](https://arxiv.org/abs/2605.07804)
3. [TRD](https://arxiv.org/abs/2606.08432)
4. [FiRe-OPD](https://arxiv.org/abs/2606.02684)
5. [TrOPD](https://arxiv.org/abs/2606.01249)

### 路线 D：OPD 与 RL 的统一视角

1. [PPO](https://arxiv.org/abs/1707.06347) 只补 policy-gradient / surrogate objective 背景
2. [G-OPD / ExOPD](https://arxiv.org/abs/2602.12125)
3. [OPD+](https://arxiv.org/abs/2606.01039)
4. [Formula-Driven Survey](https://arxiv.org/abs/2606.22793)，重点看 temporal credit 与 vocabulary routing 的区分

### 路线 E：自蒸馏与 agent

1. [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734)
2. [Many Faces](https://arxiv.org/abs/2605.11182) 的 PI-free policy 限制
3. [OPCD](https://arxiv.org/abs/2602.12275)、[Skill-SD](https://arxiv.org/abs/2604.10674)、[SDAR](https://arxiv.org/abs/2605.15155)

## 8. AwesomeOPD 2026-06-23 快照：完整项目/论文导航

下面覆盖本地 README 主资源表的所有条目（交叉列出的项目会注明）。这是**链接级清单**，不是对每个结果的独立复现。

标记：`[repo]` = README 指向公开作者/机构仓库（不保证完整复现）；`[paper]` = 本轮索引层只有论文；`[page]` = 官方博客、项目页、模型页或报告；`[strict?]` = 与“学生当下采样 + 教师在这些样本上监督”的严格定义存在边界。

### 8.1 综述、奠基与立场/诊断

- [GKD](https://arxiv.org/abs/2306.13649) `[paper]`（另有 [TRL GKDTrainer](https://github.com/huggingface/trl/tree/main/trl/experimental/gkd)）；[Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation/) `[page]`；[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation) `[repo]`。
- [Revisiting OPD](https://github.com/hhh675597/revisiting_opd) `[repo]`；[Tencent OPD Survey](https://arxiv.org/abs/2604.00626) `[paper]`；[THUNLP Rethinking OPD](https://github.com/thunlp/OPD) `[repo]`。
- [Lightning OPD](https://arxiv.org/abs/2604.13010) `[paper][strict?]`（缓存 SFT rollout 上的 teacher log-prob，作者称 offline OPD）；[OPSD Survey](https://arxiv.org/abs/2605.18141) `[paper]`；[The Many Faces of OPD](https://arxiv.org/abs/2605.11182) `[paper]`；[Li Jiang/TRD reflection](https://louieworth.github.io/blog/opd_reflection/) `[page]`。

### 8.2 外部白盒教师

- [MiniLLM / LMOps](https://github.com/microsoft/LMOps/tree/main/minillm) `[repo]`；[DistiLLM](https://github.com/jongwooko/distillm) `[repo]`；[Speculative KD](https://github.com/google-research/google-research/tree/master/speculative_kd) `[repo]`；[DistiLLM-2](https://github.com/jongwooko/distillm-2) `[repo]`；[DSKDv2](https://github.com/songmzhang/DSKDv2) `[repo]`。
- [Constrained OPD](https://arxiv.org/abs/2509.22921) `[paper][strict?]`；[AdaSwitch](https://arxiv.org/abs/2510.07842) `[paper]`；[Veto](https://arxiv.org/abs/2601.07155) `[paper]`；[G-OPD](https://github.com/RUCBM/G-OPD) `[repo]`；[Fast OPD](https://arxiv.org/abs/2602.15260) `[paper]`。
- [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) `[paper]`；[REOPOLD](https://arxiv.org/abs/2603.11137) `[paper]`（README 写 `code soon`，不能标已开源）；[PACED](https://arxiv.org/abs/2603.11178) / [共享仓库](https://github.com/HJSang/OPSD_OnPolicyDistillation) `[repo]`；[TSD-KD](https://github.com/kmswin1/TSD-KD) `[repo]`。
- [SCOPE](https://github.com/machine981/SCOPE) `[repo]`；[TIP](https://arxiv.org/abs/2604.14084) / [共享仓库](https://github.com/HJSang/OPSD_OnPolicyDistillation) `[repo]`；[HPD](https://github.com/zwhong714/Hybrid-Policy-Distillation) `[repo]`；[BRTS](https://github.com/BWGZK-keke/BRTS) `[repo]`。
- [FiRe-OPD](https://github.com/YuYingLi0/FiRe-OPD) `[repo]`；[TRD](https://github.com/louieworth/trd) `[repo]`；[OPRD](https://github.com/ShenzhiYang2000/OPRD) `[repo]`。

### 8.3 黑盒 / outcome 教师

- [ORPO-Distill](https://arxiv.org/abs/2509.25100) `[paper]`；[GAD / LMOps](https://github.com/microsoft/LMOps) `[repo]` 与 [作者项目页](https://ytianzhu.github.io/Generative-Adversarial-Distillation/)；[OVD](https://arxiv.org/abs/2601.21968) `[paper]`（README 记录其独立项目页 404，故不要给“项目可用”标记）。
- [SPOT](https://github.com/Visual-AI/SPoT) `[repo]`；[SODA](https://arxiv.org/pdf/2604.03873) `[paper][strict?]`；[ROPD](https://github.com/Peregrine123/ROPD_official) `[repo]`。

### 8.4 特权上下文自蒸馏（OPSD）

- [Self-Distilled Reasoner / OPSD](https://github.com/siyan-zhao/OPSD) `[repo]`；[SDFT-Continual](https://github.com/idanshen/Self-Distillation) `[repo]`；[MTP self-distill](https://github.com/jwkirchenbauer/mtp-lm) `[repo]`；[OPCD / LMOps](https://github.com/microsoft/LMOps) `[repo]`；[GATES](https://arxiv.org/abs/2602.20574) `[paper]`。
- [EMPO²](https://agent-lightning.github.io/posts/empo2/) `[page]` / [Agent Lightning code](https://github.com/microsoft/agent-lightning/tree/main/contrib/recipes/envs) `[repo]`；[CRISP / OPSDC](https://github.com/HJSang/CRISP_Reasoning_Compression) `[repo]`；[OEL / LMOps](https://github.com/microsoft/LMOps) `[repo]`。
- [Why Does Self-Distillation (Sometimes) Degrade Reasoning?](https://github.com/beanie00/self-distillation-analysis) `[repo]`；[Apple Embarrassingly Simple Self-Distillation](https://github.com/apple/ml-ssd) `[repo]`；[Skill-SD](https://skill-sd.github.io/) `[page]`；[SD-Zero](https://arxiv.org/abs/2604.12002) `[paper]`。
- [pi-Play](https://arxiv.org/abs/2604.14054) `[paper]`；[OPSDL](https://arxiv.org/abs/2604.17535) `[paper]`；[MSD](https://arxiv.org/abs/2605.02971) `[paper]`；[COPSD](https://github.com/cisnlp/COPSD) `[repo]`。
- [SGSD](https://github.com/walawalagoose/SGSD) `[repo]`；[CODE](https://github.com/CrashBugger/CODE) `[repo]`；[SSOPD](https://arxiv.org/abs/2605.17497) `[paper]`；[RLCSD](https://github.com/THU-BPM/RLCSD) `[repo]`；[d-OPSD](https://github.com/xingzhejun/d-opsd-code) `[repo]`。

### 8.5 迭代自举（前一 checkpoint 作教师）

- [SPIN](https://github.com/uclaml/SPIN) `[repo][strict?]`；[rStar / rStar-Math / rStar2-Agent](https://github.com/microsoft/rStar) `[repo][strict?]`。这类方法与“冻结外部教师在当前学生 rollout 上给 logits”的标准 OPD 不同，网站应单列。

### 8.6 OPD–RL 混合

- [BOND](https://arxiv.org/abs/2407.14622) `[paper][strict?]`；[Faster WIND](https://arxiv.org/abs/2410.20727) `[paper][strict?]`；[AlignDistil](https://github.com/songmzhang/AlignDistil) `[repo]`；[LUFFY](https://github.com/ElliottYan/LUFFY) `[repo][strict?]`；[KETCHUP](https://arxiv.org/abs/2504.19024) `[paper]`。
- [KDRL](https://arxiv.org/abs/2506.02208) `[paper]`；[SDPO](https://github.com/lasgroup/SDPO) `[repo]`；[KEPO](https://github.com/Corleno/KEPO) `[repo]`；[Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) `[repo][strict?]`；[Towards On-Policy SFT / DDT](https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT) `[repo][strict?]`。
- [X-KD](https://arxiv.org/abs/2602.12674) `[paper]`；[RLAD](https://arxiv.org/abs/2602.22495) `[paper]`；[OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) `[repo]`；[ExGRPO](https://github.com/Zhen-Tan-dmml/ExGRPO) `[repo][strict?]`；[HDPO](https://arxiv.org/abs/2603.23871) `[paper]`。
- [RLSD](https://arxiv.org/abs/2604.03128) `[paper]`；[NPO / AutoNPO](https://arxiv.org/abs/2604.20733) `[paper]`；[ROSD](https://arxiv.org/abs/2605.28014) `[paper]`；[TGPO](https://arxiv.org/abs/2605.13230) `[paper]`；[OPD+](https://arxiv.org/abs/2606.01039) `[paper]`。

### 8.7 Reasoning 交叉索引

- [G-OPD](https://github.com/RUCBM/G-OPD)（与白盒章节重复）；[OPD-AVMP](https://arxiv.org/abs/2604.07944) `[paper]`；[THUNLP Rethinking OPD](https://github.com/thunlp/OPD)（与基础/白盒章节重复）。README 说明 OPSD、rStar、RL hybrids、REOPOLD/Fast/Entropy/TIP/SCOPE/PACED 等 reasoning 项已在别处列出，故未重复铺表。

### 8.8 多模态

- [pi-Flow](https://github.com/Lakonik/piFlow) `[repo]`；[VOLD](https://arxiv.org/abs/2510.23497) `[paper]` / [项目页](https://walidbousselham.com/VOLD/) `[page]`（README 称 repo 是 placeholder）；[Step-Audio-R1](https://github.com/stepfun-ai/Step-Audio-R1) `[repo]`；[CORD](https://arxiv.org/abs/2601.16547) `[paper]`。
- [Video-OPD](https://arxiv.org/abs/2602.02994) `[paper]`；[X-OPD](https://arxiv.org/abs/2603.24596) `[paper]`；[Uni-OPD](https://github.com/WenjinHou/Uni-OPD) `[repo]`；[Flow-OPD](https://github.com/CostaliyA/Flow-OPD) `[repo]`；[Decomposed-OPD / VGS](https://github.com/hee-suk-yoon/Decomposed_OPD) `[repo]`。

### 8.9 Agent 与 embodied

- [LLM4Teach](https://github.com/ZJLAB-AMMI/LLM4Teach) `[repo][strict?]`；[Refined Policy Distillation / RPD](https://github.com/Refined-Policy-Distillation/RPD) `[repo]`；[easydistill / SCoRe](https://github.com/modelscope/easydistill) `[repo][strict?]`；[EMPO²](https://agent-lightning.github.io/posts/empo2/)（与 OPSD 重复）；[OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)（与 RL hybrid 重复）。
- [VLA-OPD](https://irpn-lab.github.io/VLA-OPD/) `[page]`（README 写 code coming soon，不能标已开源）；[Skill-SD](https://arxiv.org/abs/2604.10674)（与 OPSD 重复）；[TCOD](https://arxiv.org/abs/2604.24005) `[paper]`；[Healthcare AI GYM](https://arxiv.org/abs/2605.02943) `[paper]` / [项目仓库](https://github.com/minstar/Healthcare_GYM) `[repo]`；[HyperEyes](https://github.com/DeepExperience/HyperEyes) `[repo]`；[SDAR](https://github.com/ZJU-REAL/SDAR) `[repo]`。

### 8.10 Speculative decoding / draft model 蒸馏

- [Online Speculative Decoding](https://github.com/LiuXiaoxuanPKU/OSD) `[repo]`；[DistillSpec](https://arxiv.org/abs/2310.08461) `[paper]`；[HASS](https://github.com/HArmonizedSS/HASS) `[repo]`；[Falcon](https://github.com/Bestpay-inc/Falcon) `[repo]`。
- [CORAL](https://arxiv.org/abs/2502.16880) `[paper]`；[EAGLE / EAGLE-3](https://github.com/SafeAILab/EAGLE) `[repo]`；[MASSV](https://arxiv.org/abs/2505.10526) `[paper]`；[DVI](https://arxiv.org/abs/2510.05421) `[paper]`。
- [SpecKD / SelecTKD](https://arxiv.org/abs/2510.24021) `[paper]`（v1/v2 改名，教程要标版本）；[ReSpec](https://arxiv.org/abs/2510.26475) `[paper]`；[SpecForge](https://github.com/sgl-project/SpecForge) `[repo]`；[Draft-OPD](https://github.com/bingyang-lei/Draft-OPD) `[repo]`。

这组“on-policy”常指 drafter 在自己诱导的状态上学习，动作/验证机制与常规 LLM post-training 不同，应作为应用扩展而不是 vanilla OPD 主线证据。

### 8.11 Frameworks 与工具链

- [TRL](https://github.com/huggingface/trl) `[repo]`（GKD/GOLD/MiniLLM/SDFT/self-distillation 等实验 trainer）；[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) `[repo][strict?]`（README 明确：依赖 TRL 集成，没有原生 OPD trainer）；[ms-swift](https://github.com/modelscope/ms-swift) `[repo]`（封装 TRL GKD）。
- [verl](https://github.com/volcengine/verl) `[repo]`；[rllm](https://github.com/rllm-org/rllm) `[repo]`；[SkyRL](https://github.com/NovaSky-AI/SkyRL) `[repo]`；[ROLL](https://github.com/alibaba/ROLL) `[repo]`。
- [AReaL](https://github.com/inclusionAI/AReaL) `[repo][strict?]`；[slime](https://github.com/THUDM/slime) `[repo]`；[NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL) `[repo]`；[KDFlow](https://github.com/songmzhang/KDFlow) `[repo]`。

网站的工程主线优先级建议：`tinker-cookbook`（最小可读 recipe）→ `TRL`（单机/常规 trainer）→ `verl`（分布式 teacher/student/rollout）→ `KDFlow`（KD-first 与后端解耦）。不要把“框架 README 写支持 OPD”与“我们已验证该版本代码路径”混为一谈。

### 8.12 工业/生产报告

- [Gemma 2](https://arxiv.org/abs/2408.00118) `[report]` / [模型仓库](https://github.com/google-deepmind/gemma)；[Qwen3](https://arxiv.org/abs/2505.09388) `[report]` / [仓库](https://github.com/QwenLM/Qwen3)；[GLM-4.5/4.6](https://arxiv.org/abs/2508.06471) `[report][strict?]` / [官方仓库](https://github.com/zai-org/GLM-4.5)（AwesomeOPD 自己也注明报告未明确使用 OPD 术语）。
- [HY-MT](https://github.com/Tencent-Hunyuan/HY-MT) `[repo/report]`；[MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash) `[repo/report]`；[Typhoon-S](https://arxiv.org/pdf/2601.18129) `[report]`；[Baichuan-M3](https://github.com/baichuan-inc/Baichuan-M3-235B) `[repo/report]`；[GLM-5](https://github.com/zai-org/GLM-5) `[repo/report]`。
- [Nemotron Cascade 2](https://arxiv.org/abs/2603.19220) `[report]` / [NVIDIA 项目页](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)；[Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) `[repo/report]`；[KAT-Coder-V2](https://arxiv.org/abs/2603.27703) `[report]` / [产品页](https://streamlake.com/product/kat-coder)。
- [HY-Embodied-0.5](https://github.com/Tencent-Hunyuan/HY-Embodied) `[repo/report]`；[DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) `[report]` / [模型页](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)；[Qwen3.5-Omni](https://arxiv.org/abs/2604.15804) `[report]`；[Composer 2.5](https://cursor.com/cn/blog/composer-2-5) `[page]`（AwesomeOPD 写成 `Composor2.5`，应在网站纠正拼写）。

工业报告通常只公开配方描述和模型，不等于公开训练实现。网站应使用“报告称/作者描述”，并把“explicit OPD”“疑似相邻做法”“只有模型权重”分成不同徽标。

## 9. 快照之外、用户点名且必须补入网站的条目

这些条目在本地 AwesomeOPD 2026-06-23 主表中缺失，但均已由 arXiv 一手页面核验：

- [AOPD — arXiv:2605.06387v3](https://arxiv.org/abs/2605.06387v3)
- [Prune-OPD — arXiv:2605.07804v3](https://arxiv.org/abs/2605.07804v3)
- [TrOPD — arXiv:2606.01249v3](https://arxiv.org/abs/2606.01249v3)
- [IW-OPD / Position Bias — arXiv:2606.22600v3](https://arxiv.org/abs/2606.22600v3)
- [Formula-Driven Survey — arXiv:2606.22793v1](https://arxiv.org/abs/2606.22793v1)

## 10. 网站写作时的事实边界

1. 每个结果写成“作者在某模型、数据与评测设置中报告”，不要把单篇论文的消融提升改写为普遍定律。
2. 区分四个层次：目标散度、采样估计器、rollout 分布、工程实现。比如同叫 reverse KL，full-vocabulary RKL、teacher-top-k RKL 与 sampled-token stop-gradient advantage 并不是同一个优化对象。
3. “代码公开”至少要链接到核心 trainer/训练脚本；只有模型权重、评测脚本、项目页或 `code soon` 时使用不同状态。
4. 对 2026 年快速迭代论文使用无版本 arXiv 链接作为默认入口，同时在版本审计卡片中记录本次核验版本。
5. 对 Lightning OPD、BoN/RL hybrids、iterative bootstrapping、speculative decoding 等边界方法明确写 strictness note，避免把任何“on-policy + distillation”字样都当作同一种 OPD。

## 11. 一手来源总索引（重点条目）

- [GKD / On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- [Thinking Machines Lab: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [tinker-cookbook distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Revisiting OPD](https://arxiv.org/abs/2603.25562v2) / [official code](https://github.com/hhh675597/revisiting_opd)
- [G-OPD / ExOPD](https://arxiv.org/abs/2602.12125) / [official code](https://github.com/RUCBM/G-OPD)
- [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079)
- [SCOPE](https://arxiv.org/abs/2604.10688) / [official code](https://github.com/machine981/SCOPE)
- [Rethinking OPD](https://arxiv.org/abs/2604.13016) / [official code](https://github.com/thunlp/OPD)
- [Formula-Driven Survey](https://arxiv.org/abs/2606.22793)
- [IW-OPD](https://arxiv.org/abs/2606.22600)
- [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734) / [official code](https://github.com/siyan-zhao/OPSD)
- [AOPD](https://arxiv.org/abs/2605.06387)
- [Prune-OPD](https://arxiv.org/abs/2605.07804)
- [The Many Faces of OPD](https://arxiv.org/abs/2605.11182)
- [Veto](https://arxiv.org/abs/2601.07155)
- [TrOPD](https://arxiv.org/abs/2606.01249)
