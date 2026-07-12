# Tinker / GKD / PPO 基础调研（供学习站编写使用）

> 调研日期：2026-07-13（Asia/Shanghai）  
> 本地源码：`asset/tinker-cookbook-src`，提交 `b939b4f3c14502e84c60e05e560ef93b9da31baf`（2026-06-26），工作树核验为干净。  
> 范围：Thinking Machines Lab 的 OPD 博文与 Tinker recipe、GKD（arXiv:2306.13649v3 / ICLR 2024）、PPO（arXiv:1707.06347v2）。  
> 写作规则：下文把“源码事实”“论文/博文报告的实验结果”“根据公式作出的推导”分开陈述；实验数字均不是本站独立复现结果。

## 0. 先给初学者一张地图

OPD 最容易学混的地方，是把两个互相独立的选择揉成一个词：

1. **在哪些前缀（state）上训练？**
   - off-policy：固定数据、人工答案或教师生成轨迹上的前缀；
   - on-policy：学生当下自己生成的轨迹上的前缀。
2. **在每个前缀上，用什么分布差异训练？**
   - forward KL、reverse KL、JSD，或者别的 divergence。

所以“on-policy”不等于“reverse KL”。GKD 明确允许 `on-policy + forward KL`；Thinking Machines 的 Tinker recipe 选择的是 `on-policy + reverse KL 的采样估计 + policy-gradient/importance-sampling 更新`。[GKD Algorithm 1 与式 (4)](https://arxiv.org/pdf/2306.13649)；[Thinking Machines 博文 Implementation](https://thinkingmachines.ai/blog/on-policy-distillation/)。

### 0.1 最少知识前置

建议在正文进入 OPD 前，用一个短章节补齐：

- **自回归语言模型**：给定 prompt `x` 与已生成前缀 `y_<t`，模型输出下一 token 分布 `p(y_t | x, y_<t)`；整条序列概率是各步条件概率的乘积，log-prob 是各步 log-prob 的和。
- **softmax 与温度**：`p_i(T)=exp(z_i/T)/sum_j exp(z_j/T)`；`T` 高会变平、探索更多，`T` 低会变尖、趋近贪心。GKD 在论文第 2 节给出相同定义，并在训练时把学生温度设为 1。[GKD §2, PDF pp.1–2](https://arxiv.org/pdf/2306.13649)
- **teacher / student**：教师是固定的参考分布 `q`；学生 `p_θ` 的参数要更新。教师强不等于它在每个输入上都正确，蒸馏只保证逼近教师行为。
- **RL 视角**：prefix 是 state，下一 token 是 action，语言模型是 policy，一次 completion 是 trajectory/rollout。
- **log-prob、KL、Monte Carlo**：不会枚举期望时，可从期望所对应的分布采样并取平均估计。
- **stop-gradient**：采样出来的 token 被当作数据；训练不对离散采样操作反向传播。GKD 在式 (4) 后明确说明不穿过学生采样分布反传。[GKD §3.1](https://arxiv.org/pdf/2306.13649)
- **advantage 与 importance ratio**：advantage 决定某个采样 action 应被增大还是压低概率；`p_learner/q_sampler` 修正采样模型和当前训练模型之间的小偏差。[Tinker Importance Sampling 官方文档](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/)
- **SFT / LoRA / checkpoint**：SFT 用已知目标 token 做交叉熵；LoRA 只训练低秩适配参数；从 checkpoint 载入“权重”与恢复“权重 + optimizer state”不是一回事。

### 0.2 一句话直觉

- off-policy 蒸馏：看老师写的标准答案；优点是答案强、监督密，缺点是学生从未练习如何从自己的错误前缀恢复。
- on-policy RL：学生自己作答，最后只拿到对/错或标量奖励；状态匹配，但监督稀疏。
- OPD：学生自己作答，老师在学生真正走到的每个前缀上评价下一 token；状态匹配且监督密集。

这个“三分法”直接对应 Thinking Machines 博文的表：SFT 是 off-policy+dense，RL 是 on-policy+sparse，OPD 是 on-policy+dense。[博文 lines “On-policy distillation — best of both worlds”](https://thinkingmachines.ai/blog/on-policy-distillation/)

## 1. KL：方向、温度和 Tinker 的采样估计

令固定前缀 `s_t=(x,y_<t)` 上：

- 教师分布 `q(a|s_t)=π_teacher(a|s_t)`；
- 学生分布 `p_θ(a|s_t)=π_θ(a|s_t)`。

### 1.1 Forward KL

$$
D_{KL}(q\|p_θ)=\sum_a q(a|s_t)\log\frac{q(a|s_t)}{p_θ(a|s_t)}.
$$

期望在教师 `q` 下。忽略与 `θ` 无关的教师熵后，最小化它等价于最小化教师 soft targets 对学生的交叉熵。教师有概率质量、学生却给很低概率的区域会受到强罚，因此在容量不足时倾向 **mode-covering / mean-seeking**。GKD 把 `D_KL(P||Q)` 称作 forward KL，并用双峰分布示意 forward/reverse 的覆盖/择模差异。[GKD §2](https://arxiv.org/pdf/2306.13649)

若只从教师采样一条序列，再对采样 token 做 NLL/SFT，这在样本期望上仍对应 forward-KL 交叉熵，但有限样本会丢掉完整 soft distribution 的信息。Thinking Machines 博文也明确区分“完整 next-token 分布（logit distillation）”和“采样序列”。[博文开头 off-policy distillation 段落](https://thinkingmachines.ai/blog/on-policy-distillation/)

### 1.2 Reverse KL

$$
D_{KL}(p_θ\|q)=\sum_a p_θ(a|s_t)\log\frac{p_θ(a|s_t)}{q(a|s_t)}
=\mathbb E_{a\sim p_θ}\left[\log p_θ(a|s_t)-\log q(a|s_t)\right].
$$

期望在学生 `p_θ` 下。学生把质量放到教师认为很差的 token 上会被强罚；容量不足时它可以集中到教师的一个高质量 mode，因此常称 **mode-seeking**，代价是多样性可能下降。GKD 强调最优 divergence 与任务有关，不能把 reverse KL 写成无条件更优。[GKD §3.1 “Choice of Divergence”](https://arxiv.org/pdf/2306.13649)

“support”要用工程语言讲清：softmax 理论上通常给每个 token 非零概率，但某些知识、格式或推理路径在学生中可能低到几乎采不到。纯 reverse-KL OPD 只能从学生实际采到的 action 获得信号，因此很难凭空发现这种“有效 support”外的行为。Thinking Machines 的实验先做 off-policy SFT/mid-training，再做 OPD；GKD 也从已经 SFT、能生成足够质量序列的学生开始。[博文 pseudocode 后的初始化说明](https://thinkingmachines.ai/blog/on-policy-distillation/)；[GKD §3.1 Remark](https://arxiv.org/pdf/2306.13649)

### 1.3 为什么只查“被采中的 token”也能优化 reverse KL

这是 Tinker recipe 的核心推导。固定一个前缀，

$$
\nabla_θD_{KL}(p_θ\|q)
=\mathbb E_{a\sim p_θ}
\left[(\log p_θ(a)-\log q(a)+1)\nabla_θ\log p_θ(a)\right].
$$

又因为 score-function identity：

$$
\mathbb E_{a\sim p_θ}[\nabla_θ\log p_θ(a)]=0,
$$

常数 `+1` 是零期望 baseline，可以去掉。因此，采样 `a~p_θ`，令

$$
A(a)=-(\log p_{sample}(a)-\log q(a))=\log q(a)-\log p_{sample}(a),
$$

再做 policy-gradient 更新，就给出固定前缀上 reverse-KL 梯度的 Monte Carlo 估计。这里的推导是由 KL 与 score-function identity 得出；源码事实是：Tinker 恰好把 `sampled_logprobs - teacher_logprobs` 写成 reverse KL，再把负值加入 advantage：

- `incorporate_kl_penalty()` 的定义与注释：`asset/tinker-cookbook-src/tinker_cookbook/distillation/train_on_policy.py:53-71`；[固定提交源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L53-L71)
- 重建完整序列并批量查询教师 log-prob：同文件 `:72-85`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L72-L85)
- `reverse_kl = (sampled_logprobs - teacher_logprobs[1:]) * mask`：同文件 `:86-96`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L86-L96)
- `kl_advantages = -coef * reverse_kl`：同文件 `:101-108`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L101-L108)

注意两层“ratio”不要混淆：

- `log p_sample - log q_teacher` 是教师监督，决定 advantage；
- `exp(log p_learner - log p_sample)` 是 importance ratio，修正 learner 与 sampler 的偏差。

Tinker 的官方 loss 是：

$$
L_{IS}=-\sum_t \frac{p_{learner}(a_t|s_t)}{p_{sample}(a_t|s_t)}A_t.
$$

官方实现说明见 [Tinker Importance Sampling](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/)。本地 `train_step()` 把 `target_tokens / sampled logprobs / advantages` 交给该 loss，再执行 Adam：`asset/tinker-cookbook-src/tinker_cookbook/rl/train.py:278-362`；[固定提交源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/rl/train.py#L278-L362)。

### 1.4 为什么 `kl_discount_factor=0` 是一个刻意的近似

在自回归序列里，较早 token 会改变以后访问的所有前缀。若要让较早 action 为未来 KL 负责，应把后续 KL 作为 return-to-go 回传。Tinker 在 `kl_discount_factor>0` 时计算：

$$
G_t=\sum_{k\ge 0}\gamma^k(-KL_{t+k}).
$$

实现位于 `discounted_future_sum_vectorized()`：`asset/tinker-cookbook-src/tinker_cookbook/rl/metrics.py:192-211`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/rl/metrics.py#L192-L211)。默认 `0.0` 时，每个 token 只优化当下 next-token divergence，不把未来分歧归因给当前 token。博文称使用未来折扣“数学上更完整”，但作者在实验中未观察到收益；cookbook README 也作同样说明。[博文 Loss function](https://thinkingmachines.ai/blog/on-policy-distillation/)；[Tinker distillation 文档](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)。

### 1.5 温度必须拆成三件事

1. **学生 rollout 温度**：决定学生访问哪些前缀、采到哪些 token。GKD 训练采样用 `T=1` 以保持多样性；Tinker 底层 `Config.temperature` 默认也是 1.0，并在 rollout 处使用。`train_on_policy.py:133-152, 321-338`；[源码 Config](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L133-L152)、[源码 rollout](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L321-L338)。
2. **教师 softmax 温度**：决定 target distribution 尖锐程度。GKD 的 XSum 消融发现，当学生以 `T=1` 采样评估时，教师温度低于 1 有时更好；这只是任务相关实验观察，不是通则。[GKD Appendix A.3 与 Figure 7](https://arxiv.org/pdf/2306.13649)
3. **评估温度**：会改变 accuracy/pass@k/多样性，必须与训练温度分开报告。当前 Tinker README 的 AIME'24 数字使用 `temperature=1.0, top_p=1.0, top_k=-1, max_tokens=64000`。[官方 recipe 文档](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)

当前 Tinker OPD recipe 没有显式教师温度，也没有经典 KD 常见的 `T²` loss rescaling；它调用固定教师的 `compute_logprobs_async()`。不要在教材中凭空给它加上 `T²`。

更重要的是，**当前固定提交有一个 CLI 接线缺口**：`CLIConfig.temperature` 定义在 `on_policy_distillation.py:67-74`，但构造底层 `train_on_policy.Config` 的 `:153-176` 没有传 `temperature=cli_config.temperature`。因此默认实验仍是 1.0，但命令行改这个字段不会影响单轮 recipe；多轮 Harbor recipe 则显式传了它。证据：[单轮 CLI 字段](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L67-L82)、[单轮 Config 构造](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L153-L176)、[Harbor 正确传参](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/on_policy_distillation_harbor_multi_turn.py#L152-L173)。该结论只针对上述提交，网站应注明版本。

## 2. GKD（arXiv:2306.13649）到底提出了什么

### 2.1 核心形式

GKD 用 `λ∈[0,1]` 控制学生生成数据的比例：

$$
L_{GKD}=(1-λ)\,\mathbb E_{(x,y)\sim(X,Y)}D(q\|p_θ)(y|x)
+λ\,\mathbb E_{x\sim X,y\sim p_S(\cdot|x)}D(q\|p_θ)(y|x).
$$

- `λ=0`：完全 fixed-data / supervised；
- `λ=1`：完全 on-policy；
- `0<λ<1`：mixed；
- `D` 可选 forward KL、reverse KL、generalized JSD。

Algorithm 1 每一步先按 `λ` 决定用学生新生成输出还是固定 `(x,y)`，然后在这些序列的每个前缀上最小化 teacher/student token-distribution divergence，并明确不穿过采样反传。[GKD Algorithm 1, §3.1](https://arxiv.org/pdf/2306.13649)

这带来两个重要教学结论：

- **on-policy 解决的是 prefix/state distribution mismatch**；
- **reverse KL 解决的是容量不足时的 mode-covering vs mode-seeking 取舍**。

它们可以单独消融，不能互为定义。

### 2.2 GKD 与 Thinking Machines/Tinker 不是同一份 loss 实现

GKD 论文的 `D` 是访问到的前缀上 teacher/student **完整 token distribution** 的 divergence，直接对这个 divergence 求梯度；只对“生成这条前缀的离散采样过程”stop-gradient。Thinking Machines 的基础 recipe 没有取完整词表 logits，而只取学生采中 token 的 teacher log-prob，再用 policy-gradient/importance-sampling 形成 reverse-KL 的 Monte Carlo 梯度估计。博文明确说其实验没有做 top-k logit distillation。[GKD 式 (2)/(4)](https://arxiv.org/pdf/2306.13649)；[Thinking Machines Pseudocode](https://thinkingmachines.ai/blog/on-policy-distillation/)。

因此正文可写“二者在固定前缀上的目标相关”，但不应写“Tinker 就是 GKD Algorithm 1 的逐行实现”。特别是 GKD 的 forward KL/JSD 需要分布级信息，而这份 Tinker on-policy recipe 固定为采样 reverse-KL 信号。

### 2.3 GKD 的实证结论（严格限定设置）

论文使用 SFT 后的 T5 系列：T5-XL（约 3B）教师，以及 T5-small/base/large 学生；任务是 XSum、WMT、GSM8K，并非 Qwen 推理训练。作者报告：

- 相对“初始学生到蒸馏后”的增益，对不同学生规模平均，on-policy GKD 相比基线 KD 的性能增益倍数约为 XSum 2.1×、翻译 1.7×、算术 1.9×；另在 held-out BBH/MMLU 报告 2%/1% absolute improvement。[GKD Introduction contributions](https://arxiv.org/pdf/2306.13649)
- XSum 中，0.5% 训练数据上的 on-policy GKD（论文相应设置）可超过使用 100% 数据的 supervised KD/ImitKD；论文据此强调数据效率。[GKD Figure 1/5 与实验正文](https://arxiv.org/pdf/2306.13649)
- WMT 中，使用学生生成数据以及 mode-seeking divergence 往往更好；具体最佳组合随学生大小变化。[GKD Figure 6](https://arxiv.org/pdf/2306.13649)
- GSM8K 中，on-policy 方案整体较强；forward KL 也表现不错，而 reverse KL 在 T5-base/large 上超过 on-policy forward KL，说明“最佳 divergence 任务相关”。[GKD §4.3 / Figures 7–9](https://arxiv.org/pdf/2306.13649)
- reverse KL 对较高学习率更敏感；XSum 附录默认 reverse-KL LR 为 `3e-4`。这是 T5/Adafactor 设置，不能直接搬成 Qwen/Tinker 超参。[GKD Appendix A.3](https://arxiv.org/pdf/2306.13649)
- 学生在线采样有额外成本；GSM8K 中论文报告相对固定输出数据，随师生规模比不同约为 1.8×/2×/2.2×。论文认为收益可能值得该成本。[GKD Appendix A.2](https://arxiv.org/pdf/2306.13649)

### 2.4 GKD + RL

GKD 式 (5) 把序列标量奖励与 teacher divergence 组合：

$$
\mathbb E[(1-α)r(y)-αD(q\|p_θ)].
$$

`α=1` 是纯蒸馏；小于 1 时同时优化环境目标和保持教师能力。论文特别说明 RLHF/RLAIF 常用 reverse KL 约束初始 policy，因此若想最小改动接入 GKD，可考虑 reverse KL 或 JSD(0.9)。这不是说所有 OPD 都必须带环境 reward；Tinker 当前基础 recipe 的环境 reward 明确为零。[GKD §3.2](https://arxiv.org/pdf/2306.13649)。

## 3. 为什么要 on-policy，以及它解决不了什么

### 3.1 它解决 train–inference state mismatch

自回归生成时，`y_t` 成为下一步上下文。固定教师轨迹只覆盖教师常去的前缀；学生一旦早期犯错，就进入训练中未见过的前缀，后续误差会累积。GKD §3.1 把这个问题连接到 imitation learning；Thinking Machines 博文用“看棋谱 vs 老师点评你自己的每一步棋”作直觉类比。[GKD](https://arxiv.org/abs/2306.13649)；[Thinking Machines 博文](https://thinkingmachines.ai/blog/on-policy-distillation/)。

OPD 每轮都从最新学生采样，因此监督追着学生的 state distribution 移动。Tinker 的同步循环在每次更新后保存新权重并取得新的 sampling client，下一批 rollout 使用更新后的学生，见 `train_on_policy.py:299-338, 345-357`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L299-L357)。

### 3.2 它给出 token 级 dense signal

结果奖励常在整条 completion 结束后只有一个标量；OPD 对每个已生成 token 都有 `log p_student - log p_teacher`。博文报告在其 matched-architecture 实验中，蒸馏在少得多的梯度步数内重现 RL 教师；原因解释为每 episode 的监督从近似 `O(1)` bits 变为 `O(N)` token-level signal。这个 bits 表述和 7–10×/50–100×数字是作者在特定实验中的分析，不应写成定理。[博文 Discussion](https://thinkingmachines.ai/blog/on-policy-distillation/)。

### 3.3 它不能凭空创造知识或保证超越教师

- 学生从未有效采到的知识/行为，reverse-KL 采样估计几乎收不到信号，所以常需 SFT/mid-training 打底。
- 教师若系统性错误，KL 仍会让学生模仿错误；“low KL 不可 reward-hack”只表示对这个固定教师分布忠实，不等于对真实世界绝对正确。
- 学生容量不足时不可能完整复制教师；这正是 GKD 研究 divergence 选择的原因。
- on-policy 仍需要大量 student rollout 与 teacher scoring；它节省何种成本取决于模型大小、并行能力、上下文长度和是否已有离线教师数据。

## 4. Tinker Cookbook：从源码走完一轮

### 4.1 文件地图（固定提交）

| 作用 | 本地路径与关键行 | 固定 GitHub 链接 |
|---|---|---|
| 单轮 OPD CLI / 组装配置 | `tinker_cookbook/recipes/distillation/on_policy_distillation.py:50-181` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L50-L181) |
| OPD 核心训练与 KL | `tinker_cookbook/distillation/train_on_policy.py:53-514` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L53-L514) |
| prompt-only 环境/数据集 | `tinker_cookbook/distillation/datasets.py:32-276` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/datasets.py#L32-L276) |
| rollout 转 Datum、mask/shift | `tinker_cookbook/rl/data_processing.py:23-224` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/rl/data_processing.py#L23-L224) |
| importance-sampling 训练调用 | `tinker_cookbook/rl/train.py:265-362` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/rl/train.py#L265-L362) |
| 当前复现实验说明 | `tinker_cookbook/recipes/distillation/README.md:1-160` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/README.md#L1-L160) |
| reasoning SFT 初始化 | `tinker_cookbook/recipes/distillation/off_policy_reasoning.py:43-198` | [source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation/off_policy_reasoning.py#L43-L198) |

### 4.2 阶段 A：先做 off-policy SFT 初始化

当前 README 的 reasoning 路线先在 OpenThoughts3 上 SFT，再在 DeepMath prompts 上 OPD。`off_policy_reasoning.py`：

- 流式读取 `open-thoughts/OpenThoughts3-1.2M`：`:43-55`；
- 把 `human/gpt` 会话映射为 `user/assistant`，默认只训练 assistant messages：`:57-76`；
- 默认 rank-128 LoRA、batch 128、LR `1e-3`、最大序列 16384：`:90-109`；
- 最后调用通用 supervised `train.main`，不是 OPD RL loop：`:160-198`。

所以这里的复现“off-policy distillation”是教师生成 reasoning traces 上的 sequence-level hard-target SFT。仓库另有通用 `distillation/train_off_policy.py`，它在固定 SFT 数据的每个位置查询教师 top-k（默认 20）soft targets、在 top-k 内重归一化并做 cross-entropy：`train_off_policy.py:1-15, 87-110, 139-225, 329-337, 456-465`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_off_policy.py#L1-L15)。**README 的 reasoning 命令没有调用这个 soft-target trainer**，网站应避免混称。

### 4.3 阶段 B：构造无环境奖励的 prompt-only rollout

`PromptOnlyEnv` 的 `check_format()` 恒真、`check_answer()` 恒假，`step()` 总是返回 reward 0 且 episode 结束：`datasets.py:85-121`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/datasets.py#L85-L121)。这证明基础 recipe 没有混入正确性或格式奖励。

数据：

- DeepMath 读取 `question` 字段：`:180-189`；
- Tulu3 只取每条 conversation 的第一个 user message，因此基础 recipe 是单轮 prompt，不保留后续 turns：`:192-218`；
- prompt 默认最多 1024 tokenizer tokens：`:221-256`；
- 每个 prompt 建一个 group，`group_size` 决定独立 rollouts 数：`:158-174`。

环境 reward 全零，所以 `compute_advantages()` 的 group-centered reward advantage 也全零（`rl/data_processing.py:23-42`）。核心循环显式 `do_remove_constant_reward_groups=False`，否则这些 group 会被 RL 逻辑删掉；随后才把教师 KL 加进 token advantage（`train_on_policy.py:321-338, 191-224`）。因此在此 recipe 中 `group_size=4` 的主要意义是“同一 prompt 多采几条学生轨迹以覆盖/降方差”，不是 GRPO 式正负 reward 排序。

### 4.4 阶段 C：创建学生与固定教师

学生：

- 若从同一日志目录 resume，加载权重和 optimizer state：`train_on_policy.py:399-425`；
- 若给 `load_checkpoint_path` 作为新 run 初始化，只载入权重，optimizer 重新开始：`:426-434`；
- 否则创建 LoRA training client：`:435-438`。

教师：每个 dataset config 创建一个只推理的 `SamplingClient`；若给 teacher checkpoint，则以 `base_model + model_path` 创建。教师没有 backward/optimizer，训练期间固定：`:443-472`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/distillation/train_on_policy.py#L443-L472)。

一个工程前提是 teacher/student tokenization 与 renderer 必须可兼容。源码在构造 teacher 输入处明确提醒：教师 renderer 不同时，应改造 `full_sequence_inputs_D`（`:72-73`）。当前实现直接把学生 token 序列送给教师，因此不能随意混用 token ID 空间不同的模型。

### 4.5 阶段 D：rollout 变成 `Datum`

`trajectory_to_data()` 把 observation 与学生 action 拼起来：

- prompt/环境 observation token 的 sampled log-prob 填 0、mask 填 0；
- 学生生成 token 保存 rollout log-prob、mask 填 1；
- 再 right-shift input / left-shift targets，使每个位置预测下一 token。

证据：`rl/data_processing.py:95-190`，尤其 `:138-160, 176-185`；[源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/rl/data_processing.py#L138-L185)。因此只有学生真正生成的 token 接受 KL 梯度，prompt 不训练。

教师打分时，源码把 `model_input` 再追加最后一个 target，复原完整未 shift 序列；`compute_logprobs()` 返回的首 token 没有前文可预测，故使用 `teacher_logprobs[1:]` 与长度为 `N` 的 targets 对齐（`train_on_policy.py:74-96`）。这也是讲解代码时最容易漏掉的 off-by-one。

### 4.6 阶段 E：教师 dense feedback + 学生更新

忠于源码的一轮伪代码：

```python
# 0) sampler 是本轮开始时学生最新权重的快照
groups = dataset.get_batch(step)                  # prompts × group_size
trajectories = rollout(student_sampler, groups)   # tokens + sampled logprobs

data = assemble_training_data(trajectories)       # prompt mask=0, action mask=1
teacher_lp = teacher.compute_logprobs(full_student_sequences)[1:]

sample_log_ratio = sampled_lp - teacher_lp
data.advantages += -kl_coef * sample_log_ratio
if gamma > 0:
    data.advantages = discounted_future_sum(data.advantages, gamma)

# learner/sampler 可能略有偏差，loss 内使用 exp(learner_lp - sampled_lp)
forward_backward(data, loss_fn="importance_sampling")
optim_step(Adam(lr))
student_sampler = snapshot(updated_student_weights)
```

源码对应：组装和 KL 在 `prepare_minibatch():174-226`；训练调用在 `do_train_step_and_get_sampling_client():229-279`；同步外循环在 `do_sync_training():282-369`；每步更新后新 sampler 来自 `compute_full_batch_metrics_and_get_sampling_client()` 与 `save_checkpoint_and_get_sampling_client()`（`rl/train.py:1257-1292, 1356-1405`）。

有限 batch 的 `teacher_kl` 是采样 log-ratio 平均。真 KL 非负，但单 token 的 `log p-log q` 可以为负，有限样本均值也可能暂时小于零；不要把任意一个负 metric 判成“KL 数学错误”。若 rollout 温度/截断采样改变了采样分布，还需重新核对它估计的是哪个 tempered/truncated objective；当前推荐设置使用 `T=1`、不做 top-k/top-p 截断。

### 4.7 超参：代码默认与当前复现命令要分开

| 参数 | 单轮 CLI 默认 | 当前 README reasoning 复现 | 含义/注意 |
|---|---:|---:|---|
| student | `Qwen/Qwen3.5-9B-Base` | 同 | 先载入 README 给出的 SFT checkpoint |
| teacher | `Qwen/Qwen3.5-9B` | 同 | 固定 sampling client |
| `lora_rank` | 128 | 128 | blog 原实验模型已退役；不要混用旧结果 |
| `group_size` | 4 | 未覆盖，故 4 | 每 prompt rollout 数 |
| `groups_per_batch` | 1024 | 512 | prompts/update；复现即约 2048 trajectories/update |
| `learning_rate` | `1e-4` | `1e-4` LoRA；README 称 full FT `5e-5` | KL coefficient 与 LR 一起影响步幅 |
| `max_tokens` | 4096 | 16384 | OPD 可从部分 rollout 学，但任务结果会变 |
| prompt 上限 | 1024 | 1024（builder 默认） | 是 prompt 截断，不是 completion 上限 |
| rollout `temperature` | 实际底层默认 1.0 | 1.0 | 当前单轮 CLI 改值未接到底层，见前述版本缺口 |
| `kl_penalty_coef` | 1.0 | 1.0 | 唯一监督的缩放；设 0 后基础环境无训练信号 |
| `kl_discount_factor` | 0.0 | 0.0 | 0=只用立即 token KL；>0=未来 KL return-to-go |
| `loss_fn` | `importance_sampling` | 同 | 默认不是 PPO clipping |
| `num_substeps` | 1 | 1 | 把大 batch 切块并逐块 optimizer step，不是 PPO 的 K 轮重放 |
| eval/save cadence | 20/20 | 20/20 | `eval_every=0` 可关闭 |

CLI 定义见 `recipes/distillation/on_policy_distillation.py:50-99`；README 当前推荐命令和结果见 [Tinker Model Distillation 官方文档](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)。

## 5. 实验观察：三组证据不要拼成一张表横比

### 5.1 Thinking Machines 2025 博文（原 Qwen3 设置）

博文原 reasoning 实验用 Qwen3-8B-Base 学生与 Qwen3 系列教师；页面在 2026 年 6 月更新说明这些模型已从 Tinker 支持列表退役，cookbook 已改成 Qwen3.5-9B-Base / Qwen3.5-9B。因此以下是原博文设置，不是当前 README 命令的逐步复现：

- 400k off-policy SFT prompts 后约 60% AIME'24；OPD 约 150 steps（约 77k prompts、每 prompt 4 samples）到 70%。作者按其 FLOPs 口径估计相对继续 SFT 有 9–30× 成本收益。[博文 Distillation for reasoning](https://thinkingmachines.ai/blog/on-policy-distillation/)
- 从同一 base 出发，先用 RL 得到教师、再把该策略蒸回 base 的 matched experiment 中，作者报告 OPD 约少 7–10× gradient steps；连同较短 context 与较小 batch，估算 50–100× compute efficiency。这个结果高度依赖“教师策略处于学生 initialization 的有效 support 内”。[博文 Dense supervision](https://thinkingmachines.ai/blog/on-policy-distillation/)
- personalization 实验中，OPD 把 IF-eval 从 midtrain 后的 79% 恢复到 83%，原模型为 85%，同时 internal QA 从 36% 到 41%；这是内部数据设置，外部无法仅凭文章完整复现。[博文 Personalization](https://thinkingmachines.ai/blog/on-policy-distillation/)
- 重复使用单一 prompt 的实验表明 OPD 可以持续从同一 prompt 的新 rollouts 学教师分布；它不等于说所有任务一个 prompt 都足够。[博文 Data efficiency](https://thinkingmachines.ai/blog/on-policy-distillation/)

### 5.2 当前 Tinker cookbook（Qwen3.5 设置）

README/官方文档报告：

- rank-128 LoRA 在 OpenThoughts3 SFT 3000 steps 后，AIME'24 约 65%，LoRA LR `1e-3`；
- 从该 checkpoint 出发，在 DeepMath 上 OPD 200 steps、16k-token rollouts 后约 76.7%，LoRA LR `1e-4`、`groups_per_batch=512`；
- AIME 评估为 `temperature=1, top_p=1, top_k=-1, max_tokens=64000`；
- personalization 中 IF-eval 约 100 steps 恢复，但使用者需提供自己的 SFT 初始化。

来源：[官方 Model Distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)；本地同文 `recipes/distillation/README.md:9-80`。措辞应保持“作者观察到”，不要写成本项目已经复现。

### 5.3 如何正确监控

至少同时画：

- held-out task metric（AIME/IF-eval 等）；
- `teacher_kl` 的趋势；
- rollout length / stop reason；
- learner–sampler KL 或 importance ratio 稳定性；
- token 数、prompts、trajectories、teacher forward FLOPs 与 student training FLOPs。

只看 training `teacher_kl` 会有两类误判：同一小 prompt 上可以很快降 KL 却不泛化；截短或低温 rollout 可能让访问状态变窄，从而“更容易”得到低 KL。

## 6. OPD 与 PPO：像在哪里，不像在哪里

### 6.1 PPO 原论文的最小核心

PPO 从旧策略 `π_old` 采样，计算 advantage，再多轮优化 surrogate。PPO-Clip：

$$
L^{CLIP}=\mathbb E_t\left[\min(r_t(θ)\hat A_t,
\operatorname{clip}(r_t(θ),1-ε,1+ε)\hat A_t)\right],
\quad r_t=\frac{π_θ(a_t|s_t)}{π_{old}(a_t|s_t)}.
$$

原论文给出示例 `ε=0.2`；Actor-Critic 版本还包括 value loss 和 entropy bonus，并在 Algorithm 1 中每次由 `N` 个 actors 收集 `T` steps，再以 minibatch 做 `K` epochs。[PPO paper §3, §5, Algorithm 1](https://arxiv.org/pdf/1707.06347)；[arXiv abstract](https://arxiv.org/abs/1707.06347)。

### 6.2 对照表

| 维度 | Tinker 基础 OPD | 原始 PPO |
|---|---|---|
| state/action | prefix / next token | 一般环境 state/action；LLM 中也可映射为 prefix/token |
| 数据 | 最新学生 rollout，逐批刷新 sampler | `π_old` 与环境交互收集 on-policy data |
| advantage 来源 | `log q_teacher - log p_sample`，逐 token dense；环境 reward=0 | reward/return 减 baseline，常由 value/GAE 估计 |
| 更新 ratio | `p_learner/p_sampler` importance correction | `π_θ/π_old` |
| 默认 surrogate | 未裁剪 importance-sampling | clipped surrogate 是论文主方案 |
| teacher | 必需固定教师 log-prob | 不需要教师；目标来自环境 reward |
| KL 的对象 | 学生对固定教师，是训练目标/监督 | 论文的 KL constraint/penalty 是新旧策略的 trust region |
| critic/value | 无 | actor-critic 版本通常有 value loss |
| entropy | 无显式 entropy bonus；采样温度控制 rollout diversity | 可在目标加入 entropy bonus |
| credit assignment | 默认即时 token KL；可折扣累计未来 KL | return/GAE 处理跨时刻奖励 |
| 同一数据更新 | 默认每 datum 一次，`num_substeps` 是切大 batch | `K` epochs 重用同批数据，clip 限制过大偏移 |

### 6.3 三个必须避免的误解

1. **“用了 advantage 就是 PPO”——错。** Tinker 默认 `loss_fn="importance_sampling"`；源码虽然允许把 `loss_fn` 换成 `"ppo"`，但默认没有 clip。`on_policy_distillation.py:79-82, 153-166`；`train_on_policy.py:149-156, 257-266`。
2. **“OPD 的 teacher KL 就是 PPO 的 KL penalty”——错。** teacher KL 拉向一个固定目标行为；PPO/TRPO 的 KL 用来限制本次更新相对旧策略的步长。PPO 论文 §2–4 明确其 KL 对象是 `π_old` 与 `π_θ`。[PPO paper](https://arxiv.org/pdf/1707.06347)
3. **“RL 天生就是 reverse KL”——需要限定。** 带 entropy/KL regularization 的最大熵 RL 可改写成逼近 reward-tilted target distribution 的 reverse KL；但裸 PPO 论文的 clipped surrogate 本身不是“对教师做 reverse KL”。Thinking Machines 博文的说法是“RL generally optimizes a form of sequence-level reverse KL induced by the reward model”，教学时应保留 `generally / a form of / induced by reward` 这些限定。[博文 Loss function](https://thinkingmachines.ai/blog/on-policy-distillation/)

### 6.4 如果把 Tinker `loss_fn` 改成 `ppo`

这会得到“PPO-style clipped optimizer + teacher-derived dense advantage”，而不是把 OPD 变成普通结果奖励 RL。clip 的对象仍是 learner/sampler ratio，教师信号仍来自 `log q_teacher-log p_sample`。它可能在一批数据做多步更新或 sampler/learner 偏移大时更稳，但是否更好需要实验；当前 README 结果使用 importance sampling，不能把 PPO-clip 的收益写成已验证事实。[Tinker Loss Functions 官方文档](https://tinker-docs.thinkingmachines.ai/tinker/losses/)。

## 7. 建议的网站教学顺序

1. 用一个三 token toy distribution 手算 softmax、forward/reverse KL；强调 KL 不对称。
2. 用“固定前缀”和“谁生成前缀”画二维表，先消除 on-policy=reverse-KL 的误会。
3. 从 SFT 的 teacher-forcing 讲 exposure bias，再展示学生早期错误如何改变后续 state。
4. 推导 `∇ KL(p||q)` 的采样 estimator，专门解释为什么没有 `+1`。
5. 逐行走 Tinker：zero-reward env → rollout logprobs → mask/shift → teacher `[1:]` 对齐 → KL advantage → IS loss → 刷新 sampler。
6. 再讲 GKD 的 `λ` 与 divergence 两个旋钮，说明它比 Tinker 基础 recipe 更一般。
7. 最后对比 PPO：相同的训练骨架，不同的监督来源与稳定化手段。
8. 实验章节把“原博文旧 Qwen3”“当前 Qwen3.5 cookbook”“GKD T5”分成三张卡，不横向拼数字。

可设计的互动检查题：

- `q_teacher(token)=0.01, p_student(token)=0.20` 时，`A=log q-log p` 正还是负？模型应提高还是压低该 token？
- 为什么 prompt mask 必须为 0？
- 若 `group_size=1`，基础 Tinker OPD 是否仍有 KL 梯度？答案是有，因为 KL 在 group reward centering 后另行加入。
- 为什么 `teacher_kl` 的一次有限样本估计可能为负，但真 KL 不会？
- `temperature<1` 使 training KL 更低，是否必然说明泛化更好？答案是否定的，可能只是访问状态变窄。
- 把 `loss_fn` 从 IS 改成 PPO 后，teacher 是否消失？不会。

## 8. 常见错误清单（写作审校用）

- 不要把 GKD 论文标题中的 on-policy 与 reverse KL 绑定；论文系统消融 forward/reverse/JSD。
- 不要说 Tinker 当前基础 OPD 查询完整 teacher logits；它只查学生已采 token 的 log-prob。仓库另有 top-k off-policy/SDFT 代码，那是不同路径。
- 不要把 `groups_per_batch` 当 trajectories 数；实际约为 `groups_per_batch × group_size`。
- 不要说基础 environment 有 math correctness reward；源码明确恒为 0。
- 不要说 `group_size` 用于正负 reward 相对排序；此 recipe 的环境 rewards 都相同，KL 后加。
- 不要把 `kl_penalty_coef` 解释成限制新旧学生的 PPO trust region；它缩放固定教师监督。
- 不要忽略 tokenizer/renderer 兼容性；代码直接把学生 token IDs 送给教师。
- 不要把 README 的 76.7% 写成我们独立复现，也不要与博文旧 Qwen3 的 70% 直接比较。
- 不要把 GKD 的 T5 学习率照抄给 Tinker Qwen/LoRA。
- 不要声称 OPD 必然超过教师；目标是匹配教师在学生访问状态上的分布。
- 不要省略当前单轮 `temperature` CLI 未接到底层这一版本事实。

## 9. 一手资料与可继续阅读的官方实现

- Thinking Machines Lab, **On-Policy Distillation**（含 2026-06 模型退役更新）：<https://thinkingmachines.ai/blog/on-policy-distillation/>
- Thinking Machines Lab, **Tinker Cookbook distillation recipe**：<https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation>
- 本次核验固定提交：<https://github.com/thinking-machines-lab/tinker-cookbook/tree/b939b4f3c14502e84c60e05e560ef93b9da31baf/tinker_cookbook/recipes/distillation>
- Tinker 官方 **Model Distillation** 文档：<https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/>
- Tinker 官方 **Importance Sampling** 数学与输入字段：<https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/>
- Agarwal et al., **On-Policy Distillation of Language Models / GKD**, ICLR 2024：<https://arxiv.org/abs/2306.13649>；[PDF](https://arxiv.org/pdf/2306.13649)；[Google DeepMind publication page](https://deepmind.google/research/publications/48050/)
- Schulman et al., **Proximal Policy Optimization Algorithms**：<https://arxiv.org/abs/1707.06347>；[PDF](https://arxiv.org/pdf/1707.06347)
- 可运行的后续 GKD 工程参考（不是原论文官方代码）：Hugging Face TRL `GKDTrainer`，<https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md>

