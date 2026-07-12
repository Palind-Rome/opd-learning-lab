# verl 原生 OPD 源码研究报告

> 面向网站作者与第一次阅读 LLM post-training 工程代码的学习者。本文只以本地 `asset/verl` 快照中的源码、配置、测试和文档为证据；没有把论文结论反向“套”进代码。

## 0. 审计范围、版本与结论先行

- 本地仓库：`asset/verl`
- Git remote：`https://github.com/verl-project/verl.git`
- 审计 commit：`1ff76cc625e9820d2434dad1b6d9b8e5dd26a359`
- commit 时间：`2026-06-26 23:35:06 -0700`
- verl 自带 OPD 文档更新时间：`docs/algo/opd.md:3-5`，2026-05-26
- 固定版本源码链接前缀：`https://github.com/verl-project/verl/blob/1ff76cc625e9820d2434dad1b6d9b8e5dd26a359/`

### 一句话理解 verl 的 OPD

学生先在自己的状态分布上生成 response；冻结的 teacher 不重新生成答案，而是把 `prompt + student response` 当作完整 prefix 做一次 prompt-logprob 前向；teacher 给出每个位置的“已采样 token 概率”或“teacher top-k 分布”；actor 随后在相同状态上用直接蒸馏（GKD）或 PPO 形式的策略梯度蒸馏（PG-OPD）更新。

### 当前快照中最重要的六个事实

1. **默认执行路径已经是 V1 trainer。** `verl/trainer/config/ppo_trainer.yaml:200-207` 令 `trainer.use_v1=true`、`trainer_mode=sync`；`verl/trainer/main_ppo.py:153-179` 因而默认进入 `TaskRunnerV1`，不是旧版 `RayPPOTrainer.fit()`。网站不应只照抄旧架构图。
2. **teacher 使用独立 GPU resource pool。** actor/rollout/critic/ref 在 `global_pool`，teacher 在 `teacher_pool`；见 `verl/trainer/ppo/v1/trainer_base.py:572-626`。
3. **teacher query 是逐样本异步发出的。** 一个样本 rollout 完成后就发 teacher 请求，不等整个 batch；V1 路径见 `verl/trainer/ppo/v1/agent_loop_tq.py:130-150`。但一个同步训练 step 最终仍会等待够一批数据再更新。
4. **同一 loss 函数被调用两次。** top-k 模式先在 full logits 尚在显存时把 `distillation_ppo_loss` 当 logits processor，算出逐 token 稀疏 KL；随后再把它当最终 loss 聚合并反传；见 `verl/trainer/distillation/losses.py:165-222`。
5. **原生实现同时支持两条算法线。** `forward_kl_topk + use_policy_gradient=false` 是分布级 GKD；`k1 + use_policy_gradient=true` 是 Thinking Machines 风格的 PG-OPD。单样本 `k3` 也可以直接反传，得到一种不需要 top-k 的 GKD 梯度估计。
6. **“纯 OPD”仍走 PPO 数据管线。** 即便 `use_task_rewards=false`，代码仍计算 reward、old logprob、advantage 和 `ppo_loss`，只是最后把 policy loss 置零；见 `verl/trainer/ppo/v1/trainer_base.py:422-456` 与 `verl/trainer/distillation/losses.py:207-220`。这是理解依赖与性能开销的关键。

---

## 1. 源码地图：先看哪些文件

| 层次 | 文件与关键位置 | 作用 |
|---|---|---|
| Hydra 总配置 | `verl/trainer/config/ppo_trainer.yaml:7-47` | 把 `distillation/distillation.yaml` 挂到顶层 `distillation.*` |
| OPD YAML 默认值 | `verl/trainer/config/distillation/distillation.yaml:13-113` | 运行 `main_ppo` 时真正可见的默认值 |
| 配置 dataclass | `verl/workers/config/distillation.py:31-320` | loss、teacher、资源池约束与启动前校验 |
| 入口选择 | `verl/trainer/main_ppo.py:98-179` | V1/legacy trainer 分流 |
| V1 trainer | `verl/trainer/ppo/v1/trainer_base.py:101-456` | 初始化、一步训练的总控制流 |
| 同步权重更新 | `verl/trainer/ppo/v1/trainer_sync.py:24-42` | 每 step 后把 actor 权重同步给 rollout |
| V1 agent-loop 适配 | `verl/trainer/ppo/v1/agent_loop_tq.py:47-237` | rollout、teacher query、结果写入 TransferQueue |
| 通用 agent loop | `verl/experimental/agent_loop/agent_loop.py:481-521,1000-1023` | teacher manager 初始化与逐样本 teacher dispatch |
| teacher 请求 | `verl/experimental/teacher_loop/teacher_manager.py:30-128` | 路由、sampling params、teacher 输出张量化 |
| teacher 服务编排 | `verl/experimental/teacher_loop/teacher_model.py:37-203` | teacher replicas、子资源池、load balancer |
| actor worker | `verl/workers/engine_workers.py:434-657` | OPD 开关、绑定 loss、actor infer/update RPC |
| loss 总入口 | `verl/trainer/distillation/losses.py:39-394` | loss registry、GKD/PG 分流、mask/聚合、指标 |
| FSDP top-k | `verl/trainer/distillation/fsdp/losses.py:26-149` | dense/chunked log-softmax、teacher-top-k KL |
| Megatron top-k | `verl/trainer/distillation/megatron/losses.py:27-312` | TP-aware log-softmax、自定义 forward/backward |
| FSDP forward | `verl/workers/engine/fsdp/transformer_impl.py:945-1325` | causal shift、SP 切分/聚合、inline top-k processor |
| Megatron forward | `verl/workers/engine/megatron/transformer_impl.py:844-1001` | THD/BSHD、CP/TP 路径、inline top-k processor |
| VeOmni fused top-k | `verl/workers/engine/veomni/transformer_impl.py:847-900` | 把 teacher top-k 送进 chunked fused lm-head |
| padding/mask 对齐 | `verl/workers/utils/padding.py:23-143,196-230` | padded↔jagged、response 左移一位 |
| vLLM teacher 输出 | `verl/workers/rollout/vllm_rollout/utils.py:447-480` | 跳过首 token None，末尾补 dummy |
| SGLang teacher 输出 | `verl/workers/rollout/sglang_rollout/async_sglang_server.py:62-109,582-667` | 转成与 vLLM 一致的 contract |
| 官方示例 | `examples/on_policy_distillation_trainer/` | FSDP、Megatron、VeOmni、VL、MOPD 启动脚本 |
| 原生 OPD 文档 | `docs/algo/opd.md` | 算法、配置、指标和架构说明 |

### 默认 V1 与 legacy 的关系

`main_ppo` 的分流非常明确：

```text
python -m verl.trainer.main_ppo
  └─ trainer.use_v1 == true（默认）
       └─ TaskRunnerV1
            └─ PPOTrainerSync（默认 trainer_mode）
                 └─ TransferQueue + AgentLoopWorkerTQ

  └─ trainer.use_v1 == false
       └─ main_ppo_v0.TaskRunner
            └─ RayPPOTrainer（已标 deprecated）
```

证据：

- V1 默认：`verl/trainer/config/ppo_trainer.yaml:200-207`
- 分流与 deprecated 警告：`verl/trainer/main_ppo.py:170-179`
- legacy 构造 `RayPPOTrainer`：`verl/trainer/main_ppo_v0.py:224-241`
- legacy 也会创建 teacher manager 并把 client 交给 agent loop：`verl/trainer/ppo/ray_trainer.py:913-953`

因此，面向当前快照的教学正文应以 V1 为主；`ray_trainer.py` 可放在“旧版/兼容路径”折叠块中。

---

## 2. 配置：YAML 默认值、dataclass 默认值与启动前校验

### 2.1 网站应展示 YAML 默认值，而不是只看 dataclass

两处默认值并不完全一致：

| 字段 | Hydra YAML 默认 | dataclass 默认 | 证据 |
|---|---:|---:|---|
| `loss_mode` | `k3` | `k3` | YAML `:23-24`；dataclass `:65` |
| `topk` | `32` | `128` | YAML `:26-27`；dataclass `:66` |
| `use_task_rewards` | `true` | `True` | YAML `:29-30`；dataclass `:67` |
| `loss_max_clamp` | `null` | `10.0` | YAML `:36-37`；dataclass `:69` |
| `log_prob_min_clamp` | `null` | `-10.0` | YAML `:39-40`；dataclass `:70` |
| `use_policy_gradient` | `false` | `True` | YAML `:42-43`；dataclass `:84` |
| teacher `n_gpus_per_node` | `8` | `0` | YAML `:58-59`；dataclass `:263` |
| teacher `nnodes` | `0` | `0` | YAML `:61-62`；dataclass `:264` |

正常执行 `python -m verl.trainer.main_ppo` 时，Hydra 先加载 YAML，再通过 `_target_` 转成 dataclass，所以“用户可见默认值”应以 YAML 为准。直接在 Python 中写 `DistillationLossConfig()` 或测试中手工实例化时才会落到 dataclass 默认。

`use_chunked_topk` 与 `chunked_topk_chunk_size` 只出现在 dataclass（`verl/workers/config/distillation.py:72-82`），没有写进当前 YAML。若从 Hydra CLI 开启，通常需要新增键，例如：

```bash
+distillation.distillation_loss.use_chunked_topk=true \
+distillation.distillation_loss.chunked_topk_chunk_size=4096
```

### 2.2 loss 配置的合法组合

`DistillationLossConfig.__post_init__` 位于 `verl/workers/config/distillation.py:100-124`：

- `policy_loss_mode != vanilla`：直接 `NotImplementedError`。
- `use_policy_gradient=true + forward_kl_topk`：只打印 warning，不阻止运行；源码明确说 top-k 分布信号大部分会被浪费。
- `use_policy_gradient=false + k1`：直接 `ValueError`，因为直接反传 k1 的梯度不依赖 teacher logprob。

loss registry 在 `verl/trainer/distillation/losses.py:46-106,294-394`：

- top-k 家族：`forward_kl_topk`，`use_topk=True`
- 单样本估计器：`kl`、`k1`、`abs`、`mse`、`k2`、`low_var_kl`、`k3`，`use_estimator=True`

每个 loss 必须且只能属于一个家族，约束见 `DistillationLossSettings.__post_init__`（`:63-68`）。

### 2.3 teacher 资源与模型配置

核心 dataclass：

- `DistillationTeacherModelConfig`：`verl/workers/config/distillation.py:127-218`
- `DistillationConfig`：`verl/workers/config/distillation.py:221-320`

单个 teacher replica 的 GPU 数：

$$
W_{\text{replica}}
=
\text{TP}\times\text{DP}\times\text{PP}.
$$

代码：`per_replica_world_size`，`verl/workers/config/distillation.py:152-162`。

整个 teacher pool 必须严格满足：

$$
\sum_j n^{(j)}_{\text{replica}}W^{(j)}_{\text{replica}}
=
\text{distillation.n\_gpus\_per\_node}
\times
\text{distillation.nnodes}.
$$

否则 `DistillationConfig.__post_init__` 在 `:274-287` 抛错。

#### 单 teacher

- YAML 里默认有一个占位项 `teacher_models.teacher_model`：`verl/trainer/config/distillation/distillation.yaml:65-110`。
- 只配置这一个 teacher 时，代码自动令其占满整个 teacher pool：`num_replicas = pool_size // per_replica_world_size`，并把路由 key 设为 `default`；见 `verl/workers/config/distillation.py:289-307`。
- pool size 必须能整除单 replica world size，否则启动前报错（`:300-305`）。

#### 多 teacher

- 一旦 `teacher_models` 长度大于 1，默认占位项 `teacher_model` 会被 `pop`：`verl/workers/config/distillation.py:308-310`。
- 因而不要把真正的第一个 teacher 仍命名为 `teacher_model`，再额外添加第二个；前者会被静默丢掉。官方配置注释也专门警告：`verl/trainer/config/distillation/distillation.yaml:67-81`。
- 每个 teacher 必须显式给 `key`、`model_path`、`num_replicas`；见 `check_configured()`，`verl/workers/config/distillation.py:164-170`。
- 最终 dict 不再以 YAML entry name 为 key，而是以业务路由 key 为 key；重复 key 报错：`:312-319`。

### 2.4 teacher context 长度会被重写

teacher 不是再生成完整 response，而是对学生已经生成的 `prompt + response` 评分。因此启动时：

```python
required_context_len = student_prompt_length + student_response_length + 1
teacher.inference.prompt_length = prompt_length + response_length
teacher.inference.response_length = 1
```

源码：`verl/workers/config/distillation.py:172-186`。

`+1` 是因为 backend 仍被要求生成一个 dummy token。若显式 `max_model_len` 小于所需长度，会在启动前报错。

### 2.5 teacher inference backend 与 top-k 上限

top-k 模式的校验位于 `verl/workers/config/distillation.py:188-218`：

- vLLM：若 `engine_kwargs.vllm.max_logprobs` 未设，自动设成 `topk`；若显式值小于 `topk`，报错。
- SGLang：top-k 是 per-request 参数，没有 vLLM 那种 boot-time cap。
- 其他 engine：top-k 模式直接 `NotImplementedError`。

teacher YAML 默认继承 student rollout 的 engine、dtype、prompt/response 长度和 temperature，见 `verl/trainer/config/distillation/distillation.yaml:87-110`。

---

## 3. 一次同步训练 step 的完整数据流

下面是**当前默认 V1 + sync** 的真实路径。

### 3.1 初始化阶段

1. `main_ppo` 校验总配置并选择 `TaskRunnerV1`：`verl/trainer/main_ppo.py:153-179`。
2. `TaskRunnerV1.run()` 初始化 TransferQueue、构造 trainer、构造 agent-loop manager，再调用 `fit`：`verl/trainer/main_ppo.py:129-150`。
3. `PPOTrainer._setup()` 创建 actor/critic worker group，初始化 reward loop、teacher manager、student rollout server 与 checkpoint manager：`verl/trainer/ppo/v1/trainer_base.py:152-283`。
4. OPD 开启时，actor worker 接收 `distillation_config`：`verl/trainer/ppo/v1/trainer_base.py:160-169`。
5. `ActorRolloutRefWorker.init_model()` 将 actor loss 从普通 `ppo_loss` 换成绑定好配置的 `distillation_ppo_loss`：`verl/workers/engine_workers.py:544-590`。
6. `MultiTeacherModelManager` 在独立 teacher pool 上起 teacher replicas，并为每个 teacher 建立独立 `GlobalRequestLoadBalancer`：`verl/experimental/teacher_loop/teacher_model.py:154-203`。
7. teacher client dict 被交给 `AgentLoopWorker`；worker 再构造 `AsyncTeacherLLMServerManager`：`verl/experimental/agent_loop/agent_loop.py:491-521`。

### 3.2 step 控制流

`PPOTrainer.step()` 的顺序见 `verl/trainer/ppo/v1/trainer_base.py:406-458`：

```text
1. _add_batch_to_generate()
2. replay_buffer.sample() 等够一个训练 batch
3. 可选：colocated reward
4. 按 DP 工作量重排/补齐 batch
5. 计算 old_log_probs
6. 可选：reference log_probs
7. 可选：critic values
8. 计算 task advantage/returns
9. 可选：更新 critic
10. 更新 actor（包含 OPD）
```

注意：teacher query 不在第 10 步才开始；它已经嵌在第 1 步的 agent loop 里，和其他样本仍在进行的 rollout 重叠。

### 3.3 每个样本的 rollout 与 teacher query

`AgentLoopWorkerTQ.generate_sequences()`：

- 从配置读取 student sampling params：`verl/trainer/ppo/v1/agent_loop_tq.py:54-72`
- 每个样本建立一个 asyncio task：`:80-100`
- 按 `rollout.n` 再 fan-out 多条 trajectory：`:102-125`

样本完成后进入 `_agent_loop_postprocess()`：

1. 先计算 reward score：`verl/trainer/ppo/v1/agent_loop_tq.py:140`
2. 对最后一个 output 发 teacher query：`:142-150`
3. 把 prompt/response 拼成 `input_ids`，构造 `position_ids`：`:163-183`
4. 令 `loss_mask = response_mask`：`:178-180`
5. 把结果写入 TransferQueue：`:202-207`

teacher 调用本身位于 `AgentLoopWorker._compute_teacher_logprobs()`：

```python
if distillation_enabled and not validate:
    route = sample[teacher_key]
    teacher_ids, teacher_logprobs = await teacher_manager(
        sequence_ids=prompt_ids + response_ids,
        multimodal_data=...,
        routing_key=route,
    )
```

源码：`verl/experimental/agent_loop/agent_loop.py:1000-1023`。

由此可知：

- validation 不查 teacher，不计算 OPD validation loss；只做任务评估。
- teacher 看到的是学生实际走过的完整 state trajectory。
- 多模态图片/视频/音频与 `mm_processor_kwargs` 会一并传给 teacher。

### 3.4 teacher 请求到底做什么

`_get_teacher_sampling_params()` 返回：

```python
{
    "max_tokens": 1,
    "temperature": 1.0,
    "prompt_logprobs": K if topk_loss else 0,
}
```

源码：`verl/experimental/teacher_loop/teacher_manager.py:30-43`。

- top-k loss：请求每个输入位置 teacher top-k 的 token id 与 logprob。
- estimator loss：`prompt_logprobs=0`，只要学生实际采样 token 在 teacher 下的 logprob。
- teacher 只生成 1 个无用 token；真正的监督来自整个输入 prefix 的 prompt logprobs。

teacher 请求经 `LLMServerClient.generate()`：

- 用 Ray RPC 向全局 load balancer 原子申请 least-loaded server：`verl/workers/rollout/llm_server.py:189-197`
- 远程调用选中的 server：`:222-249`
- 每个 teacher 各自有一组 server 和一个 load balancer：`verl/experimental/teacher_loop/teacher_model.py:146-203`

teacher manager 最终把 backend 的 list 转成：

```text
teacher_ids:      [sequence_length, 1 or K]
teacher_logprobs: [sequence_length, 1 or K]
```

并断言第一维等于 `len(prompt_ids + response_ids)`；见 `verl/experimental/teacher_loop/teacher_manager.py:102-128`。

### 3.5 TransferQueue 中的关键字段

actor update 前，一个 trajectory 至少涉及：

| 字段 | 语义 | 典型形状 |
|---|---|---|
| `prompts` | 未 padding 的 prompt token | jagged `[B, P_i]` |
| `responses` | student rollout token | jagged `[B, R_i]` |
| `input_ids` | `prompt + response` | jagged `[B, P_i+R_i]` |
| `response_mask` | 1=学生生成且参与 loss，0=tool/pad | jagged `[B, R_i]` |
| `loss_mask` | 当前直接设为 `response_mask` | 同上 |
| `teacher_logprobs` | sampled-token 或 teacher top-k logprob | jagged `[B, P_i+R_i, 1/K]` |
| `teacher_ids` | 与 teacher logprob 对应的 vocab ids | jagged `[B, P_i+R_i, 1/K]` |
| `old_log_probs` | actor update 前的 student action logprob | jagged `[B, R_i]` |
| `advantages` | task reward 路径算出的 advantage | jagged `[B, R_i]` |
| `rollout_is_weights` | 可选的异步/rollout correction 权重 | jagged `[B, R_i]` |

`AgentLoopOutput.as_dict()` 把 teacher 张量从 `extra_fields` 提升为正式字段，见 `verl/experimental/agent_loop/agent_loop.py:116-147`。3D jagged teacher tensor 的 chunk/index 回归测试在 `tests/test_protocol_v2_on_cpu.py:888-920`。

### 3.6 old logprob、advantage 与 actor update

- `_compute_old_log_prob()` 用当前 actor 再做一次 forward，并把 full-sequence logprob 左移、截成 response logprob：`verl/trainer/ppo/v1/trainer_base.py:1222-1281`。
- `_compute_advantage()` 即使是 pure OPD 也会构造 task reward 与 GRPO/PPO advantage：`:1331-1390`。
- `_update_actor()` 加入 `distillation_use_topk`、global batch size、PPO epochs、temperature，再 RPC 到 actor worker：`:1415-1445`。
- actor worker 进入 `TrainingWorker.train_mini_batch()`，按 mini-batch/epoch 切数据并调用 engine：`verl/workers/engine_workers.py:233-321`。
- `TrainingWorker.train_batch()` 最终执行 `engine.train_batch(data, loss_function=self.loss_fn)`：`:323-377`。

### 3.7 engine 内的两阶段 loss

#### 非 top-k estimator

engine 只需输出学生对下一 token 的 `log_probs`。最终调用：

```python
distillation_ppo_loss(model_output=..., data=...)
```

它从 `teacher_logprobs` 取学生采样 action 在 teacher 下的 logprob，构造 k1/k2/k3 等逐 token loss。

#### `forward_kl_topk`

仅有 sampled-token logprob 不够；必须读取学生完整 vocab logits。因此 engine 在 full logits 还活着时调用：

```python
distillation_ppo_loss(student_logits=full_logits, data=micro_batch)
```

这个分支返回：

- `distillation_losses`
- `student_mass`
- `teacher_mass`
- `overlap_count`
- `overlap_token_advantage`

随后 full logits 可以释放；最终 loss 阶段只聚合这些逐 token 张量。总入口证据：`verl/trainer/distillation/losses.py:165-222`；FSDP 接线：`verl/workers/engine/fsdp/transformer_impl.py:1167-1177,1264-1274`；Megatron 接线：`verl/workers/engine/megatron/transformer_impl.py:942-978`。

### 3.8 step 结束后的权重同步

默认同步 trainer：

- 初始化 checkpoint 后先把 actor 权重推给 rollout：`verl/trainer/ppo/v1/trainer_sync.py:31-34`
- 每个 step 结束再更新 rollout 权重：`:35-38`
- rollout 完成后让 replicas sleep，释放权重/KV cache：`:40-42`

teacher 是独立、冻结的 inference replica，不参与这个权重同步。

---

## 4. 最容易错的地方：token 索引、causal shift 与 mask

### 4.1 一个具体例子

假设：

```text
prompt   = [x0, x1]          P = 2
response = [y0, y1, y2]      R = 3
sequence = [x0, x1, y0, y1, y2], S = 5
```

teacher backend 原生 prompt logprobs 的语义是：

```text
位置 0: None                         # 没有左上下文
位置 1: p(x1 | x0)
位置 2: p(y0 | x0,x1)
位置 3: p(y1 | x0,x1,y0)
位置 4: p(y2 | x0,x1,y0,y1)
```

vLLM extractor 跳过第一个 `None`，然后在末尾补 dummy：

```text
teacher row 0 = p(x1 | x0)
teacher row 1 = p(y0 | x0,x1)
teacher row 2 = p(y1 | ...)
teacher row 3 = p(y2 | ...)
teacher row 4 = dummy
```

证据：`verl/workers/rollout/vllm_rollout/utils.py:452-480`。SGLang 也显式复刻同一 contract：`verl/workers/rollout/sglang_rollout/async_sglang_server.py:68-109`。

student model 的 logits row 也预测“下一 token”：

```text
student row 0 -> x1
student row 1 -> y0
student row 2 -> y1
student row 3 -> y2
student row 4 -> sequence 后一个 token（不训练）
```

训练端最终用 `no_padding_2_padding()` 取：

```python
values[seq_offset - response_len - 1 : seq_offset - 1]
```

即本例 row `[1:4]`，恰好对应 `y0,y1,y2`；见 `verl/workers/utils/padding.py:129-143`。

所以：

- prompt 上的 teacher rows 被丢弃；
- teacher 尾部 dummy 被丢弃；
- 第一个 response token 使用最后一个 prompt token 位置的 logits；
- 最后一个 response token也有正确的 teacher/student 对齐。

### 4.2 为什么 full sequence 上的 `torch.roll` 没把 batch 边界污染进 loss

FSDP no-padding 路径把 flatten 后的 `input_ids` 左滚一位作为 label：`verl/workers/engine/fsdp/transformer_impl.py:972-1023`。flatten 序列末尾确实会滚到下一序列开头，但 response 截取永远到 `seq_offset - 1` 为止，排除了每条序列最后一个 logits row；`no_padding_2_padding()` 的切片正是这个保护。

### 4.3 multi-turn/tool mask

通用 agent loop 对 mask 的定义写得很清楚：

- `response_attention_mask`：真实 response token 为 1，pad 为 0
- `response_mask`：LLM 生成 token 为 1，tool response/pad 为 0

见 `verl/experimental/agent_loop/agent_loop.py:727-743,760-774`。

teacher 会评分完整 response，包括 tool token，因为它们属于后续状态的上下文；但最终 `response_mask=0` 的位置不会计入 OPD loss。这既保留状态，又不让学生模仿外部工具输出。

### 4.4 padded 与 jagged 的转换

legacy 路径先用 `left_right_2_no_padding()`：

- 利用 `attention_mask` 去掉左/右 padding；
- 把 `input_ids`、`position_ids`、teacher top-k 变为 jagged nested tensor；
- 保留 response mask。

见 `verl/workers/utils/padding.py:23-96`。

V1 TransferQueue 原生存 jagged 数据，后续同样依赖 `response_from_nested()` 或 `no_padding_2_padding()` 完成一位 causal shift：`verl/workers/utils/padding.py:196-230`。

---

## 5. 公式与源码逐项对应

令：

- $q_\theta(\cdot\mid s_t)$：student policy
- $p(\cdot\mid s_t)$：teacher policy
- $a_t\sim q_\theta(\cdot\mid s_t)$：student rollout 的 token
- $d_t=\log q_\theta(a_t\mid s_t)-\log p(a_t\mid s_t)$

### 5.1 teacher-top-k forward KL：分布级 GKD

代码计算：

$$
\ell_t^{K}
=
\sum_{v\in\operatorname{TopK}(p)}
p(v\mid s_t)
\left[
\log p(v\mid s_t)-\log q_\theta(v\mid s_t)
\right].
$$

对应源码：

- FSDP：`verl/trainer/distillation/fsdp/losses.py:66-72,121-130`
- Megatron：`verl/trainer/distillation/megatron/losses.py:151-163`

#### 它不是归一化后的 top-k KL

teacher top-k 概率没有重新归一化；令：

$$m_p=\sum_{v\in K}p(v\mid s_t)<1.$$

所以这个截断和可能出现负值。final wrapper 明确 `clamp_min(0.0)`：`verl/trainer/distillation/losses.py:340-356`。这不是数学上完整的 $\mathrm{KL}(p\Vert q)$，而是 teacher-top-k 支持上的稀疏近似。

源码同时记录：

$$
\text{teacher\_mass}=\sum_{v\in K}p(v),\qquad
\text{student\_mass}=\sum_{v\in K}q(v).
$$

FSDP：`verl/trainer/distillation/fsdp/losses.py:125-126`。

#### 对 student logits 的梯度

忽略 clamp 时：

$$
\frac{\partial\ell_t^K}{\partial z_j}
=
m_p q_j-p_j\mathbf 1[j\in K].
$$

Megatron 自定义 backward 把这个公式直接写进注释与实现：`verl/trainer/distillation/megatron/losses.py:227-261`。其中 clamp 后只对 active teacher tokens 计算 $m_A$。

这解释了为什么 top-k GKD 能同时：

- 降低 teacher 偏好 token 之外的 student 概率；
- 提高 teacher top-k token 的概率；
- 使用远比 full-vocab teacher distribution 少的通信量。

### 5.2 单样本 reverse-KL estimators

registry 最终调用 `kl_penalty()`；源码为 `verl/trainer/ppo/core_algos.py:2126-2183`。

用 $d=\log q(a)-\log p(a)$ 表示：

| `loss_mode` | 逐 token 代码值 | 备注 |
|---|---|---|
| `kl` / `k1` | $d$ | 可为负；$\mathbb E_{a\sim q}[d]=\mathrm{KL}(q\Vert p)$ |
| `abs` | $|d|$ | 非标准但稳定的惩罚 |
| `mse` / `k2` | $\frac12d^2$ | 非负 |
| `low_var_kl` / `k3` | $e^{-d}+d-1$ | Schulman k3；代码将内部 ratio clamp 后再把结果 clamp 到 `[-10,10]` |

teacher/student response logprob 的提取与 shape 断言：`verl/trainer/distillation/losses.py:362-394`。

### 5.3 为什么 k1 适合 PG，却不能直接反传

若直接把 $d=\log q(a)-\log p(a)$ 当 supervised loss：

$$
\nabla_\theta d=\nabla_\theta\log q_\theta(a),
$$

teacher 项是常数，梯度完全不含 teacher 信号。代码因此禁止 `k1 + use_policy_gradient=false`：`verl/workers/config/distillation.py:120-124`。

PG-OPD 则把：

$$
A_t=-\operatorname{sg}(d_t)
$$

作为 PPO advantage，并用：

$$
\rho_t(\theta)
=
\exp\left(\log q_\theta(a_t\mid s_t)-\log q_{\text{old}}(a_t\mid s_t)\right)
$$

构造 clipped policy loss。源码：

- `advantages=-distillation_losses.detach()`：`verl/trainer/distillation/losses.py:257-277`
- vanilla PPO ratio 与 clip：`verl/trainer/ppo/core_algos.py:1278-1369`

在严格 on-policy、$q_{old}=q$ 的极限，期望梯度是 reverse KL 的 policy-gradient 形式。`detach()` 是必要的；否则 reward 本身也反传，会破坏该推导。

### 5.4 为什么直接 k3 可以成为 GKD estimator

$$
k_3=e^{-d}+d-1=\frac{p(a)}{q(a)}+\log\frac{q(a)}{p(a)}-1.
$$

由于 $a\sim q$：

$$
\mathbb E_q[k_3]=\mathrm{KL}(q\Vert p).
$$

但把 rollout 样本视为固定、直接对 $k_3$ 反传时：

$$
\nabla_\theta k_3
=
\left(1-\frac{p(a)}{q(a)}\right)
\nabla_\theta\log q(a).
$$

再对 $a\sim q$ 取期望：

$$
\mathbb E_q[\nabla k_3]
=
-\mathbb E_{a\sim p}[\nabla\log q(a)],
$$

正是 forward KL $\mathrm{KL}(p\Vert q)$ 对 student 的梯度（$\mathbb E_q[\nabla\log q]=0$）。这解释了 dataclass 注释为什么把 `k3` 推荐给 direct/supervised distillation：`verl/workers/config/distillation.py:48-52`。

要注意：代码允许 PG 搭配 k3/abs/k2，但注释只推荐 PG 使用 k1。初学者应先遵循 canonical 组合，不要把“registry 允许”误写成“算法上等价”。

### 5.5 最终 loss 的组合

代码先计算：

```python
distill_loss = distillation_loss(...)
policy_loss = ppo_loss(...)
```

然后：

$$
\mathcal L=
\begin{cases}
\mathcal L_{policy}+\lambda\mathcal L_{distill}, & \text{use_task_rewards=true},\\
\mathcal L_{distill}, & \text{use_task_rewards=false}.
\end{cases}
$$

源码：`verl/trainer/distillation/losses.py:207-222`。

细节：

- `use_task_rewards=false` 时 `distillation_loss_coef` 被忽略，强制等效为 1：`:216-219`。
- policy loss 仍被计算后再置零：`:209-213`。
- 因而 task entropy bonus、reference KL 等也会随整个 policy term 一起丢弃，但它们可能已经造成额外计算。

### 5.6 token 聚合

直接 GKD 用 `agg_loss()` 聚合；PG-OPD 的 PPO surrogate 最终也用同一聚合器。支持：

- `token-mean`
- `seq-mean-token-sum`
- `seq-mean-token-sum-norm`
- `seq-mean-token-mean`

源码：`verl/trainer/ppo/core_algos.py:1138-1199`。

`response_mask` 是最终 loss mask：`verl/trainer/distillation/losses.py:247-289`。`loss_min/max` 在 `loss_max_clamp` 之前计算，因此监控里看到的是 clamp 前范围：`:250-255`。

---

## 6. FSDP、Megatron、VeOmni 的工程实现

### 6.1 FSDP / FSDP2

#### eager top-k 路径

`verl/trainer/distillation/fsdp/losses.py:75-149`：

1. teacher top-k 是 jagged nested tensor，先取 `.values()` 变成 `[1,total_tokens,K]`。
2. 若 Ulysses sequence parallel > 1，沿 sequence 维切给各 SP rank：`:99-103`。
3. 默认 materialize `F.log_softmax(student_logits)`，再按 teacher ids `gather`：`:121-124`。
4. 算 sparse forward KL 与 mass/overlap 指标。
5. engine 再把 SP 输出 gather/unpad 回完整 jagged sequence：`verl/workers/engine/fsdp/transformer_impl.py:1174-1203`。

#### chunked top-k 路径

`_chunked_topk_log_probs()` 使用恒等式：

$$
\log\operatorname{softmax}(z)_{i}
=z_i-\log\sum_j e^{z_j},
$$

逐 `(B*T)` chunk 做 fp32 `logsumexp + gather`，避免另一个 `[B,T,V]` log-softmax buffer；源码 `verl/trainer/distillation/fsdp/losses.py:26-63`。

配置注释给出的权衡：

- 默认关，短上下文更快；
- `N=14K,V=152K` 约有 6 倍时间开销；
- 长上下文（注释举例 `>=64K`）可避免默认路径 OOM；
- chunk size 调 launch 开销与单 chunk 内存，不改变数学结果。

见 `verl/workers/config/distillation.py:72-82`。

它只省掉 log-softmax 中间 buffer，不代表普通 FSDP 完全不 materialize student logits。若要连 `[B,L,V]` lm-head logits 都不落地，当前示例指向 VeOmni fused path。

### 6.2 Megatron：vocab parallel + context parallel

`verl/trainer/distillation/megatron/losses.py` 的核心设计：

1. `vocab_parallel_log_softmax()` 在 TP group 上：
   - all-reduce 全局 max；
   - 本地 exp/sum；
   - all-reduce 全局 sum；
   - 得到每个 vocab shard 的正确 log-softmax。
   证据：`:27-55`。
2. teacher ids 是 global vocab id；每个 TP rank 只保留落入自身 vocab range 的 teacher entries：`:73-120`。
3. 每 rank 在本地 vocab shard gather student logprob，再对逐 token KL 做 TP all-reduce：`:122-163`。
4. 自定义 backward 直接实现 $m_Aq-p$，避免构造 full-vocab teacher target：`:218-261`。
5. 为计算 student/teacher top-k overlap，每 rank 先取本地 top-k，再 TP all-gather candidates，二次 top-k 得全局 student top-k：`:176-208`。
6. THD 与 BSHD teacher tensor 都先做 CP preprocess/split：`:264-304`。

这条路径的通信可以概括为：

```text
TP: all_reduce(max) + all_reduce(sum)      # vocab-parallel softmax
TP: all_reduce(per-token KL / mass)        # 合并各 vocab shard
TP: all_gather(local student top-k)         # 只为 overlap 诊断
CP: teacher sequence tensor按 context rank切分
DP: 各自处理不同样本，梯度/指标再按训练框架聚合
```

### 6.3 VeOmni fused top-k

VeOmni 基于 FSDP2，但在 `use_fused_kernels=true` 时：

- 把已经 causal-shift 的 labels 传给 fused causal LM loss；
- 把 `teacher_topk_ids/logprobs` 传给 `chunk_topk_distill`；
- SP 情况下按同样规则切 teacher tensor；
- chunked lm-head projection 直接产出 per-token distillation outputs。

源码：`verl/workers/engine/veomni/transformer_impl.py:847-900`；示例解释：`examples/on_policy_distillation_trainer/run_qwen3_0.6b_opd_veomni.sh:4-15,70-75`。

### 6.4 data parallel 的 loss 与指标

engine 在 forward/backward 前 all-reduce global valid-token count，并写入 `batch_num_tokens`、`dp_size`：

- FSDP：`verl/workers/engine/fsdp/transformer_impl.py:638-652`
- Megatron：`verl/workers/engine/megatron/transformer_impl.py:633-650`

worker 输出：

- loss 在 DP group 做 AVG：`verl/workers/engine_workers.py:182-190`
- 其他 metrics 用 `allgather_dict_into_dict` 汇总：`:198-207`

这也是 `agg_loss()` 为什么乘 `dp_size`：用来抵消分布式梯度平均，维持 global-token/global-sequence 归一化。

---

## 7. teacher 分布式服务、路由与多教师

### 7.1 资源池拆分

`MultiTeacherModelManager`：

1. 按每个 teacher 的 `world_size` 把总 teacher pool 顺序切成子池：`verl/experimental/teacher_loop/teacher_model.py:180-194`。
2. 每个 teacher 子池再按 `per_replica_world_size` 切成 replica pools：`:62-99`。
3. 每个 replica 用配置的 vLLM/SGLang rollout class 启动，模型路径来自 teacher 自己的 `HFModelConfig`：`:75-89`。

### 7.2 节点边界约束

`split_resource_pool` 是线性切 bundle，不懂节点边界。代码因此显式检查每个 replica 实际跨节点数是否等于由 `ceil(W/P)` 推出的期望值；见 `verl/experimental/teacher_loop/teacher_model.py:103-144`。

这意味着“GPU 总数刚好相等”仍可能启动失败。例如每节点 8 GPU，前一个 teacher 用掉 6 个 bundle，后一个 5-GPU replica 从 bundle 6 开始，就会横跨两个节点；但 5 GPU 本应能装进单节点，校验会报错。解决方式是调整 teacher 顺序、TP/DP/PP 或 replica 数，使每个连续区间对齐。

### 7.3 per-teacher load balancing

每个 teacher 的 replicas 共享一个 `GlobalRequestLoadBalancer`：

- sticky mapping 优先；
- 新 request 选 in-flight 最少的 server；
- request 完成后 fire-and-forget release counter。

源码：`verl/workers/rollout/llm_server.py:65-118,165-249`。

原生 OPD 每个样本生成新的 uuid，因此 teacher query 的主要效果是 least-inflight 分流，而不是跨 turn prefix stickiness。

### 7.4 多教师路由

路由逻辑：`verl/experimental/teacher_loop/teacher_manager.py:86-100`。

- 单 teacher：忽略样本路由值，所有样本都走唯一 teacher。
- 多 teacher：`sample[distillation.teacher_key]` 必须存在且完全匹配某个 teacher `key`。
- 默认 `teacher_key=data_source`。
- client dict keys 必须与已解析 teacher keys 完全相同：`:73-84`。

官方 MOPD 示例用：

```text
openai/gsm8k       -> Qwen3-32B text teacher
hiyouga/geometry3k -> Qwen3-VL-32B vision-language teacher
```

配置见 `examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh:109-139`。

数据应 shuffle，否则 concat 后的不同数据集可能让某一个 teacher 长时间独占请求；verl 文档也提示这一点：`docs/algo/opd.md:553-557`。

---

## 8. 启动示例与如何选择

### 8.1 官方脚本矩阵

| 脚本 | student train engine | teacher/rollout engine | 默认算法 |
|---|---|---|---|
| `run_qwen3_8b_fsdp.sh` | FSDP | vLLM | `k1 + PG` |
| `run_qwen3_8b_megatron.sh` | Megatron | vLLM | `forward_kl_topk + direct` |
| `run_qwen3_0.6b_opd_veomni.sh` | VeOmni/FSDP2 fused | vLLM | `k1 + PG`，可切 top-k fused |
| `run_qwen3_vl_8b_fsdp.sh` | FSDP | vLLM | VL 单 teacher、`k1 + PG` |
| `run_qwen3_8b_mopd_fsdp.sh` | FSDP | vLLM | text+VL 多 teacher、`k1 + PG` |
| `run_qwen3_8b_mopd_veomni.sh` | VeOmni fused | vLLM | fused MOPD |
| `run_qwen3_5_4b_fsdp.sh` | FSDP2，GPU/NPU | vLLM | Qwen3.5 VL、`k1 + PG` |

索引：`examples/on_policy_distillation_trainer/README.md:5-24`。

### 8.2 canonical PG-OPD

`run_qwen3_8b_fsdp.sh` 默认：

- student `Qwen/Qwen3-8B`
- teacher `Qwen/Qwen3-32B`
- `loss_mode=k1`
- `use_policy_gradient=True`
- `use_task_rewards=False`
- student rollout vLLM TP=2
- teacher TP=2，teacher pool=4 GPU，因此自动起 2 个 replicas

证据：`examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh:6-29,79-87,102-116`。

启动：

```bash
cd /path/to/verl
STUDENT_MODEL=Qwen/Qwen3-8B \
TEACHER_MODEL=Qwen/Qwen3-32B \
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
```

脚本里的 GSM8K/MATH parquet 路径是 `$HOME/data/...`，需要提前准备或改脚本：`:39-45`。

### 8.3 canonical top-k GKD

Megatron 示例默认正好是推荐组合：

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk
USE_POLICY_GRADIENT=False
DISTILLATION_TOPK=64
bash examples/on_policy_distillation_trainer/run_qwen3_8b_megatron.sh
```

默认值与关键配置：`examples/on_policy_distillation_trainer/run_qwen3_8b_megatron.sh:16-18,106-121`。

FSDP 也能跑同一组合，只需覆盖环境变量；若长上下文因 log-softmax OOM，再考虑 chunked top-k 或 VeOmni fused path。

### 8.4 GPU 数不要只看 `trainer.n_gpus_per_node`

teacher pool 与 global pool 是独立的，总 Ray cluster 资源至少是两者之和。

例如 `run_qwen3_8b_fsdp.sh` 默认：

```text
global_pool  = 8 GPU
teacher_pool = 4 GPU
总需求       = 12 GPU
```

而 VeOmni 0.6B 示例特意配置成 `7 + 1 = 8 GPU`：`examples/on_policy_distillation_trainer/run_qwen3_0.6b_opd_veomni.sh:23-26,97-113`。

脚本变量 `TEACHER_WORLD_SIZE` 实际被写到 `distillation.n_gpus_per_node`；多节点时总 teacher GPU 还要再乘 `distillation.nnodes`，不要被变量名误导。

### 8.5 常用最小覆盖

```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="$STUDENT" \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$VAL_PARQUET" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.rollout.max_model_len=3073 \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  distillation.enabled=true \
  distillation.n_gpus_per_node=4 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2 \
  distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=true \
  distillation.distillation_loss.use_task_rewards=false
```

这是教学伪配置，不是对任意硬件都可直接运行的资源建议；batch、offload、TP/SP/PP 仍要按模型与集群调整。

---

## 9. 原生同步、V1 异步与旧 `recipe/gkd` 不要混为一谈

### 9.1 默认 sync

`trainer.v1.trainer_mode=sync`：每一步更新后把新 actor 权重同步给 rollout，再生成下一批。teacher query 虽逐样本异步，但状态来自当前 student version，最接近严格 on-policy。

### 9.2 V1 async / fully async

V1 配置还提供：

- `colocate_async`
- `separate_async`

默认 warmup/parameter sync 配置见 `verl/trainer/config/ppo_trainer.yaml:203-225`。这些模式通过 replay buffer 和 model-version threshold 允许旧 policy trajectory，吞吐更高，但“on-policy”应更准确地称作 bounded-staleness OPD。

独立 fully-async 实验路径：

- teacher pool 创建：`verl/experimental/fully_async_policy/fully_async_rollouter.py:623-657`
- teacher clients 交给 FullyAsyncAgentLoopManager：`:698-713`
- actor worker 仍接收同一个 distillation loss：`verl/experimental/fully_async_policy/fully_async_trainer.py:334-346`
- E2E 脚本用 `staleness_threshold=0.5`、每 4 step 触发参数同步、partial rollout：`tests/special_e2e/run_fully_async_policy_opd.sh:47-59,245-252`

该 E2E 使用 `k1 + PG` 的多 teacher 配置，并将 8 GPU 分成 `2 rollout + 4 training + 2 teachers`：`tests/special_e2e/run_fully_async_policy_opd.sh:9-22,177-212`。

### 9.3 `docs/advance/async-on-policy-distill.md` 是另一条旧 recipe 叙事

这份 2025-11 文档描述：

- 外部 ZMQ teacher service
- one-step-off / two-step-off scheduler
- `recipe/gkd/main_gkd`
- Megatron + vLLM

见 `docs/advance/async-on-policy-distill.md:24-64,97-159,173-188,225-232`。

但当前本地 `recipe` 是一个**未 checkout 的 git submodule**，`.gitmodules` 指向 `https://github.com/verl-project/verl-recipe.git`；本地没有 `recipe/gkd` 源码可供核验。因此网站应把它标为“外部/旧 recipe 架构”，不能把其中的 ZMQ、batch teacher service、one/two-step scheduler 当成当前原生 `distillation.*` 实现事实。

---

## 10. 指标怎么读

指标由 worker 最终加上 `actor/` 前缀，因此常见名称是 `actor/distillation/*`。

### 10.1 所有模式

- `actor/distillation/loss`：未乘 `distillation_loss_coef` 的聚合蒸馏 loss；写入处 `verl/trainer/distillation/losses.py:214-220`。
- `actor/distillation/abs_loss`：单样本 estimator 的绝对值均值，k1 可正可负时特别有用；`:386-393`。
- `actor/distillation/loss_min/max`：mask 后逐 token loss 范围，且发生在 `loss_max_clamp` 之前；`:109-120,250-255`。
- `actor/distillation/pg_clipfrac`、`ppo_kl`、`pg_clipfrac_lower`：PG-OPD 内部 PPO surrogate 的 clip/漂移诊断；`:257-279` 与 `core_algos.py:1364-1369`。

### 10.2 top-k 模式

- `student_mass`：student 在 teacher top-k token 集合上的概率总质量。
- `teacher_mass`：teacher 自己 top-k 覆盖的概率质量；低说明 K 过小或 teacher 在该学生状态下分布较平。
- `overlap_ratio`：teacher top-k 与 student top-k 集合交集比例。
- `overlap_token_advantage`：重叠 teacher token 的负 KL contribution 平均；是 logging-only。

聚合逻辑：`verl/trainer/distillation/losses.py:294-356`；FSDP token 级 overlap：`verl/trainer/distillation/fsdp/losses.py:132-148`；Megatron：`verl/trainer/distillation/megatron/losses.py:176-216`。

### 10.3 一个实用调试基线

把 student 和 teacher 设成同一 checkpoint，loss 应接近 0，但不会精确为 0，因为 train engine 与 inference engine 的数值实现、dtype、kernel、temperature/processor 可能不同。verl 文档也推荐这个方法：`docs/algo/opd.md:611-613`。

---

## 11. 已知限制、工程坑与排错清单

### A. 启动前/配置错误

#### 1. `enabled=true` 但 teacher pool 为 0

症状：`n_gpus_per_node must be greater than 0` 或 `nnodes must be greater than 0`。

证据：`verl/trainer/ppo/v1/trainer_base.py:615-624`。YAML 默认 `nnodes=0`，所以必须覆盖。

#### 2. teacher GPU 总数不精确匹配

症状：`Sum of teacher ... must match ... resource pool size`。

检查公式：`sum(num_replicas*TP*DP*PP) == n_gpus_per_node*nnodes`。证据：`verl/workers/config/distillation.py:274-287`。

#### 3. `teacher_model` 被静默 pop

多 teacher 时不要沿用默认 entry 名作为真实 teacher；改名为 `gsm8k`、`geo3k` 等。证据：`verl/workers/config/distillation.py:239-257,308-310`。

#### 4. replica 跨节点区间不对齐

症状：`replica ... span N nodes but ... expects M`。

调整 teacher 顺序、TP/DP/PP 或 replica 数。证据：`verl/experimental/teacher_loop/teacher_model.py:103-144`。

#### 5. teacher `max_model_len` 少了 `+1`

必须容纳 `prompt + full response + one generated token`。证据：`verl/workers/config/distillation.py:172-185`。

#### 6. 改了 student rollout temperature，却忘了 teacher

`_get_teacher_sampling_params()` 要求 teacher inference temperature 恰为 1.0，否则直接 `NotImplementedError("vLLM does not support temperature for prompt_logprobs")`；见 `verl/experimental/teacher_loop/teacher_manager.py:30-43`。

YAML 默认 teacher temperature 又继承 student rollout temperature：`verl/trainer/config/distillation/distillation.yaml:108-110`。所以如果 student rollout 用 0.7，应显式把每个 teacher 的 `inference.temperature=1.0`。

#### 7. teacher/student tokenizer 或 vocab 不一致

原生管线把 student token ids 原样送给 teacher，并把 teacher global vocab ids 直接用于 student `gather`；见 `agent_loop.py:1016-1020`、`fsdp/losses.py:115-124`。源码没有 tokenizer/vocab 等价性校验。

后果：

- vocab size 不同：可能直接 out-of-range；
- id 空间大小相同但语义不同：训练悄悄学错；
- multimodal special tokens/processor 不同：状态语义错位。

实践上应选同一模型家族并显式核对 tokenizer files、special token ids、vocab size。

#### 8. vLLM `max_logprobs < topk`

代码会自动补默认值，但若用户显式设得更小则启动失败；见 `verl/workers/config/distillation.py:194-208`。

### B. 算法组合错误

#### 9. `k1 + direct`

启动即 `ValueError`，teacher 梯度消失；见 `verl/workers/config/distillation.py:120-124`。

#### 10. `forward_kl_topk + PG`

只 warning，不阻止，但 PG 只沿 sampled action 的 `∇logπ(a)` 更新，浪费 top-k 其他 token 的分布信号；见 `verl/workers/config/distillation.py:112-118`。正确入门组合是 `forward_kl_topk + use_policy_gradient=false`。

#### 11. pure OPD 仍要求 reward/advantage 字段

`use_task_rewards=false` 不是“跳过 PPO pipeline”。trainer 仍算 reward、old logprob、advantage，loss 里仍先算 `ppo_loss` 再置零。删除 reward dataset 字段或 advantage 步骤会破坏当前实现。

#### 12. 不必要的 reference KL

若 `actor.use_kl_loss` 或 `algorithm.use_kl_in_reward` 开着，trainer 会起/ref forward，学生同时被 base reference 与 teacher 拉扯；pure OPD 下这些计算甚至可能最终被整个 policy term 置零。canonical OPD 通常两者都关。

### C. 内存、kernel 与 backend

#### 13. FSDP eager top-k OOM

默认会 materialize `[B,T,V]` log-softmax。可选：

- 开 `use_chunked_topk`，省 log-softmax buffer但更慢；
- 降低 micro-batch/token budget；
- 用 VeOmni fused chunk-top-k，连完整 lm-head logits materialization 也规避。

#### 14. top-k GKD 与 fused kernel 的支持边界

源码明确接通 teacher top-k fused inputs 的是 VeOmni：`verl/workers/engine/veomni/transformer_impl.py:867-890`。

标准 FSDP fused 分支只在输出里已经存在 `fused_linear_aux.distillation_losses` 时提取，注释明确称其为 “veomni's chunk_topk_distill path”：`verl/workers/engine/fsdp/transformer_impl.py:1120-1133`。Megatron fused 分支 `:917-929` 没有调用 distillation logits processor，而非 fused 分支才在 `:973-976` 调用。

因此在没有额外 kernel 证据时：

- VeOmni：可以按官方脚本开 fused top-k；
- 普通 FSDP/Megatron 的 `forward_kl_topk`：先关 `use_fused_kernels`，避免最终阶段缺 `distillation_losses`。

这是基于当前源码接线得出的工程边界，不是对未来版本的永久限制。

#### 15. top-k 不是 full KL

只拿 teacher top-k，未归一化，负截断 loss 最后被 clamp 到 0。`teacher_mass` 太低时应加 K；不能把它在教材里称为“精确 forward KL”。

### D. 数据与异步

#### 16. MOPD routing key 缺失或不匹配

多 teacher 时会直接报 `Routing key is required` 或 `No teacher configured`；见 `verl/experimental/teacher_loop/teacher_manager.py:86-100`。

#### 17. 多数据集不 shuffle

会造成 teacher 请求长时间偏置到单一 domain；代码能跑，但优化顺序差。MOPD 示例主动 `data.shuffle=True`：`examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh:56-68`。

#### 18. async 不是严格 on-policy

V1 async/fully async 的 trajectory 可能来自旧 student version。监控 off-policy threshold、parameter sync 周期与 rollout correction；教材应称“有限陈旧度的 OPD”。

#### 19. validation 没有 teacher loss

`AgentLoopWorker._compute_teacher_logprobs()` 在 `validate=True` 时跳过。validation 指标是 task reward/accuracy，不是蒸馏 KL。

#### 20. teacher query 不是 batch teacher forward

当前原生 manager 是逐样本 `client.generate`，靠 asyncio 与 replicas 并发；不要拿旧 `recipe/gkd` 的 ZMQ micro-batching 描述替代它。吞吐不够时优先看 teacher replicas 与 in-flight load，而不是寻找原生 `teacher_batch_size` 配置。

---

## 12. 源码审计中发现的两个“待回归验证”风险

以下不是 verl 文档承诺，也没有在当前机器上跑 GPU E2E 确认；应在网站高级章节中标为“读源码发现的潜在风险”，不能写成已确认 bug。

### 12.1 distillation global-batch 归一化刷新顺序

调用顺序是：

```python
distill_loss = distillation_loss(...)
policy_loss  = ppo_loss(...)
```

见 `verl/trainer/distillation/losses.py:207-210`。

但 `ActorConfig.global_batch_info` 只在 `ppo_loss()` 内由当前 `data[dp_size,batch_num_tokens,global_batch_size]` 刷新：`verl/workers/utils/losses.py:57-68`。direct GKD 在此之前就用 `config.global_batch_info` 聚合（`distillation/losses.py:280-289`）；PG 分支也在此前把 actor 的 dict 复制到 distillation loss config（`:257-277`）。

静态阅读因此提示：第一个 micro-batch 可能使用空默认值，后续 micro-batch 可能使用上一次调用留下的 global stats；当 `token-mean`、多 DP、每 mini-batch token 数变化时，可能影响归一化。

建议回归：构造两个 global token count 不同的连续 mini-batch，比较 OPD scalar/gradient 与手算 global-token mean；若确认，应在进入 `distillation_loss()` 前直接从当前 `data` 刷新 global info。

### 12.2 FSDP 与 Megatron 的 `teacher_mass` clamp 口径

- FSDP 在 logprob clamp **之前**计算 `teacher_mass/student_mass`：`verl/trainer/distillation/fsdp/losses.py:125-129`。
- Megatron 在 teacher logprob clamp **之后**构造 `target_topk_probs` 和 `target_topk_mass`：`verl/trainer/distillation/megatron/losses.py:111-120`。

当 `log_prob_min_clamp` 截断了 teacher tail 时，两 backend 的 `teacher_mass` 监控口径可能不同，即使主 loss/gradient测试在常规输入上相符。建议增加专门覆盖 teacher logprob 低于 clamp 的跨 backend metric test。

---

## 13. 测试覆盖与本地验证状态

### 13.1 静态审计到的测试

| 测试 | 覆盖点 | 未覆盖点 |
|---|---|---|
| `tests/workers/test_chunked_topk_log_probs_on_cpu.py:49-192` | dense vs chunked 数值、低精度 GPU、chunk-size invariance、梯度、空输入 | 完整 trainer/teacher 链路 |
| `tests/workers/test_distillation_topk_symmetry_on_cpu.py:14-238` | FSDP remove-padding true/false 的输出传播；overlap 数学与聚合 | 用 stub processor，不测完整 logprob 数值/teacher server |
| `tests/utils/test_special_megatron_kl_loss_tp.py:141-234` | TP Megatron vs full-vocab FSDP 的 loss、gradient、overlap；THD/BSHD | 需 distributed GPU；不比较 clamp 后 teacher_mass 口径 |
| `tests/utils/test_megatron_bshd_preprocess.py:65-124` | 3D top-k jagged tensor 保留 K 维与 padding | 不测最终 loss |
| `tests/test_protocol_v2_on_cpu.py:888-920` | 3D jagged teacher tensor 的 chunk/index layout | teacher query 本身 |
| `tests/special_e2e/run_fully_async_policy_opd.sh:1-279` | 8-GPU fully async、Megatron、MOPD、k1 PG | shell E2E；不是细粒度断言；不覆盖 sync top-k GKD |

`test_distillation_topk_symmetry_on_cpu.py:14-30` 还记录了一个真实回归背景：过去 `use_remove_padding=false` 会静默丢失 distillation outputs，最终 KeyError；当前测试同时守护两种 padding 模式。

### 13.2 明显的测试空白

在 `tests/` 中没有找到以下符号的直接 unit test：

- `AsyncTeacherLLMServerManager`
- `_resolve_teacher_key`
- `compute_teacher_logprobs_single`
- `MultiTeacherModelManager` 的 pool/node 校验
- `distillation_ppo_loss` 的完整 task+distill 组合
- reverse-KL estimator 的 OPD 专用数值/梯度测试
- vLLM/SGLang extractor 的首 token skip + 尾 dummy 对齐测试
- sync V1 的 single-teacher top-k GKD E2E

### 13.3 本地执行状态

尝试运行：

```text
python -m pytest \
  tests/workers/test_chunked_topk_log_probs_on_cpu.py \
  tests/workers/test_distillation_topk_symmetry_on_cpu.py \
  tests/test_protocol_v2_on_cpu.py::...
```

当前 Python 环境在 collection 阶段报 `ModuleNotFoundError: No module named 'torch'`，因此本文的 test 结论是**阅读测试源码所得的覆盖审计**，不是这台机器上的通过记录。没有为报告擅自安装 PyTorch/GPU 依赖。

### 13.4 推荐新增的最小回归测试

1. 用长度 5 的 toy sequence 明确断言 teacher extractor rows 与 response slice 完全对齐。
2. 单 teacher 忽略 routing、多 teacher 缺/错 key 报错。
3. pool size、默认 `teacher_model` pop、跨节点 bundle 三类 config test。
4. k1 direct 必须失败；k1 PG 的 detach 梯度必须含 teacher advantage；k3 direct 的期望梯度 toy test。
5. `use_task_rewards=false` 时 task policy gradient 为零、distill gradient 非零。
6. 当前/不同 global token count 下 distill aggregation 的 DP invariance。
7. FSDP/Megatron 在 teacher 与 student logprob 都跨 clamp 时 loss、gradient、mass 指标一致。
8. SGLang 与 vLLM extractor 对同一人工 backend payload 输出一致。

---

## 14. 可直接用于学习网站的教学伪代码

### 14.1 算法层伪代码

```python
student = trainable_model()
teacher = frozen_inference_server()

for prompts in dataloader:
    # 1) On-policy：状态来自当前 student
    responses, rollout_logp = student.generate(prompts)
    states = concat(prompts, responses)

    # 2) Teacher 不续写答案，只在 student states 上给 next-token 分布
    if loss_mode == "forward_kl_topk":
        teacher_ids, teacher_logp = teacher.prompt_topk_logprobs(
            states, k=K
        )
    else:
        teacher_ids, teacher_logp = teacher.prompt_sampled_token_logprobs(
            states
        )

    # 3) causal shift：只截取“预测 response token”的位置
    response_mask = generated_by_student_and_not_tool(states)

    old_logp = student.logprobs(states).response_slice()

    if loss_mode == "forward_kl_topk":
        student_logits = student.forward(states)
        per_token_distill = sum_over_teacher_topk(
            exp(teacher_logp) *
            (teacher_logp - log_softmax(student_logits)[teacher_ids])
        )
        distill_loss = masked_aggregate(per_token_distill, response_mask)

    elif use_policy_gradient:
        d = student_sampled_logp - teacher_sampled_logp
        distill_advantage = -stop_gradient(d)
        distill_loss = ppo_clipped_surrogate(
            new_logp=student_sampled_logp,
            old_logp=old_logp,
            advantage=distill_advantage,
            mask=response_mask,
        )

    else:  # e.g. direct k3 GKD
        d = student_sampled_logp - teacher_sampled_logp
        per_token_distill = exp(-d) + d - 1
        distill_loss = masked_aggregate(per_token_distill, response_mask)

    task_loss = ppo_or_grpo_loss(...)  # 当前 verl 即便 pure OPD 也会先算
    total_loss = (
        task_loss + coef * distill_loss
        if use_task_rewards
        else distill_loss
    )

    total_loss.backward()
    optimizer.step()
    sync_student_weights_to_rollout_servers()
```

### 14.2 系统层伪代码

```python
# Controller
global_pool  = RayPool(actor_rollout_critic_ref_gpus)
teacher_pool = RayPool(dedicated_teacher_gpus)

teacher_managers = split_by_teacher_world_size(teacher_pool)
for teacher in teachers:
    replicas = split_into_replicas(teacher.pool, TP * DP * PP)
    teacher.client = LeastInflightRayClient(replicas)

for prompt_batch in dataloader:
    for sample in prompt_batch:                    # asyncio tasks
        trajectory = await student_rollout(sample)
        reward = await score(trajectory)
        route = sample[teacher_key]
        teacher_signal = await teachers[route].client.prompt_logprobs(
            trajectory.prompt + trajectory.response
        )
        transfer_queue.put(trajectory, reward, teacher_signal)

    batch = replay_buffer.sample(train_batch_size)
    batch = balance_by_sequence_length(batch)
    batch.old_logp = actor_forward(batch)
    batch.advantages = compute_task_advantage(batch)
    actor_update(batch, distillation_loss)
    checkpoint_engine.update_rollout_weights()
```

### 14.3 初学者版 token 对齐伪代码

```python
# backend 给出：
# [None, score(token_1), score(token_2), ..., score(token_{S-1})]

teacher_rows = backend_prompt_logprobs[1:]
teacher_rows.append(dummy_row)   # 现在长度仍为 S

# student logits row i 预测 input_ids[i+1]
# response 从 input_ids[P] 开始，因此用 student/teacher row P-1 开始
response_rows = rows[P - 1 : S - 1]
loss = (response_rows * response_mask).sum()
```

---

## 15. 可直接改写成站点正文的章节草稿

### 章节标题：从一次 rollout 走进 verl 的 OPD

#### 开场

先忘掉“蒸馏就是拿 teacher 答案做 SFT”。在 verl 的 OPD 里，答案首先由 student 自己写。student 写到哪里，teacher 就走到哪里；teacher 不要求 student回到自己的标准轨迹，而是在 student 已经到达的状态上回答：“如果下一步由我选择，我会把概率放在哪些 token 上？”

这就是 on-policy 的工程含义：训练数据里的 state 不是 teacher 预先生成的静态语料，而是当前 student rollout 动态产生的。

#### 第一幕：student 先走

`AgentLoopWorkerTQ` 为 batch 中每个 prompt 建一个异步 task。rollout server 用当前同步过来的 student 权重生成 response。多轮任务里，模型 token、工具调用 token、工具返回 token 一起构成后续 state；但只有模型自己生成的 token 被 `response_mask` 标成 1。

#### 第二幕：teacher 只评分，不重写

rollout 一结束，worker 就把 `prompt + response` 的 token ids 原样发给 teacher。请求参数里 `max_tokens=1`，真正需要的是 `prompt_logprobs`。若做 PG-OPD，teacher 只返回学生已选 token 的 logprob；若做 top-k GKD，teacher 返回每个状态下概率最高的 K 个 token 及其 logprob。

这个设计有两个好处：teacher 不用自回归生成整条答案，监督又覆盖 response 的每一步；不同样本 teacher query 可以和其他样本仍在进行的 rollout 重叠。

#### 第三幕：为什么需要一个 dummy row

causal LM 无法给第一个输入 token 计算条件概率。vLLM/SGLang 的第 0 项因此是 None。verl 跳过它，把第 1 项放到 row 0——它正好对应 student logits row 0 对“下一个 token”的预测。为了保持张量长度等于输入长度，末尾再补一行 dummy；response slice 会排除最后一行，所以 dummy 不进入 loss。

这里可以配一张五 token 索引图，使用本文第 4.1 节的例子。

#### 第四幕：两条 loss 路线

第一条是 GKD。teacher 给 top-k 分布，student 直接最小化 teacher-top-k 上的 forward KL。它一次能告诉 student 多个候选 token 应该升还是降，但需要在 full student logits 还在显存时计算。

第二条是 PG-OPD。teacher 只比较 student 已经选中的 token。`log q - log p` 越大，说明 student 对这个 token 比 teacher 更自信，负的 distillation reward 就越强。这个 reward 必须 stop-gradient，再通过 PPO ratio 作用于 `log q`。

若资源有限，还有 direct k3：teacher 仍只返回 sampled-token logprob，但 k3 直接反传的期望梯度对应 forward-KL GKD。它在通信上便宜，却没有显式 top-k 那样低方差、丰富的分布监督。

#### 第五幕：为什么 pure OPD 里还能看到 GRPO

这是 verl 当前实现的一个工程事实，而不是算法必需条件。trainer 总是沿用 PPO 数据流，先算 reward、old logprob 和 advantage；`distillation_ppo_loss` 也先调用普通 `ppo_loss`。当 `use_task_rewards=false` 时，普通 policy loss 最后才被置零。因此配置里出现 `algorithm.adv_estimator=grpo`，不代表 pure OPD 的 teacher signal 被 GRPO 计算；它只是共享了训练管线。

#### 第六幕：并行系统如何把它跑起来

actor/rollout 与 teacher 用不同 Ray pools。teacher 可以有多个 replica，每个 replica 内又用 TP/DP/PP。请求由 least-inflight load balancer 分发。student 侧，FSDP 用 sequence parallel 切序列；Megatron 还要在 tensor-parallel vocab shards 上 all-reduce softmax normalization 与 KL。teacher top-k ids 一直使用 global vocab 坐标，每个 TP rank只处理自己拥有的词表区间。

#### 收束

当你在日志里看到 OPD loss 时，问自己四个问题：这些 state 是哪一版 student 生成的？teacher top-k 覆盖了多少概率质量？response mask 是否只选中了模型 token？当前 loss 是直接 GKD，还是把 teacher 差异当作 PPO advantage？只要这四件事说清楚，verl 的 OPD 主链就已经读懂了。

---

## 16. 建议的网站拆章

1. **五分钟看懂 verl OPD**：用 1 张数据流图 + 1 个 token 对齐例子。
2. **配置实验室**：单 teacher/MOPD、pool 计算器、合法 loss 组合交互表。
3. **Teacher prompt-logprob 解剖**：vLLM/SGLang contract、dummy row、temperature=1。
4. **Mask 与 causal shift**：可视化 prompt/response/tool/pad 四类 token。
5. **PG-OPD 源码推导**：k1、stop-gradient、old/new ratio、clip。
6. **GKD 源码推导**：teacher-top-k sparse forward KL、mass、负截断项。
7. **k3 为什么能 direct**：用 2-token toy distribution 做数值互动。
8. **FSDP/Megatron/VeOmni**：并行通信与显存权衡。
9. **MOPD**：业务 key 路由、replicas 与 bundle 对齐。
10. **启动与排错**：按报错字符串反查配置。
11. **测试阅读课**：从 padding regression、TP 对齐、chunked gradient 学工程验证。
12. **高级审计**：global-batch normalization 顺序、backend metric 口径、async staleness。

---

## 17. 最终核对清单（网站写作时不要写错）

- [ ] 当前快照默认是 V1 sync，不是 legacy `RayPPOTrainer`。
- [ ] teacher query 逐样本异步，但默认 step 仍同步更新。
- [ ] teacher 对 student 的 `prompt + response` 做 prompt logprob，不是重新生成 teacher response。
- [ ] top-k 是 teacher top-k，且未重新归一化。
- [ ] `forward_kl_topk` 推荐 direct；`k1` 推荐 PG。
- [ ] k1 direct 在配置阶段被禁止。
- [ ] pure OPD 仍运行 reward/old-logprob/advantage/PPO plumbing。
- [ ] `response_mask` 排除 tool/pad；prompt 与 trailing dummy 由 causal response slice 排除。
- [ ] teacher/student 必须共享 token-id 语义，源码没有自动校验。
- [ ] teacher temperature 必须是 1.0；YAML 默认会继承 student temperature。
- [ ] teacher pool GPU 与 global pool GPU 要相加。
- [ ] MOPD 不要把真实首 teacher 命名为默认 `teacher_model`。
- [ ] 普通 FSDP/Megatron top-k 先用非 fused；VeOmni 才有明确 fused teacher-top-k 接线。
- [ ] async/fully async 应说明 policy staleness，不把它称作严格 on-policy。
- [ ] 本报告的测试结论是静态覆盖审计；当前机器缺 PyTorch，未宣称 runtime pass。

