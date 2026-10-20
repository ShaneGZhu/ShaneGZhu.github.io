---
title: '大模型强化学习（RLHF/RLVR）中的 KL 散度与 K3 估计器完全指南：从数学推导到工程实现'
description: >-
  KL 散度在大模型强化学习里的双重身份：算法层的安全绳，系统工程层检验训推一致性的探针。
  从 Schulman 的三种估计器推导到 k3 的方差优势、PPO 与 GRPO 的梯度差异，以及如何用它
  观察推理与训练引擎的一致性，最后是 PyTorch 实现对照。
pubDate: 2026-10-20
tags: ['RL', 'Inference']
kind: 'Tutorial'
lang: 'zh'
density: 'dense'
readingMinutes: 60
---

> **摘要**：在以 PPO、GRPO（DeepSeek-R1）为代表的大模型强化学习训练中，**KL 散度（Kullback-Leibler Divergence）** 扮演着双重核心角色：在**算法层**，它是防止策略崩溃与 Reward Hacking 的“安全绳”；在**系统工程层**，它的 **K3 蒙特卡洛估计器（$k_3$）** 是检验推理引擎（vLLM/SGLang）与训练引擎（FSDP/Megatron）**“训推一致性”**的黄金探针。本文从第一性原理出发推导 Schulman 的三种 KL 近似估计器（$k_1, k_2, k_3$），比较 PPO 与 GRPO 在梯度反传上的差异，并说明同一个 k3 为什么还能当作尺子，去量推理引擎与训练引擎之间的数值一致性。

---

## 1. 引言：为什么大模型 RL 离不开 KL 散度？

### 1.1 什么是 KL 散度？
在信息论中，**KL 散度（又称相对熵，Relative Entropy）** 用于衡量用概率分布 $Q$ 来近似真实概率分布 $P$ 时所损失的信息量。对于离散分布，其数学定义为：

$$
D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right]
$$

KL 散度有两个在强化学习中至关重要的性质：
1. **非负性（Gibbs 不等式）**：对于任意分布 $P$ 和 $Q$，恒有 $D_{\text{KL}}(P \parallel Q) \ge 0$，当且仅当 $P = Q$（几乎处处相等）时取等号。
2. **非对称性**：$D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$。因此它不是严格意义上的“距离（Metric）”，但在优化中被广泛用作分布差异的度量。

### 1.2 大模型 RL 中的“橡皮筋”与 Reward Hacking
在大模型强化学习（无论是传统的 RLHF 还是 DeepSeek-R1 带火的 RLVR 可验证奖励强化学习）中，通常同时存在两个关键模型：
* **Actor 模型（策略模型 $\pi_\theta$）**：正在接受梯度更新、不断探索高分回答的模型。
* **Reference 模型（参考模型 $\pi_{\text{ref}}$）**：冻结权重的初始模型（通常是经过 SFT 微调后的 Base 模型）。

如果我们只让 $\pi_\theta$ 去无限制地最大化奖励 $R(x)$，模型很快就会发现系统的漏洞——即 **Reward Hacking（奖励作弊）**：
* 在 RLHF 中，模型可能发现不断重复某几个特定形容词、或者输出奇怪的符号组合就能让 Reward Model 给出极高分，从而彻底丧失语言流畅度。
* 在数学/代码等 RLVR 任务中，如果没有约束，模型为了刷高某个特定格式的奖励，可能会退化出乱码式的思维链（CoT），或者遗忘原有的多语言与通用对话能力。

因此，现代 RL 算法的目标函数几乎都遵循 **KL 约束优化目标**：

$$
\max_{\theta} \; \mathbb{E}_{x \sim \pi_\theta} [R(x)] - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})
$$

这里的 $\beta$ 是 KL 惩罚系数。KL 散度就像一根**橡皮筋**，把正在狂奔探索的 $\pi_\theta$ 拴在 $\pi_{\text{ref}}$ 附近，迫使模型在“拿高分”和“说人话（保持先验分布）”之间寻找帕累托最优。

### 1.3 补充细节：为什么 RL 用的是逆向 KL（Reverse KL）？
细心的读者会发现，在监督微调（SFT）中，我们最小化的是交叉熵损失，等价于最小化真实数据分布 $P_{\text{data}}$ 与模型分布 $\pi_\theta$ 的**前向 KL（Forward KL）**：$D_{\text{KL}}(P_{\text{data}} \parallel \pi_\theta)$。
而在 RL 中，采样分布 $\pi_\theta$ 在左边，$\pi_{\text{ref}}$ 在右边，即 $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$，这被称为**逆向 KL（Reverse KL）**：
* **前向 KL（Mass-Covering）**：要求凡是 $P_{\text{data}}(x) > 0$ 的地方，$\pi_\theta(x)$ 必须有概率，倾向于覆盖所有模式。
* **逆向 KL（Mode-Seeking）**：因为期望是对 $x \sim \pi_\theta$ 求的，只要 $\pi_{\text{ref}}(x) \to 0$，为了不让 $\log \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}$ 爆炸，就逼迫 $\pi_\theta(x)$ 也必须趋近于 $0$。也就是说：**Reference 模型认为不可能说出的胡话，Actor 模型绝对不能说**；至于 Reference 认为合理的多种回答，Actor 只需要集中精力挑出奖励最高的那一种（Mode-Seeking）即可！这完美契合了 RL 激发模型最优潜能的目标。

---

## 2. 工程瓶颈：为什么我们算不起“精确的 KL”？

既然 $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ 这么好，为什么不直接按照定义把它精确算出来呢？

让我们算一笔工程账。大语言模型是自回归生成序列 $x = (x_1, x_2, \dots, x_T)$ 的。在第 $t$ 个 Token 位置上，给定上文 $s_t = x_{<t}$，模型输出的是在整个词表 $V$ 上的条件概率分布。如果要计算第 $t$ 步的**精确全词表 KL（Exact Full-Vocab KL）**：

$$
D_{\text{KL}}\left(\pi_\theta(\cdot \mid s_t) \parallel \pi_{\text{ref}}(\cdot \mid s_t)\right) = \sum_{v \in V} \pi_\theta(v \mid s_t) \left( \log \pi_\theta(v \mid s_t) - \log \pi_{\text{ref}}(v \mid s_t) \right)
$$

在分布式 RL 框架（如 `verl`、`OpenRLHF`）中，Actor 模型和 Reference 模型往往分布在不同的 GPU 节点或分片上。如果我们要算全词表 KL，意味着需要将整个词表的 Logits 在网络间传输或缓存在显存中：
* 假设词表大小 $|V| = 152,064$（如 Qwen2.5 系列），序列长度 $T = 8,192$，Batch Size $B = 64$，采用 `FP32`（4 Bytes）存储。
* 单次前向需要缓存/传输的张量大小为：

$$
64 \times 8192 \times 152064 \times 4 \text{ Bytes} \approx 318.9 \text{ GB}
$$

这还仅仅是一个 Micro-Batch 的开销！在工程上，跨节点传输数百 GB 的 Logits 会瞬间打满 IB/RoCE 网卡带宽，显存也会直接 OOM（Out of Memory）。

**工业界的标准解法**：
在采样（Rollout）和前向计算时，我们**丢弃整个词表的 Logits，只提取并保存实际被采样出来的那一个 Token $x_t$ 对应的标量对数概率 `logprob`**！
这样一来，张量维度从 $[B, T, |V|]$ 骤降了十五万倍，变成了 $[B, T]$，仅需几 MB 显存。

但这也带来了核心的数学挑战：**当我们手里只有单条采样轨迹上的 Token 对数概率 $\log \pi_\theta(x_t)$ 和 $\log \pi_{\text{ref}}(x_t)$ 时，该如何无偏、稳定地估计出真实的 KL 散度？**

---

## 3. 核心理论：Schulman 的三种蒙特卡洛估计器（k1, k2, k3）

2020 年，OpenAI 联合创始人、PPO 算法作者 **John Schulman** 在其个人博客上发表了著名的短文 *《Approximating KL Divergence》*，系统性地给出了只利用采样 Token 的 `logprob` 来近似估计 KL 散度的三种蒙特卡洛估计器（Monte Carlo Estimators）：$k_1, k_2, k_3$。

为了书写优雅，我们严格遵循 Schulman 原文的符号约定：
假设样本 $x$ 是从当前策略分布 $P = \pi_\theta$ 中采样得到的（即 $x \sim \pi_\theta$），目标参考分布为 $Q = \pi_{\text{ref}}$。我们定义**概率比值（Ratio）$r$** 为参考分布与当前策略分布之比（**请务必注意分子是 $\pi_{\text{ref}}$，分母是 $\pi_\theta$，与 PPO Clip 里的 ratio 刚好互为倒数！**）：

$$
r = \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} = \exp\left( \log \pi_{\text{ref}}(x) - \log \pi_\theta(x) \right)
$$

对其取自然对数，则有：

$$
\log r = \log \pi_{\text{ref}}(x) - \log \pi_\theta(x)
$$

我们要估计的目标是真实 KL 散度：

$$
D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \mathbb{E}_{x \sim \pi_\theta} [-\log r]
$$

基于单个采样点 $x$ 的比值 $r$，Schulman 给出了如下三种估计器：

| 估计器名称 | 紧凑公式 (基于 $r$) | 展开公式 (基于 $\log \pi$) | 是否无偏 (Unbiased) | 是否恒非负 ($\ge 0$) | 采样方差 (Variance) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$k_1$** (Naive) | $-\log r$ | $\log \pi_\theta(x) - \log \pi_{\text{ref}}(x)$ | ✅ 是 | ❌ **否** | 🔴 极高 |
| **$k_2$** (Squared) | $\frac{1}{2} (\log r)^2$ | $\frac{1}{2} \left(\log \pi_\theta(x) - \log \pi_{\text{ref}}(x)\right)^2$ | ❌ 否 | ✅ 是 | 🟢 低 |
| **$k_3$** (**K3 KL**) | $(r - 1) - \log r$ | $\frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - 1 - \log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)}$ | ✅ **是** | ✅ **是** | 🟢 **低** |

---

## 4. 数学揭秘：为什么 K3（k3）是唯一的最优解？

为什么在现代 RL 框架中，$k_3$ 几乎一统天下？我们通过严谨的数学对比来拆解。

### 4.1 k1 的致命缺陷：正负抵消与“骗奖励”漏洞
$k_1 = \log \pi_\theta(x) - \log \pi_{\text{ref}}(x)$ 是直接把定义式里的期望符号 $\mathbb{E}$ 去掉得到的最朴素估计器。它的期望显然等于真实 KL：$\mathbb{E}_{x \sim \pi_\theta}[k_1] = D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$。

但在单次采样中，**$k_1$ 完全可能是负数！**
想象一下：在某个 Token 位置上，Actor 模型由于随机探索，采样出了一个概率较低的词（比如 $\pi_\theta(x) = 0.01$），而 Reference 模型原本在这里的概率较高（比如 $\pi_{\text{ref}}(x) = 0.1$）。此时：

$$
k_1 = \log(0.01) - \log(0.1) = -2.3026 < 0
$$

这在训练中会引发灾难性后果：
1. **引发负惩罚（即正奖励激励）**：在 PPO 中，总奖励是 $R_{\text{total}} = R - \beta \cdot \text{KL}$。当 $k_1$ 为负数时，$-\beta \cdot k_1 > 0$！模型会震惊地发现：“只要我故意挑那些我概率比 Reference 低的词输出，我不仅不会受罚，还能白嫖一笔正奖励！”这会导致训练迅速崩溃。
2. **一阶方差极大**：当两模型非常接近（即 $r \approx 1$）时，真实 KL 散度其实非常小（二阶小量），但单步的 $k_1 = -\log r \approx -(r-1)$ 依然在 $0$ 上下剧烈正负震荡，信噪比极低。

> **注：为什么不能简单地给 $k_1$ 加个绝对值变成 $|k_1|$ 或 $\max(0, k_1)$？**
> 因为一旦加了绝对值或 `ReLU` 截断，虽然保证了非负，但原本用来抵消的正负项无法抵消，会导致数学期望被严重高估（产生极大的正向偏差 Bias），导致模型被死死锁在原地不敢更新。

### 4.2 k2 的妥协：有偏的二阶近似
为了消除负数并降低方差，$k_2 = \frac{1}{2}(\log r)^2$ 直接对对数比值做了平方。
* **优点**：天然非负（$\ge 0$），且在 $r \approx 1$ 附近，它的期望约等于真实 KL（可以通过二阶泰勒展开证明），梯度也非常平滑。
* **缺点**：它是一个**有偏估计器（Biased Estimator）**。当 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 差距稍微拉大时，$\mathbb{E}[k_2]$ 测量的其实是一种对称化的 $f$-散度，不再等于真实的 KL 散度。

### 4.3 k3 的神来之笔：控制变量法（Control Variates）
怎么才能既保持 $k_1$ 的**严格无偏性**，又拥有 $k_2$ 的**非负性与低方差**？
Schulman 使用了蒙特卡洛积分中的经典降方差技巧——**控制变量法（Control Variates）**。

基本思想是：如果我们要估计随机变量 $Y = -\log r$ 的期望，我们可以给它加上任意一个**已知期望严格为 $0$ 的随机变量 $Z$**（即 $Y_{\text{new}} = Y + \lambda Z$）。只要 $\mathbb{E}[Z] = 0$，新变量的期望就绝对不会变；而如果我们精心挑选 $Z$，让它和 $Y$ 的波动正好相反，就能把方差抵消掉！

在这里，我们天然知道一个期望为 $0$ 的量，那就是**概率比值与 $1$ 的差：$Z = r - 1$**。

#### 第一步：证明 E[r − 1] = 0（从而证明 k3 严格无偏）
我们对 $(r - 1)$ 在采样分布 $\pi_\theta$ 下求期望：

$$
\begin{aligned}
\mathbb{E}_{x \sim \pi_\theta}[r - 1] &= \sum_{x \in V} \pi_\theta(x) \left( \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - 1 \right) \\
&= \sum_{x \in V} \left( \pi_{\text{ref}}(x) - \pi_\theta(x) \right) \\
&= \underbrace{\sum_{x \in V} \pi_{\text{ref}}(x)}_{=1} - \underbrace{\sum_{x \in V} \pi_\theta(x)}_{=1} = 1 - 1 = 0
\end{aligned}
$$

（*注意：上述证明成立的前提是 $\pi_\theta$ 的支撑集包含 $\pi_{\text{ref}}$，在 LLM 的 Softmax 输出下，所有 Token 概率均严格大于 $0$，因此恒成立。*）

既然 $\mathbb{E}[r - 1] = 0$，我们把 $k_1$ 和 $(r - 1)$ 直接相加，就得到了 **$k_3$ 估计器**：

$$
k_3 = -\log r + (r - 1) = (r - 1) - \log r
$$

对其求期望：

$$
\mathbb{E}_{x \sim \pi_\theta}[k_3] = \mathbb{E}_{x \sim \pi_\theta}[-\log r] + \mathbb{E}_{x \sim \pi_\theta}[r - 1] = D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) + 0 = D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})
$$

**无偏性得证！**

#### 第二步：证明 k3 ≥ 0 恒成立（非负性）
在高等数学中，有一个最基础的对数不等式：对于任意正实数 $r > 0$，函数 $g(r) = \log r$ 的图像恒位于其在 $r=1$ 处的切线 $y = r - 1$ 的下方，即：

$$
\log r \le r - 1 \quad (\forall r > 0)
$$

当且仅当 $r = 1$（即 $\pi_\theta(x) = \pi_{\text{ref}}(x)$）时等号成立。
将该不等式移项，立即得到：

$$
k_3 = (r - 1) - \log r \ge 0
$$

**非负性得证！** 无论采样出什么稀奇古怪的 Token，单步算出来的 $k_3$ 永远是一个大于等于 $0$ 的实数。

#### 第三步：证明 k3 为什么方差极低（泰勒展开视角）
我们在 $r = 1$ 附近对 $\log r$ 进行二阶泰勒展开：

$$
\log r = (r - 1) - \frac{1}{2}(r - 1)^2 + \frac{1}{3}(r - 1)^3 - \mathcal{O}((r - 1)^4)
$$

将其代入 $k_3$ 的表达式中：

$$
\begin{aligned}
k_3 &= (r - 1) - \left[ (r - 1) - \frac{1}{2}(r - 1)^2 + \mathcal{O}((r - 1)^3) \right] \\
&= \frac{1}{2}(r - 1)^2 + \mathcal{O}((r - 1)^3)
\end{aligned}
$$

看到了吗？在 $k_1 = -\log r$ 中导致正负剧烈震荡的**一阶线性项 $(r - 1)$ 被控制变量完美抵消了**！剩下的主导项是平滑的二阶平方项 $\frac{1}{2}(r - 1)^2$。
根据 Schulman 在博客中的数值实验，当两个正态分布 $\mathcal{N}(0, 1)$ 和 $\mathcal{N}(0.1, 1)$ 的真实 KL 为 $0.005$ 时：
* $k_1$ 的采样标准差与期望之比高达 **$20.0$**（噪声是信号的 20 倍！）；
* 而 $k_3$ 的采样标准差与期望之比仅为 **$1.42$**，方差降低了整整两个数量级！

---

## 5. 算法博弈：KL 放进 Reward (PPO) 还是直接做 Loss (GRPO)？

搞懂了 $k_1, k_2, k_3$ 的公式后，在实际编写训练代码时，你会遇到第二个核心分水岭：**算出来的单步 KL，到底是塞进 Reward 里，还是直接加到 Loss 里做反向传播？**

这也是传统 **PPO（如 InstructGPT）** 与现代 **GRPO（如 DeepSeekMath / DeepSeek-R1）** 的重要设计差异之一。

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 路径 A：Reward Penalty (传统 PPO)                                       │
│   logprob_actor (detach) & logprob_ref ──► 计算 K1/K3 ──► 扣减 Reward    │
│   ──► 经 GAE 计算 Advantage ──► 乘进 Policy Loss 反传梯度                 │
├─────────────────────────────────────────────────────────────────────────┤
│ 路径 B：Direct KL Loss (GRPO / DeepSeek-R1)                             │
│   logprob_actor (带梯度!) & logprob_ref ──► 计算 K3 Loss ──► 直接加进总   │
│   Loss 对 θ 求偏导反传                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.1 路径 A：作为 Reward Penalty（传统 PPO 做法）
在经典的 RLHF PPO 流程中，KL 是作为环境奖励的修正项出现的。对于序列中的第 $t$ 个 Token，修正后的即时奖励为：

$$
\tilde{r}_t = r_t - \beta \cdot D_{\text{KL}}^{(t)}
$$

随后，$\tilde{r}_t$ 会通过 GAE（广义优势估计）累加到优势函数 $\hat{A}_t$ 中。
* **梯度特性**：在这条路径下，计算 $D_{\text{KL}}^{(t)}$ 时使用的是采样阶段固定下来的 `old_log_probs`（即完全 `detach()`，不带梯度）。KL 是通过改变标量优势值 $\hat{A}_t$ 的大小，利用策略梯度定理（Policy Gradient）间接地去惩罚 Actor。
* **视野特性**：因为经过了 GAE 的时间步累积（$\gamma \lambda$ 衰减），第 $t$ 步如果产生了巨大的 KL 偏移，不仅惩罚第 $t$ 步，还会影响前面所有步（$t' < t$）的价值估计。
* **选型建议**：如果放进 Reward，强烈建议使用 **$k_3$ (`low_var_kl`)** 而不是 $k_1$，防止单步负 KL 奖励投机。

### 5.2 路径 B：直接作为 KL Loss 反传（GRPO 的做法）
在 DeepSeek 提出的 **GRPO（Group Relative Policy Optimization）** 算法中，砍掉了庞大的 Critic（Value）模型，优势函数 $\hat{A}$ 直接由组内多个回答的最终结果奖励（Outcome Reward）做归一化得到。同时，DeepSeek 团队选择**不再把 KL 塞进 Reward，而是直接在 Loss 函数末尾加上 $k_3$ 作为正则化项**：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( \rho_{i,t}(\theta)\hat{A}_{i,t}, \; \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t} \right) - \beta \cdot k_3^{(i,t)}(\theta) \right]
$$

其中，单步的 $k_3(\theta)$ Loss 为（注意这里的 $\pi_\theta(x_t)$ 是当前训练步前向计算出来的、**带有梯度的动态变量**）：

$$
\mathcal{L}_{\text{KL}}^{k_3}(\theta) = \frac{\pi_{\text{ref}}(x_t)}{\pi_\theta(x_t)} - \log \frac{\pi_{\text{ref}}(x_t)}{\pi_\theta(x_t)} - 1
$$

#### 深度思考：对 k3(θ) 直接求导，梯度到底长什么样？
很多初学者会误以为：“既然 $\mathbb{E}[k_3] = D_{\text{KL}}$，那直接对单步的 $k_3(\theta)$ 求导，是不是就等于对真实 KL 求导？”
我们来亲手求一下导数！对 $\mathcal{L}_{\text{KL}}^{k_3}(\theta)$ 关于参数 $\theta$ 求偏导（利用 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$）：

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\text{KL}}^{k_3}(\theta) &= \nabla_\theta \left[ \exp\left(\log \pi_{\text{ref}}(x) - \log \pi_\theta(x)\right) - \left(\log \pi_{\text{ref}}(x) - \log \pi_\theta(x)\right) - 1 \right] \\
&= \exp\left(\log \pi_{\text{ref}}(x) - \log \pi_\theta(x)\right) \cdot \left( -\nabla_\theta \log \pi_\theta(x) \right) + \nabla_\theta \log \pi_\theta(x) \\
&= -\left( \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} - 1 \right) \nabla_\theta \log \pi_\theta(x) \\
&= \left( 1 - r \right) \nabla_\theta \log \pi_\theta(x)
\end{aligned}
$$

看这个极其漂亮的梯度公式：**$\nabla_\theta \mathcal{L}_{\text{KL}}^{k_3}(\theta) = (1 - r) \nabla_\theta \log \pi_\theta(x)$**！
它的物理意义非常直观：
1. **当 $\pi_\theta(x) > \pi_{\text{ref}}(x)$ 时（即 $r < 1$）**：说明 Actor 在这个 Token 上的概率比 Reference 太高了。此时系数 $(1 - r) > 0$，因为我们在做梯度下降（最小化 Loss），参数更新方向会**压低** $\log \pi_\theta(x)$。
2. **当 $\pi_\theta(x) < \pi_{\text{ref}}(x)$ 时（即 $r > 1$）**：说明 Actor 在这个 Token 上的概率比 Reference 太低了。此时系数 $(1 - r) < 0$，梯度下降会反过来**拉高** $\log \pi_\theta(x)$，而且拉高的力度与比值 $r$ 成正比！
3. **当 $\pi_\theta(x) = \pi_{\text{ref}}(x)$ 时（即 $r = 1$）**：梯度严格等于 $0$，达到完美平衡点。

这就是为什么 DeepSeek-R1 / GRPO 选用 $k_3$ 直接作为 Loss 项：它只惩罚当前发生偏移的那一个 Token（Token-level 精准正则），不污染全局 Advantage，而且梯度自带双向自适应拉回机制！

---

## 6. 系统工程：为什么用 K3 监控“训推一致性”？

如果你在使用 `verl`、`OpenRLHF` 或内部自研的 RL 框架看 WandB / TensorBoard 监控面板，除了看到 `actor/kl_loss`（Actor 与 Ref 的 KL）之外，通常还会看到另一个极其关键的系统指标：**`rollout_train_kl` 或 `mismatch_k3`（训推一致性 K3）**。

而且老手都会告诉你：**“看训推一致性的 K3 时，这个值必须越接近 0 越好，一旦飙高，训练必崩！”**

### 6.1 什么是“训推分离”架构下的训推不一致？
在现代大规模 RL 训练中，为了极致吞吐，架构上普遍采用 **推训分离（Disaggregated Rollout & Training）**：
1. **推理阶段（Rollout Engine）**：把当前模型权重同步给 **vLLM** 或 **SGLang** 引擎，利用 PagedAttention 和 Continuous Batching 高速生成回答序列，并顺手记录下生成每个 Token 时的对数概率：$\log \pi_{\text{rollout}}(x_t)$。
2. **训练阶段（Training Engine）**：把生成的序列送给 **FSDP** 或 **Megatron-LM** 引擎。在开始第一个 PPO Epoch 更新前（**此时训练引擎里的模型权重与刚刚推理引擎里的权重在理论上 100% 相同！**），训练引擎会用完整序列做一次 Prefill 前向传播，重算出一份基准对数概率 `old_log_probs`：$\log \pi_{\text{train}}(x_t)$。

**诡异的事情发生了**：明明加载的是同一份 Checkpoint 权重，输入的是完全相同的 Token ID，但训练引擎算出来的 $\log \pi_{\text{train}}(x_t)$ 和推理引擎算出来的 $\log \pi_{\text{rollout}}(x_t)$ 居然不一样！

为了量化这两个引擎之间的**系统数值误差（System Mismatch）**，框架开发者直接拿来了现成的 $k_3$ 公式这把尺子，把两边的 `logprob` 喂进去：

$$
\log r_{\text{sys}} = \log \pi_{\text{train}}(x_t) - \log \pi_{\text{rollout}}(x_t)
$$

$$
\text{K3}_{\text{consistency}} = \exp\left(\log r_{\text{sys}}\right) - 1 - \log r_{\text{sys}}
$$

### 6.2 为什么必须用 K3 测一致性，且越低越好？
1. **数学真值必须为 0**：
   因为此时对比的是**同一个模型权重在两个引擎下的输出**，没有任何算法层面的策略更新。在理想世界里，$\pi_{\text{train}}(x_t)$ 应当严格等于 $\pi_{\text{rollout}}(x_t)$，即 $r_{\text{sys}} = 1$，此时 $\text{K3}_{\text{consistency}} = (1 - 1) - \log(1) = 0$。任何大于 $0$ 的数值，纯粹代表底层算子、精度或配置不对齐带来的**系统噪声**，自然是越低越好！
2. **绝不发生“正负抵消”（完胜 $k_1$）**：
   如果用 $k_1 = \log \pi_{\text{rollout}}(x_t) - \log \pi_{\text{train}}(x_t)$ 来算整条序列的平均误差，假设第 1 个 Token 算偏了 $+0.5$，第 2 个 Token 算偏了 $-0.5$，求平均后 $k_1 = 0.0$！你会误以为系统完美对齐，实则内部已经千疮百孔。而 **$k_3$ 对每个 Token 都是严格非负的**，只要有任意一个 Token 对不上，误差就会实打实地累加起来。
3. **天然的“长尾异常放大器”（完胜 MAE / $k_2$）**：
   在高温采样（如 $T=1.0$）或 Top-p 边缘采样时，推理引擎经常会采样出概率只有 $10^{-4}$ 甚至 $10^{-5}$ 的小概率 Token。如果因为某个算子 Bug，训练引擎算出来的概率变成了 $10^{-2}$（$\log r_{\text{sys}} \approx 4.6$），由于 $k_3$ 含有指数项 $\exp(\log r_{\text{sys}})$，$k_3$ 的数值会瞬间飙升到接近 $100$！在监控曲线上会立刻拉出一根刺眼的尖峰（Spike），帮你瞬间定位 Bug。

### 6.3 如果训推 K3 偏高，会对 RL 训练造成什么灾难？
在 PPO/GRPO 中，重要性采样比率（Importance Sampling Ratio）定义为 $\rho_t(\theta) = \frac{\pi_\theta(x_t)}{\pi_{\text{old}}(x_t)}$。
* 如果你用推理引擎的 $\pi_{\text{rollout}}$ 作为分母 $\pi_{\text{old}}$，而分子 $\pi_\theta$ 是由训练引擎算出来的，当训推不一致时，在**第 0 步（参数还没改）**，$\rho_t$ 就已经偏离了 $1.0$（比如变成了 $1.5$ 或 $0.6$）。这会导致大量 Token 在第一步还没学东西时就直接触发了 PPO 的 Clip 边界（$1 \pm 0.2$），梯度直接被截断清零，或者产生错误的暴冲梯度！
* 如果你用训练引擎重算的 $\pi_{\text{train}}$ 作为分母 $\pi_{\text{old}}$，虽然起步时 $\rho_t = 1.0$，但因为这批数据实际上是由 $\pi_{\text{rollout}}$ 采样出来的，这违反了 On-policy 假设（变成了未做修正的 Off-policy），同样会在低概率 Token 上引发剧烈的训练震荡。

### 6.4 一个例子：这个数为什么突然变大

数值量级本身不是重点，值得看的是它**怎么变**。举个具体的情形：某次实验里 `rollout_train_kl` 一直稳定在 $10^{-4}$ 量级，换成另一个模型后突然跳到 $2\times10^{-3}$，而且每一批都稳定在这个水位上——这种"平移"而不是"抖动"，通常意味着两侧在某处没有对齐，而不是某个 token 偶然算错了。

常见的原因大致有这么几类：

* **精度与算子配置不一致**：RoPE 一侧在 `BF16` 里算、另一侧在 `FP32` 里算；`rms_norm_eps` 一边是 `1e-5`、一边是 `1e-6`；Tensor Parallel 的切分维度和 `All-Reduce` 顺序两边不同。
* **采样参数没有还原**：推理时开了 `temperature != 1.0` 或 top-p，引擎返回的 `logprob` 已经过温度缩放与截断，而训练侧算的是原始 logits 上的 `logprob`。这一项最容易漏，也最容易造成量级偏差。
* **权重同步漏了东西**：Actor 往推理引擎广播权重时，某些层名没匹配上，或者 QKV 的合并/拆分顺序搞反。
* **位置或 mask 错位**：多轮拼接、left/right padding 转换时 `position_ids` 没有扣掉 padding。
* **MoE 的路由翻转**：这类模型对微小误差格外敏感，`BF16` 下 Router 的 top-k 可能选出不同的专家，之后整条轨迹的激活就完全不同了。

排查时一般就从这条线往下走：先确认采样参数是否被还原，再比较两三层的中间激活，最后才怀疑算子实现。

---

## 7. torch 代码示例

这一节只做一件事：把上面的公式落到张量上，看清掩码和数值边界该怎么处理。代码是示意性的，不是某个框架的最终实现——两者的差别放在 7.3 里对照。

### 7.1 为什么必须处理 `exp` 上溢
在 $k_3 = \exp(\log r) - 1 - \log r$ 中，包含了一个指数运算 `torch.exp(log_r)`，其中 `log_r = logprob_ref - logprob_actor`。
当序列里混进某个生僻 Token 时，完全可能出现 Reference 认为它概率正常（如 $\log \pi_{\text{ref}} = -2.0$）、而 Actor 极度排斥它（如 $\log \pi_\theta = -100.0$）的情形。
此时，`log_r = 98.0`！在 `FP16/BF16` 甚至 `FP32` 下计算 `torch.exp(98.0)`（约为 $3.6 \times 10^{42}$）会直接导致 **数值上溢（`Inf` / `NaN`）**，一旦反传梯度，整个模型的权重瞬间全部变成 `NaN`！

所以实现里有两件事是躲不掉的：**对 `log_r` 做截断（clamping）**，以及**用 `response_mask` 把 Prompt 和 Padding 位置剔除**。

### 7.2 一个最小示例

```python
import torch
import torch.nn.functional as F
from typing import Literal

def approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: Literal["k1", "k2", "k3", "abs"] = "k3",
    clamp: float = 20.0,
) -> torch.Tensor:
    """
    逐 Token 的近似 KL：D_KL(pi_theta || pi_ref)。

    同一个函数也能用来观察训推一致性，只要把两侧的 logprob 换成
    log_probs=rollout_log_probs、ref_log_probs=train_old_log_probs。

    Args:
        log_probs: 当前策略的 log pi_theta(x)，形状 [batch, seq]
        ref_log_probs: 参考策略的 log pi_ref(x)，形状 [batch, seq]
        estimator: 'k1' | 'k2' | 'k3' | 'abs'
        clamp: log_r 的截断范围；exp(20)≈4.9e8，仍在 FP32 范围内

    Returns:
        形状 [batch, seq] 的逐 Token KL
    """
    # 在 FP32 里算：BF16 下的累加误差会直接体现在这个量级很小的指标上
    log_probs_fp32 = log_probs.float()
    ref_log_probs_fp32 = ref_log_probs.float()

    # 2. 计算 log_r = log(\pi_{ref} / \pi_\theta) = log \pi_{ref} - log \pi_\theta
    log_r = ref_log_probs_fp32 - log_probs_fp32

    if estimator == "k1":
        # k1 = -log_r = log \pi_\theta - log \pi_{ref}
        kl = -log_r

    elif estimator == "abs":
        # abs = |log_r| (有偏，仅部分早期框架使用)
        kl = torch.abs(log_r)

    elif estimator == "k2":
        # k2 = 0.5 * (log_r)^2
        kl = 0.5 * (log_r ** 2)

    elif estimator == "k3":
        # 关键防爆技巧：对 log_r 做截断，防止 exp(log_r) 变成 Inf
        clamped_log_r = torch.clamp(log_r, min=-clamp, max=clamp)
        r = torch.exp(clamped_log_r)
        # k3 = (r - 1) - log_r
        kl = (r - 1.0) - clamped_log_r

    else:
        raise ValueError(f"unknown estimator: {estimator}")

    return kl.to(log_probs.dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """只在 response_mask == 1 的位置求平均，跳过 Prompt 与 Padding。"""
    valid_values = values * mask
    if dim is None:
        return valid_values.sum() / (mask.sum() + 1e-8)
    else:
        return valid_values.sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T = 4, 128

    # 模拟数据：前 32 个 Token 是 Prompt/Padding (mask=0)，后 96 个是生成的 Response (mask=1)
    response_mask = torch.zeros((B, T), dtype=torch.float32)
    response_mask[:, 32:] = 1.0

    # 场景 1：GRPO 算法中计算带梯度的 K3 KL Loss
    actor_log_probs = torch.randn((B, T), requires_grad=True)
    ref_log_probs = (actor_log_probs.detach() + 0.05 * torch.randn((B, T))) # 模拟轻微偏移

    token_k3_loss = approx_kl(actor_log_probs, ref_log_probs, estimator="k3")
    kl_loss = masked_mean(token_k3_loss, response_mask)
    kl_loss.backward()
    print(f"[算法层] GRPO K3 KL Loss: {kl_loss.item():.6f} (梯度已成功反传)")

    # 场景 2：系统层监控“训推一致性 K3”
    with torch.no_grad():
        rollout_log_probs = torch.randn((B, T))
        # 模拟正常的 BF16 微小算子误差 (约 1e-3 级别的 logprob 波动)
        train_old_log_probs = rollout_log_probs + 0.002 * torch.randn((B, T))

        mismatch_k3 = approx_kl(
            log_probs=rollout_log_probs,
            ref_log_probs=train_old_log_probs,
            estimator="k3",
        )
        mean_mismatch_k3 = masked_mean(mismatch_k3, response_mask)
        print(f"[系统层] 训推一致性 K3 Mismatch: {mean_mismatch_k3.item():.8f} (越接近 0 越健康)")
```

### 7.3 真实框架里长什么样

上面那段是为了讲清公式写的。真实框架会在这个基础上做取舍，以 `verl` 为例（`verl/trainer/ppo/core_algos.py`，Apache-2.0），k3 分支只有五行：

```python
if kl_penalty in ("low_var_kl", "k3"):
    kl = ref_logprob - logprob
    # For numerical stability
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
```

和示例相比有三处值得注意的差别：

1. **截断范围是写死的**，`log_r` 卡在 $\pm 20$、输出再卡在 $\pm 10$，不开放给调用方。前者保证 `exp` 不溢出，后者是为了防止个别 Token 的极端值把整个 batch 的均值带偏。
2. **没有强转 FP32**，直接在训练精度上算，省下一次显存往返。代价是精度余量更小，这也是为什么截断范围要给得比较宽。
3. **多了个 `k3+` 变体**：前向用 k3 的值，反向借用 k2 的梯度（`backward_score - backward_score.detach() + forward_score.detach()`）。原因是 k3 的**期望**无偏，但**梯度**不是，而 k2 的梯度是对的。给想做直接 KL Loss 的场景用。

另外注意命名：框架里这个估计器叫 `low_var_kl`，`k3` 只是它的别名，日志里出现 `actor/kl_loss` 时未必对应哪一种。

---

## 8. 总结与思考

回顾全文，我们可以把大模型强化学习中的 KL 知识浓缩为以下三条核心定律：

1. **为什么用近似估计**：全词表 KL（Full-Vocab KL）虽然精确，但在十几万词表与长序列下会导致显存和通信爆炸；只保存采样 Token 的 `logprob` 进行蒙特卡洛估计是分布式 RL 系统的必然选择。
2. **为什么 $k_3$ 称王**：在 Schulman 的三种估计器中，$k_1$ 虽无偏但存在负数漏洞与高方差，$k_2$ 虽非负低方差但有偏；唯有 **$k_3 = (r - 1) - \log r$** 借助控制变量 $(r - 1)$，同时做到了 **严格无偏、恒大于等于 $0$、二阶低方差**，并且对 $\theta$ 求导后拥有极其优雅的 $(1 - r)\nabla_\theta \log \pi_\theta$ 自适应梯度。
3. **一把尺子，两处测量**：
   * 测 **$\pi_\theta$ 与 $\pi_{\text{ref}}$**：监控的是算法训练的探索幅度，适度大于 $0$ 是正常的，防止模型训崩。
   * 测 **$\pi_{\text{rollout}}$ 与 $\pi_{\text{train}}$**：监控的是底层推理与训练引擎的**训推一致性**，此时权重相同，利用 $k_3$ 的非负累加性和指数放大效应，数值**越逼近 $0$ 越好**。

### 参考文献与延伸阅读
1. John Schulman, *[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)*, 2020.
2. Shao et al., *[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)* (GRPO 算法与 K3 KL Loss 推导), 2024.
3. DeepSeek-AI, *[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)*, 2025.
4. Sheng et al., *[HybridFlow: A Flexible and Efficient RLHF Framework (verl)](https://arxiv.org/abs/2409.19256)*, EuroSys 2025.