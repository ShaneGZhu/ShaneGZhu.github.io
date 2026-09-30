---
title: '聊聊 KV Cache 的显存账：一张卡到底能塞多少并发'
description: >-
  把 KV Cache 的显存占用算清楚，再看 PagedAttention 的分块管理省下的是哪一部分。
  含一份可以直接抄去改的估算脚本，以及几个在生产里反复踩到的坑。
pubDate: 2025-02-24
tags: ['Inference', 'KV Cache']
lang: 'zh'
featured: true
---

被问得最多的一个问题是「这张卡能跑多少并发」。这个问题没法直接回答，但可以算：
先把显存分成几块，再看哪一块会先撑满。绝大多数时候，先撑满的是 KV Cache。

## 先把账算清楚

一条序列在长度 $L$ 时占用的 KV Cache 为

$$
\text{KV}(L) = 2 \cdot N_{\text{layer}} \cdot N_{\text{kv head}} \cdot d_{\text{head}} \cdot L \cdot b
$$

其中前面的 2 是 K 和 V 各一份，$b$ 是每个元素的字节数（FP16 是 2，FP8 是 1）。

注意这里是 $N_{\text{kv head}}$ 而不是 attention head 数量。GQA 下这两个数差好几倍，
按 attention head 算会把显存需求估高 4 到 8 倍，这是最常见的一处算错。

以一个 32 层、8 个 KV head、head_dim 128 的模型为例，FP16 下：

$$
\text{KV}(1000) = 2 \times 32 \times 8 \times 128 \times 1000 \times 2 \ \text{B} = 128\ \text{MiB}
$$

也就是说，**每 1000 个 token 吃掉 128 MiB**。一张 80 GiB 的卡，扣掉 16 GiB 权重和
大约 4 GiB 的激活与碎片，剩下 60 GiB 能放约 47 万个 token。平均上下文 4000 token 的
话，理论并发上限是 117。

### 一份能直接用的估算脚本

```python
from dataclasses import dataclass

GIB = 1024 ** 3

@dataclass(frozen=True)
class ModelShape:
    layers: int
    kv_heads: int          # 注意是 KV head，不是 attention head
    head_dim: int
    weight_bytes: int      # 权重实际占用，含 embedding

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        # K 和 V 各一份，所以乘 2
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


def max_concurrency(
    shape: ModelShape,
    gpu_memory_bytes: int,
    avg_context_len: int,
    *,
    kv_dtype_bytes: int = 2,
    activation_reserve: float = 0.06,   # 激活 + 通信 buffer 的经验余量
    fragmentation: float = 0.04,        # 分块管理下的内部碎片
) -> int:
    overhead = gpu_memory_bytes * (activation_reserve + fragmentation)
    available = gpu_memory_bytes - shape.weight_bytes - overhead
    if available <= 0:
        raise ValueError('权重本身就放不下，先考虑量化或者切分')

    per_seq = shape.kv_bytes_per_token(kv_dtype_bytes) * avg_context_len
    return int(available // per_seq)


llama_8b = ModelShape(layers=32, kv_heads=8, head_dim=128, weight_bytes=16 * GIB)
print(max_concurrency(llama_8b, 80 * GIB, avg_context_len=4096))
```

:::info{title="这个数字是上限，不是目标"}
算出来的并发是「显存装得下」的上限。真正能达到的吞吐还受调度开销、prefill 抢占、
以及 p99 延迟目标的约束，通常只能跑到上限的 60% 到 75%。
:::

## PagedAttention 省下的是哪一部分

上面的账算的是「实际用到的 token」。而连续分配的实现必须按**可能的最大长度**预留：

| 分配方式 | 单序列预留 | 实际使用 | 浪费 |
| --- | --- | --- | --- |
| 按 max_len=8192 连续预留 | 1024 MiB | 192 MiB | 81% |
| 按 max_len=4096 连续预留 | 512 MiB | 192 MiB | 62% |
| 分块（block=16 token） | 200 MiB | 192 MiB | 4% |

分块管理把浪费从「按最大长度预留」降到「最后一个块里的内零头」，这就是那篇论文里
2 到 4 倍吞吐提升的主要来源。它不是让 attention 算得更快，而是让同样的显存装下
更多序列。

块大小的取舍很直接：

- 块太小（4 token）：块表变长，索引开销和 kernel 里的 gather 变慢。
- 块太大（128 token）：回到接近连续分配的浪费水平。
- 实践中 16 或 32 是个合理区间。

## 几个反复踩到的坑

1. **忘了 prefix cache 也占显存。** 共享前缀的复用会把命中的块 pin 住，
   在系统 prompt 很长的场景里，这部分可以占到总 cache 的 20% 以上。
2. **FP8 KV Cache 不是免费的 2 倍。** 省一半显存，但要付 per-block 的
   scale 存储和量化/反量化开销，短序列上净收益可能是负的。
3. **`max_model_len` 设得过大。** 很多框架按它来推导块表尺寸和调度水位，
   设成 128k 而实际只用 4k，会凭空少掉可用并发。
4. **多卡下按单卡显存算。** TP 切分下每张卡只持有 $1/\text{TP}$ 的 KV head，
   所以单卡的 `kv_heads` 要先除以 TP size。

:::danger{title="别用 nvidia-smi 的数字反推"}
`nvidia-smi` 看到的是 allocator 持有的显存，不是实际用到的。框架几乎都会预占一大块
再自己管理，用它反推 KV Cache 占用只会得到一个和真相无关的数。
:::

## 小结

把 $2 N H d L b$ 这个式子记住，能省掉一多半关于「跑不跑得下」的争论。剩下的一半
是调度问题，那个要看 profile，算不出来。
