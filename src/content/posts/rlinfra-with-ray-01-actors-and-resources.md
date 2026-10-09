---
title: 'Ray-Learn（一）：actor、资源账本与三个静默陷阱'
description: >-
  主流 RL 框架里有一大半把控制面建在 Ray 上，但它们几乎都不用 Ray 的上层库。
  这一篇从 task / actor 的语义讲起，把资源账本的「账面数字」和物理占用拆开，
  并用三个可复现的实验说明：多机 RL 里最常见的故障，往往不报错。
pubDate: 2026-10-09
tags: ['RL', 'Tooling']
kind: 'Tutorial'
lang: 'zh'
featured: true
density: 'dense'
readingMinutes: 18
---

先说结论：RL 基础设施的控制面，今天大部分长在 Ray 上，但用到的只是 Ray Core 里很窄的一层。窄，不等于简单——坑全在语义里。

这篇文章属于 **Ray-Learn** 系列。系列目标不是「学会 Ray」，而是**通过 Ray 读懂 RLinfra 的编排层**：资源怎么记、rank 怎么定、引擎怎么管、权重怎么同步。四篇读完，再去看 slime / verl / NeMo-RL / AReaL 的源码，会发现它们在解同一组问题，只是抽象层次不同。

## 一、先分清 task 和 actor

Ray 里可调度的基本单元就两类：

| 对比项 | task | actor |
| --- | --- | --- |
| 写法 | `@ray.remote def f()` | `@ray.remote class C` |
| 有没有状态 | 无状态，每次调用独立 | 有状态，活在实例里 |
| 调用形式 | `f.remote(x)` | `c.method.remote(x)` |
| 返回值 | `ObjectRef` | `ObjectRef` |

这里有个所有教程都会提、但很容易忘的点：

**`.remote()` 永远不返回结果，只返回句柄。**

```python
import ray

@ray.remote
def square(x: int) -> int:
    return x * x

@ray.remote
class Counter:
    def __init__(self):
        self.n = 0

    def inc(self) -> int:
        self.n += 1
        return self.n

ray.init()

ref = square.remote(7)      # 这是 ObjectRef，不是 49
print(type(ref))            # <class 'ray._raylet.ObjectRef'>
print(ray.get(ref))         # 49，要显式取值

c = Counter.remote()
print(ray.get(c.inc.remote()))  # 1
print(ray.get(c.inc.remote()))  # 2
ray.shutdown()
```

第二次 `inc` 返回 2，说明状态在 actor 进程里活下来了。task 做不到这件事——`square` 不知道上一次算过什么。

### 为什么编排层几乎全是 actor

看一下 RL 系统里需要「记住点什么」的角色：

- Driver 要记住启动流程走到哪一步
- RolloutManager 要持有推理引擎的句柄和当前权重版本
- 训练 rank 要持有模型、优化器、梯度缓冲

这些都不是纯函数。所以 slime 的 `TrainRayActor` / `RolloutManager` / `Lock`，NeMo-RL 的各种 worker，verl 的 `RayWorkerGroup` 底下，清一色是 actor。

但 actor 有一条默认语义，后面三个陷阱都从它长出来：

:::warning{title="默认串行"}
同一个 actor 上的方法调用**默认串行**。task 是并发的，actor 不是。想并发，要么开 `max_concurrency`（仅对 async 方法有效），要么拆成多个 actor。
:::

这一点看起来平平无奇。等看到第三节的饿死实验，就知道它有多致命。

## 二、`num_gpus` 是账本，不是绑卡

先说结论：`num_gpus=1` **不是**「绑定一张物理 GPU」，而是「在 Ray 的资源账本上登记 1 个 GPU 额度」。

这两件事在单机上几乎重合，所以很多人从来没区分过。一旦上多机，就会踩进 RLinfra 里最常见的一类静默失效。

具体说一下机制。Ray 调度器看到的「集群有多少卡」，来自 `ray start --num-gpus=N`（或 `ray.init(num_gpus=N)`）写进 GCS 的账面数字。它不会扫描 PCIe，也不会看 `nvidia-smi`。

```python
import ray

ray.init(num_gpus=2)          # 账上只记 2 张，哪怕机器插着 8 张
print(ray.cluster_resources()['GPU'])   # 2.0
ray.shutdown()
```

注意 `cluster_resources()` 的语义是**账本汇总**，不是硬件探测结果。后面 placement group、rank 映射、调度决策，全建立在这本账上。

### 两个直接推论

**第一，分数额度是合法的。** slime 的训练 actor 用 `num_gpus=0.4`，推理引擎占位用 `0.2`。这不是偷懒，而是表达「这张卡的账面额度被占住，但真正的 CUDA 上下文由引擎自己管理」。小于 1 的额度还能让多个 actor 在同一张卡上共置。

**第二，账本是会撒谎的。** 节点上要是还有别的进程占着卡——推理框架自己拉起的 engine、没退出的调试脚本——Ray 一概不知。它照样把任务派过去。

后果是：一切看起来正常，日志很干净，甚至显存占用也「合理」，但 GPU 已经在抢资源了。等发现的时候，通常是 loss 异常或者 OOM。

:::info{title="没有 GPU 的环境怎么练"}
把 `num_gpus` 换成自定义资源，例如 `ray.init(resources={"fake_gpu": 4})` 和 `@ray.remote(resources={"fake_gpu": 1})`。记账规则完全一样，差别只在真实 GPU 场景下 Ray 会额外设置 `CUDA_VISIBLE_DEVICES`。
:::

所以多机验收的第一条断言应该是：

```python
expected = 16  # 规划分给 Ray 的额度，不是资产清单上的卡数
assert ray.cluster_resources()['GPU'] == expected
```

这个数对不上，先别调 NCCL，先把账对齐。

## 三、陷阱一：超额创建，永久挂死

```python
import ray

ray.init(num_gpus=2)

@ray.remote(num_gpus=1)
class Worker:
    def ping(self):
        return 'ok'

workers = [Worker.remote() for _ in range(5)]   # 句柄全拿到了
print(ray.get([w.ping.remote() for w in workers]))  # 卡住
```

现象是：前两个 actor 创建成功，后三个在创建阶段排队。`.remote()` 只是**提交创建请求**，不保证立刻调度。而 actor 的坑在于——

:::danger{title="actor 创建即占资源，持有到死"}
句柄已经返回，但方法调用会一直等。Driver 若假设 `ray.get` 一定会返回，自身也会被拖住。
:::

对比 task 会清楚一些：

```python
@ray.remote(num_gpus=1)
def job(i):
    return i

# task 同样会排队，但它是「按调用排队」；
# actor 是「先创建并长期占用，再接受调用」。
# 后者更容易把资源锁死，也让超时语义更模糊。
```

**对策**有两条：

1. Driver 必须**监视 actor 创建**，而不是假设成功。可以用 `ray.wait` 轮询某个 ready 方法，或者给创建过程加超时。
2. 起 worker 之前先断言账本，见上一节。

很多「训练卡在启动、不报错」的问题，查到最后就是这句话：第 17 个 rank 在排队，前面 16 个都活着。

## 四、陷阱二：阻塞一个 actor，饿死它的全部调用

这是全系列最重要的一条。NeMo-RL 在 `refit_watchdog.py` 里把原因写得非常清楚：

> Ray runs a SYNC actor method directly in the event loop … a refit that blocks in NCCL **starves every other call to the same actor**. `max_concurrency` cannot help; it interleaves coroutines, and a coroutine blocked in C never yields.

拆开来说：

1. 同步 actor 方法直接跑在事件循环里；
2. `max_concurrency` 的作用是交织**协程**，不是让阻塞代码变非阻塞；
3. 调用一旦阻塞在 C 扩展里（NCCL、CUDA synchronize、底层 `ray.get`），协程让不出去；
4. 于是这个 actor 上的健康检查、取消、重启，全部排在后面。

实验可以很短：

```python
import time
import ray

@ray.remote
class Victim:
    def __init__(self):
        self.n = 0

    def inc(self):
        self.n += 1
        return self.n

    def block(self, sec: float):
        time.sleep(sec)       # 模拟 NCCL / 长推理
        return 'done'

ray.init()
v = Victim.remote()

blocked = v.block.remote(3.0)          # 先占住事件循环

t0 = time.time()
print(ray.get([v.inc.remote() for _ in range(10)]))
print(time.time() - t0)                # ≈ 3s，而不是毫秒级
ray.get(blocked)
ray.shutdown()
```

注意 `inc` 本身是微秒级的。它慢，不是因为它重，而是因为它排在一个阻塞调用后面。

### 架构上的结论

**rollout（长推理）和 trainer（长 NCCL）必须是不同的 actor。**

这不是性能优化，是正确性问题。如果两者共用一个 actor：

- 健康探测会被生成/训练堵死，故障发现变慢；
- 取消和超时路径会排在业务调用后面，等于没有；
- 更糟的是，监控看起来「全挂了」，实际上业务只是在长跑。

异步 actor（`async def` 方法）可以并发 IO，但同样挡不住 C 层阻塞。如果必须在 actor 内做重计算，要么拆 actor，要么把阻塞调用丢到线程/子进程。NeMo-RL 的 `refit_watchdog` 选择把 NCCL 放到守护线程里跑，事件循环保持可响应，就是这个思路。

## 五、三条陷阱串起来

| 陷阱 | 现象 | 根因 | 对策 |
| --- | --- | --- | --- |
| 账本 ≠ 绑卡 | 调度到忙节点，不报错 | 调度器只认账面额度 | 统一进账本，或 `--num-gpus=0` 排除旁路节点 |
| 超额创建 | `ray.get` 永不返回 | actor 创建即占资源 | 监视创建；先断言账本 |
| 阻塞饿死 | 监控与业务互相卡死 | sync 方法跑在事件循环里 | 长任务拆 actor / 子进程 |

这三条有一个共同特征：**都不以异常形式出现**。它们表现为「慢」「卡住」「结果不对」，日志却往往一片祥和。这也是 RLinfra 调试比普通后端更费劲的原因之一。

## 六、怎么把这些用起来

如果目标是读框架源码，下一篇之前建议先做三件事：

1. 把第二节的 `num_gpus` 实验跑通，确认理解「账本」语义；
2. 把第三节的超额创建实验跑通，观察第几个 actor 开始排队；
3. 把第四节的饿死实验跑通，记下 `inc` 的实际等待时间。

跑完再去看 slime 的 `slime/ray/actor_group.py` 和 NeMo-RL 的 `refit_watchdog.py`，很多设计选择会立刻变得具体：为什么 rollout 要独立成 actor，为什么 health check 不能和 refit 挤在一起，为什么 `num_gpus` 可以是 0.4。

下一篇讲 placement group：资源从「几张卡」推进到「卡落在哪」，以及为什么 bundle 顺序不稳定是一个正确性问题。
