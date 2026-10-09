---
title: 'Ray-Learn（三）：异步语义、引擎子进程与数据面'
description: >-
  推理引擎为什么不能塞进 actor 的 event loop，ObjectRef 为什么会自动解引用，
  故障该分成哪三类，以及 ray.util.collective 在权重广播里的正确姿势。
  这一篇覆盖 RL 系统里最容易写出「看起来能跑」的那部分代码。
pubDate: 2026-10-09
tags: ['RL', 'Tooling']
kind: 'Tutorial'
lang: 'zh'
density: 'dense'
readingMinutes: 18
---

这是 **Ray-Learn** 系列第三篇。前两篇讲「谁、在哪」，这一篇讲「怎么传、怎么活下去」。

RL 系统里有两条链路特别耗时间：一条是**推理引擎的生命周期**，另一条是**数据与权重在角色之间的流动**。它们的共同点是：写错了不会立刻报错，而是慢慢变成奇怪的性能问题或结果问题。

## 一、引擎必须是子进程

SGLang、vLLM 这类推理引擎，本质上是「长 event loop + 多线程 CUDA + 自己的调度器」。把它直接塞进 Ray actor，有三个硬伤。

**第一，事件循环互相污染。** Ray 的 actor 方法调用会排在引擎的 loop 后面。引擎一忙，健康检查就堵住；健康检查一堵住，故障发现就延迟；故障发现一延迟，故障恢复就没意义了。

**第二，信号和清理不可靠。** 进程被 `kill -9` 时 `finally` 不会跑。GPU 上下文残留、端口残留、共享内存残留，都会毒死下一次启动。

**第三，阻塞饿死。** 上一篇讲过，同步方法阻塞在 C 扩展里会拖住整个 actor。引擎的 generate、CUDA synchronize 正是这种阻塞。

所以标准形状是：**actor 持资源，子进程跑服务**。

```python
import os
import subprocess
import sys

import ray

@ray.remote(num_cpus=0.5)
class EngineSupervisor:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def spawn(self, cmd: list[str]) -> dict:
        self.proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),   # 继承 Ray 注入的 CVD 等
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return {'pid': self.proc.pid}

    def health(self) -> str:
        if self.proc is None:
            return 'not-started'
        rc = self.proc.poll()
        return 'running' if rc is None else f'exited:{rc}'

    def shutdown(self) -> str:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()          # 先 TERM
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()           # 超时再 KILL
        return 'stopped'

ray.init()
sup = EngineSupervisor.remote()
print(ray.get(sup.spawn.remote([sys.executable, '-c', 'import time; time.sleep(30)'])))
print(ray.get(sup.health.remote()))
print(ray.get(sup.shutdown.remote()))
ray.shutdown()
```

分工很清楚：

| 层 | 职责 |
| --- | --- |
| Ray actor | 拿资源、起进程、探活、优雅退出 |
| 子进程 | 真正跑引擎 event loop / CUDA |

### 两个必须记住的细节

:::warning{title="actor 活着，不等于子进程活着"}
必须异步 reap + 有界健康探测。`poll()` 是非阻塞的，适合放在 actor 方法里；`wait()` 会阻塞，不适合直接调用。
:::

:::info{title="capture_child_tasks"}
创建引擎 actor 时记得 `placement_group_capture_child_tasks=True`。否则 torch.distributed 或引擎 fork 出的子进程会被当成新任务，去和父进程抢资源。AReaL 的 launcher 明确开了这个选项，slime 的 engine 也是。
:::

清理顺序也值得写死：**TERM → 超时 → KILL → 确认无 GPU 占用 → 再释放 actor/PG**。不要反过来。反过来的话，资源已经还给调度器了，残留进程还在占卡，下一次调度又是一次静默失效。

AReaL 把这条路走到另一端：Ray 只当进程启动器，业务 RPC 全走 HTTP。逻辑是——服务本来就有自己的 loop，就不要再包一层 actor loop。这是同一问题的另一种解法。

## 二、ObjectRef 会自动解引用

rollout 输出一批 token id，可能是几十 MB 到几 GB。正确姿势是 `ray.put` 进 Object Store，再传引用。但这里有个很隐蔽的语义：

```python
import ray

@ray.remote
def consume_value(data) -> int:
    return len(data)                        # 收到的是「值」

@ray.remote
def consume_box(box) -> int:
    return len(ray.get(box['inner']))       # 收到的是「引用」

ray.init()
payload = list(range(200_000))
ref = ray.put(payload)

# 顶层 ObjectRef 参数会被自动解引用
print(ray.get(consume_value.remote(ref)))          # 200000

# 包一层容器就不解引用
print(ray.get(consume_box.remote({'inner': ref}))) # 200000
ray.shutdown()
```

规则其实只有两条：

| 传法 | task 收到什么 | 拷贝发生在何时 |
| --- | --- | --- |
| `f.remote(ref)` | 解开后的值 | 调度前，由 Ray 完成 |
| `f.remote({'r': ref})` | 仍是 ObjectRef | 显式 `ray.get` 时 |
| `f.remote(big_list)` | 值，序列化进参数 | 提交任务时 |

slime 里的 `Box(ray.put(...))` 不是过度设计，而是**为了把 ref 当句柄传递**，让 trainer 在需要的时刻再取。如果直接传 `ref`，Ray 会在调度前帮你解开，延迟加载就没有意义了。

:::info{title="Object Store 会满"}
plasma 有容量上限。打满之后通常是变慢，而不是抛异常。监控 `plasma_bytes`；超大张量走 NIXL、Mooncake 或 ZMQ。TransferQueue 的思路就是：Ray 只传元数据，张量走 ZMQ / RDMA。
:::

顺带一提，`ray.put` 本身也有成本。小对象频繁 put 会把 Object Store 打成碎片场。经验法则是：**批级别 put，不要样本级别 put**。

## 三、故障语义：三类异常

| 异常 | 触发 | RL 侧动作 |
| --- | --- | --- |
| `RayTaskError` | task / 方法 raise | 业务错误，一般不重建 |
| `RayActorError`（含 `ActorDiedError`） | `ray.kill`、进程死 | **重建** |
| `GetTimeoutError` | `ray.get(timeout=)` | 任务可能还活着，要 `cancel` 或放弃 |

```python
import time
import ray
from ray.exceptions import GetTimeoutError, RayActorError

ray.init()

@ray.remote
def slow():
    time.sleep(5)
    return 'done'

@ray.remote
class Worker:
    def ping(self):
        return 'pong'

w = Worker.remote()
ray.get(w.ping.remote())

ref = slow.remote()
try:
    ray.get(ref, timeout=0.3)
except GetTimeoutError:
    ray.cancel(ref)          # 不取消的话，它还会继续占资源

ray.kill(w)
time.sleep(0.2)
try:
    ray.get(w.ping.remote())
except RayActorError as e:
    print(type(e).__name__)  # ActorDiedError
ray.shutdown()
```

分类的意义在于：**重建策略不同**。

- `RayTaskError`：调用方逻辑问题，重建 actor 没用，可能反复失败；
- `RayActorError`：进程没了，不重建就没有服务；
- `GetTimeoutError`：最暧昧，任务可能只是慢。要么延长等待，要么 `ray.cancel`，要么放弃这个调用但保留 actor。

### 异常类型在跨 actor 时会丢

NeMo-RL 踩过这个坑：`RefitAborted` 从引擎进程传到 Ray actor 后，变成 `Exception("Call to ... failed")`。于是 `except RefitAborted` 成了死代码，watchdog 明明 abort 了，run 还是卡住。

对策有两个：

1. 看字符串里的固定 token（NeMo-RL 的做法）；
2. 事先约定错误码协议，不要依赖异常类型跨边界。

自动重启的角色（环境服务、路由）可以用 `max_restarts=-1`。训练 rank 通常手动管——重启代价大，而且要整组 re-rendezvous，不是拉起单个进程那么简单。

## 四、队列：从 ref 列表到消息泵

| 结构 | 适合 | 注意 |
| --- | --- | --- |
| `list[ObjectRef]` | 一次性 fan-out / fan-in | 无背压 |
| `ray.util.queue.Queue` | 生产者-消费者，`maxsize` 背压 | 实现是 actor |
| 自写 MessageQueue actor | 自定义协议、多播 | verl fully-async 用这个 |
| ZMQ / HTTP | 大张量、跨框架 | Ray 只协调 |

```python
from ray.util.queue import Queue

q = Queue(maxsize=8)
q.put_nowait(1)
print(q.get_nowait())
```

什么时候需要队列，而不是 ref 列表？

一个简单判据：**如果生产者和消费者的速率差一个数量级，就需要队列。** ref 列表是「我发出去了」，队列是「你慢慢取」。verl 的全异步策略里有一个 `MessageQueue` actor，生产者往里塞 rollout 结果，trainer 按自身节奏取。全局锁被换成了队列水位。

## 五、权重广播：`ray.util.collective`

训练 step 结束后，要把新权重送到所有推理引擎。常见三条路：

| 路径 | 谁在用 | 特点 |
| --- | --- | --- |
| 磁盘 / NIXL | slime disk 模式 | 解耦，带宽差一截 |
| torch.distributed 自建组 | slime、NeMo-RL | 跨 train + engine 的大组 |
| `ray.util.collective` | verl | Ray 托管 NCCL 组，适配 actor 边界 |

```python
import ray
import ray.util.collective as collective

@ray.remote
class WSync:
    def __init__(self, rank: int, world: int, name: str):
        self.rank = rank
        self.world = world
        self.name = name

    def init_group(self, uid):
        collective.init_collective_group(
            world_size=self.world,
            rank=self.rank,
            backend='gloo',     # GPU 上改 'nccl'
            name=self.name,
        )
        return f'rank {self.rank} ready'

    def broadcast_from_root(self):
        import torch
        t = (
            torch.tensor([1.0, 2.0, 3.0, 4.0])
            if self.rank == 0
            else torch.zeros(4)
        )
        collective.broadcast(t, root_rank=0, name=self.name)
        return t.tolist()

ray.init()
workers = [WSync.remote(r, 2, 'wsync') for r in range(2)]
uid = collective.get_unique_id()   # 真实系统经具名 actor 发布
ray.get([w.init_group.remote(uid) for w in workers])
print(ray.get([w.broadcast_from_root.remote() for w in workers]))
ray.shutdown()
```

verl 的 `NCCLIDStore` 就是把 `get_unique_id()` 的结果放进具名 actor，其他 rank 用 `ray.get_actor` 去取。这是「命名服务 + 集合通信」的组合拳。

:::danger{title="入队顺序必须确定"}
NCCL 对操作顺序敏感。权重切 bucket 之后，所有 rank 的 enqueue 顺序必须一致，否则死锁。verl 在源码里用注释钉死了这一点。这不是风格问题，是能不能跑起来的问题。
:::

slime 另外有一个 24 行的 `Lock` actor：权重广播与生成不能并发，于是用分布式锁把 `pause → flush → update → resume` 串起来。这种锁值得手写——需要的只是一次互斥，没必要引依赖。

## 六、版本屏障：权重更新的完整形状

把上面几块拼起来，权重更新不是「调一个函数」，而是一段有序编排。slime 的 disk 模式大致是：

```text
1. onload_weights()                         # 有 offload 时先唤醒
2. get_updatable_engines_and_lock()         # 取引擎 + 加锁
3. engine.pull_weights(version)             # 可与生成重叠的准备
4. engine.pause_generation()
5. engine.flush_cache()
6. engine.update_weights_from_disk(...)
7. 校验 weight_version == 预期
8. engine.continue_generation()
```

第 3 步值得单独看：slime 注释明确写了，pull 是 disk-only，可以提到 pause 之前，和生成重叠。这是**免费的吞吐收益**，代价只是把更新流程拆细一点。

## 七、这一层的 checklist

| 主题 | 必做 |
| --- | --- |
| 引擎 | 子进程 + `capture_child_tasks` + 异步健康探测 |
| 大对象 | `ray.put` + 容器包 ref；监控 plasma |
| 故障 | 分类捕获；`kill` / `timeout` / 业务错误分开 |
| 权重 | 版本屏障；广播顺序确定；与生成互斥 |

下一篇收尾，讨论：runtime_env 的覆盖顺序、具名服务与拓扑调度，以及五种编排范式——从 slime 的 1,228 行，到 AReaL 的两万五千行，中间的取舍就是 RLinfra 的设计空间。
