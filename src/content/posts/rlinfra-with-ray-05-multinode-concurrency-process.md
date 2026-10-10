---
title: 'Ray-Learn（五）：多机角色约束、并发池与进程监督'
description: >-
  前四篇把单机语义讲完了。这一篇补上多机上真正会疼的三块：节点角色怎么用
  自定义资源钉死、同一 actor 里长短任务怎么分池、以及 Ray 与子进程之间的
  责任边界。另附具名 actor 死亡后的调用语义。
pubDate: 2026-10-09
tags: ['RL', 'Tooling']
kind: 'Tutorial'
lang: 'zh'
density: 'dense'
readingMinutes: 20
---

这是 **Ray-Learn** 系列第五篇。前四篇主要在单机上把语义踩实；这一篇进入多机与生产形态：**资源落在哪个角色、并发怎么分池、服务进程谁来管**。

## 一、多机账本：看得见全部卡，反而更危险

单机调试时有一种偷懒做法：让 Ray 只「看见」一部分 GPU。例如推理节点 `ray start --num-gpus=0`，训练节点 `--num-gpus=8`。这样调度器永远不会把 `TrainActor` 放到推理卡上。

多机生产里这条路线经常不够用：

1. 推理节点的卡也要被 Ray 记账，才能统一做显存与调度；
2. 「少报卡」是一种**隐藏**，不是一种**约束**；
3. 一旦某次运维改了启动参数，约束就悄悄消失了。

于是更稳的形状是：**账本如实登记全部 GPU，再用自定义资源给节点打角色。**

```bash
# 训练节点
ray start --head --num-gpus=8 --resources='{"role_train":8,"nvlink_domain_0":8}'

# 推理节点
ray start --address=<head>:6379 --num-gpus=8 \
  --resources='{"role_infer":8,"nvlink_domain_1":8}'
```

```python
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init(address='auto')

# bundle 规格里写死角色，而不是事后靠运气
train_pg = ray.util.placement_group(
    bundles=[{'GPU': 1, 'CPU': 1, 'role_train': 1}] * 8,
    strategy='STRICT_PACK',
)
infer_pg = ray.util.placement_group(
    bundles=[{'GPU': 2, 'CPU': 1, 'role_infer': 2}] * 4,
    strategy='PACK',
)
ray.get([train_pg.ready(), infer_pg.ready()])
```

几个要点：

| 机制 | 作用 |
| --- | --- |
| `--num-gpus` | 账本额度，必须与规划一致 |
| `--resources='{"role_*":N}'` | 角色标签，可进 bundle |
| `STRICT_PACK` / `PACK` | 同机或尽量同机 |
| `capture_child_tasks` | 子进程继承 bundle 资源 |

:::warning{title="PACK 偏好不是硬约束"}
`strategy='PACK'` 只是偏好。池隔离必须靠**不同自定义资源**或**不同 placement group**。靠注释和运维习惯，迟早会踩到「训练跑到推理节点上」。
:::

多机验收仍然是那条断言，只是数字变成整个集群的规划额度：

```python
assert ray.cluster_resources()['GPU'] == 32
assert ray.cluster_resources()['role_train'] == 16
assert ray.cluster_resources()['role_infer'] == 16
```

后两条比第一条更有用：它们验证的是**角色拓扑**，不是单纯卡数。

### NCCL 启动失败的排查顺序

多机上最常见的是「进程都活着，就是不动」。建议顺序固定：

1. 极小 `all_reduce` 微基准，确认网络与 NCCL 初始化能通；
2. 核对 `MASTER_ADDR` / `MASTER_PORT` 是否跨机可达；
3. 核对 rank 映射是否稳定（第二篇的 bundle 重排序）；
4. 最后才怀疑模型并行配置。

真上大模型再调 NCCL，成本会高一个数量级。

## 二、并发：`max_concurrency` 不够用的时候

第三篇讲过：同步 actor 方法串行；阻塞在 C 扩展里的调用会饿死整个 actor。异步 actor 配合 `max_concurrency` 能并发 IO，但还有一个细分场景没展开——

**同一个 actor 里，既有长任务，又有短探活。**

典型例子是推理 Server actor：

- `generate` 可能跑几百毫秒到数秒；
- `health` 必须在百毫秒内返回，否则监控失效。

若两者共用一个并发池，长生成一多，健康检查就排队。miles 的做法是 **`concurrency_groups`**：

```python
import asyncio
import time

import ray


@ray.remote(concurrency_groups={'health': 1, 'generate': 4})
class Server:
    async def generate(self, i: int) -> str:
        await asyncio.sleep(1.2)
        return f'gen {i}'

    async def health(self) -> str:
        return 'healthy'


ray.init()
s = Server.remote()
t0 = time.time()
busy = [s.generate.remote(i) for i in range(3)]
print(ray.get([s.health.remote() for _ in range(3)]), f'{time.time()-t0:.2f}s')
ray.get(busy)
ray.shutdown()
```

语义拆开：

| 配置 | 含义 |
| --- | --- |
| `max_concurrency=N` | 整个 actor 最多 N 个协程 |
| `concurrency_groups={name: N}` | 多个独立执行池 |
| 方法进哪个池 | 创建时用 `method_concurrency_groups` 指定（miles） |

:::info{title="和 L05 的关系"}
L05 说的是「阻塞会饿死」；本节说的是「非阻塞但耗时的任务，也可能挤占关键路径」。前者靠拆 actor/线程，后者靠分池。
:::

verl 的 server 另一种取舍：把 `max_concurrency` 设得比引擎容量还高（例如 `max(1000, max_num_seqs+16)`），让引擎内部排队，Ray 只保证不把请求挡在门外。两种都能用，前提是**想清楚队列在哪一层**。

## 三、Ray 管资源，进程跑服务

推理引擎是「长 event loop + 多线程 CUDA」的服务。正确分工是：

| 层 | 职责 |
| --- | --- |
| Ray actor | 拿资源、起进程、探活、优雅退出 |
| 子进程 | 真正跑引擎 loop / CUDA / NCCL |

```python
import os
import subprocess

import ray


@ray.remote(num_cpus=0.5)
class ProcessSupervisor:
    def __init__(self):
        self.proc = None

    def spawn(self, cmd: list[str]) -> dict:
        self.proc = subprocess.Popen(cmd, env=os.environ.copy())
        return {'pid': self.proc.pid, 'cvd': os.environ.get('CUDA_VISIBLE_DEVICES')}

    def health(self) -> str:
        if self.proc is None:
            return 'not-started'
        rc = self.proc.poll()
        return 'running' if rc is None else f'exited:{rc}'

    def shutdown(self) -> str:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return 'stopped'
```

三个必须写进代码的细节：

1. **`placement_group_capture_child_tasks=True`**  
   不加的话，子进程会被当成新任务去抢资源，而不是继承父 actor 的 bundle。
2. **actor 活着 ≠ 子进程活着**  
   必须 `poll()` / 异步 reap + 有界健康探测。父进程被硬杀时，`finally` 不会跑。
3. **清理顺序**  
   TERM → 超时 → KILL → 确认无 GPU 占用 → 再释放 actor/PG。反过来会让残留进程占着已归还的卡。

### 第三种「共置」：fork

前几篇提过两种共置：

- 单 PG + offset（slime）；
- FusedWorker，多逻辑类合成一个 actor（verl）。

AReaL 还有第三种：**`fork_workers`**。在同一节点上通过 HTTP 向 guard 进程要端口、fork 新角色，**不新开 placement group**，直接复用该节点已占资源。

适合「节点已经占卡，还要再起一个旁路服务」的场景。代价是清理路径更复杂：要单独 kill fork 出来的进程树。

## 四、具名 actor 的死亡语义

第四篇讲了如何「找到」单例；这里补上「找到之后它死了怎么办」。

| 调用 | actor 已被 `ray.kill` | 含义 |
| --- | --- | --- |
| `actor.method.remote()` + `ray.get` | `RayActorError` / `ActorDiedError` | 进程没了 |
| `ray.get_actor(name)` | 通常仍返回句柄或报错（版本相关） | 名字还在不代表实例活 |
| 创建时 `max_restarts=-1` | 可能自动拉起 | 适合 env/router，不适合训练 rank |
| `lifetime='detached'` | driver 退出后仍在 | 跨作业单例，记得显式 kill |

工程建议：

1. 调用方按 `RayActorError` 走重建，不要按业务异常处理；
2. 具名服务要么配自动重启，要么由 Driver 负责探测与重建；
3. detached actor 用完必须杀，否则占资源直到集群回收。

## 五、作业生命周期

| 形态 | 入口 | 场景 |
| --- | --- | --- |
| 本地自起 | `ray.init()` | 学习、单机 |
| 接入已有集群 | `ray.init(address='auto')` | `ray start` 已起好 |
| 作业系统 | `JobSubmissionClient` / `ray job submit` | KubeRay、多租户 |
| 平台内自举 | rank0 `ray start --head` | Slurm gang 作业 |

```python
# 形状示意：作业系统侧
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient('http://127.0.0.1:8265')
# job_id = client.submit_job(entrypoint='python train.py', runtime_env={...})
# client.get_job_status(job_id)
# client.get_job_logs(job_id)
```

NeMo-RL 的 `nrl_k8s` 用 `JobSubmissionClient` 对接 KubeRay；AReaL 在 Slurm 里自举 Ray 集群；slime/miles 脚本层更常见 `ray job submit` CLI。

无论哪种入口，**第一道门禁都相同**：

```python
assert ray.cluster_resources()['GPU'] == expected
```

## 六、这一篇的 checklist

| 主题 | 必做 |
| --- | --- |
| 多机 | 全卡进账本 + 自定义角色资源进 bundle |
| 并发 | 长短任务分 `concurrency_groups` |
| 服务 | actor 管进程，子进程跑引擎 |
| 具名 | 按 `RayActorError` 重建；detached 显式回收 |
| 作业 | 统一入口 + 账本断言 |

下一篇是附录：把各框架真实用到的 Ray API 压成一张对账表，方便读源码时对照。
