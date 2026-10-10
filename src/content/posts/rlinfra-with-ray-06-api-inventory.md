---
title: 'Ray-Learn（六）：附录 · Ray API 对账与源码阅读地图'
description: >-
  把 slime / miles / verl / AReaL / NeMo-RL / prime-rl / TransferQueue 里真实用到的
  Ray API 压成一张对账表：谁在用、为什么用、对应哪一篇。并给出按框架的阅读顺序，
  以及明确可以不学的部分。
pubDate: 2026-10-09
tags: ['RL', 'Tooling']
kind: 'Reference'
lang: 'zh'
density: 'dense'
readingMinutes: 14
---

这是 **Ray-Learn** 系列附录。前五篇讲语义与架构；这一篇给一张**源码对照表**，方便读框架时按 API 反查「它在解决什么问题」。

## 一、谁在用 Ray

| 框架 | Ray 程度 | 编排定位 | 备注 |
| --- | --- | --- | --- |
| NeMo-RL | 最重 | Ray 是唯一编排层 | 不用 torchrun |
| verl | 重 | HybridFlow，Ray 藏在编程模型后 | `single_controller` |
| miles | 重 | slime 分叉，加 named actor / cell 容错 | 约 5k 行 |
| slime | 中 | 最小编排 | **1,228 行** |
| TransferQueue | 中 | Ray 控制面，ZMQ 张量面 | 可复用数据面 |
| AReaL | 刻意薄 | 只做 launcher + 命名服务 | 业务走 HTTP |
| prime-rl | 无 | asyncio + subprocess | 反例参考 |

规律：**抽象越多，自建代码越多。** Ray 用得越「薄」，HTTP/RPC/自研调度就越厚。

## 二、API 面对账

按主题归类。括号里是出现较多的框架。

### 1. 基本模型（L01）

| API | 用途 |
| --- | --- |
| `@ray.remote` / `.remote()` | 定义与调用 task/actor |
| `ray.get` | 取值、同步屏障 |
| `ObjectRef` / `ActorHandle` | 异步句柄 |
| `ray.put` | 大对象进 Object Store |

### 2. 资源账本（L02）

| API | 用途 | 证据 |
| --- | --- | --- |
| `num_gpus` / `num_cpus`（含 0.1 / 0.2 / 0.4） | 账面额度 | slime / miles / TQ |
| `cluster_resources` / `available_resources` | 观测账本 | 几乎所有 |
| `ray start --num-gpus` | 集群层记账 | 多机脚本 |
| 自定义资源 `resources={"role_train":1}` | 角色/拓扑 | NeMo-RL / AReaL |

### 3. Placement Group（L02 / L03 / L05）

| API | 用途 | 证据 |
| --- | --- | --- |
| `placement_group(bundles, strategy=PACK/STRICT_PACK/SPREAD)` | gang 调度 | slime / verl / AReaL |
| `pg.ready()` / `ray.wait` 轮询 | 就绪屏障 | slime / AReaL |
| `PlacementGroupSchedulingStrategy` | 钉进 bundle | 各家 |
| `placement_group_capture_child_tasks=True` | 子进程继承资源 | slime engine / AReaL |
| `placement_group_table` / `get_current_placement_group` | 读物理布局 | verl / NeMo-RL |
| `remove_placement_group` | 故障清理 | verl / AReaL |
| 单 PG + offset 切分 | 共置/分离一张表 | slime |

### 4. 调度与拓扑（L17 / 第五篇）

| API | 用途 | 证据 |
| --- | --- | --- |
| `NodeAffinitySchedulingStrategy(soft=False)` | 钉节点 | miles pin-head / verl 局部性 |
| `list_nodes` + `is_head_node` | 找 head | miles |
| `ray.nodes()` | 存活过滤 | slime HTTP 舰队 |
| `nvlink_domain_*` / `topo_rank` | NVLink 域 | NeMo-RL |

### 5. 身份与 rendezvous（L06 / L08）

| API | 用途 | 证据 |
| --- | --- | --- |
| `ray.get_gpu_ids()` | rank ↔ 设备 | slime / miles / verl / NeMo-RL |
| `get_node_ip_address()` | MASTER_ADDR | 几乎所有 |
| `get_runtime_context()` | node_id / job_id | miles / verl |
| 手工 `MASTER_ADDR/PORT` | torch 进程组 | slime 10 行基类 |

### 6. 生命周期与故障（L04 / L12）

| API | 用途 | 证据 |
| --- | --- | --- |
| `name=` / `namespace=` / `get_actor` | 单例查找 | miles / TQ / AReaL |
| `lifetime='detached'` / `get_if_exists` | 跨 driver、幂等 | AReaL KV |
| `ray.kill(no_restart=True)` | 释放/隔离 | 各家 |
| `max_restarts=-1` / `max_task_retries=-1` | 自动拉起 | NeMo-RL env/router |
| `RayActorError` / `RayTaskError` / `GetTimeoutError` | 分类捕获 | miles / AReaL |
| `ray.cancel` / `ray.wait(timeout=)` | 有界等待 | AReaL / slime |

### 7. 并发（L05 / L18）

| API | 用途 | 证据 |
| --- | --- | --- |
| `max_concurrency` | async actor 上限 | verl server / NeMo-RL env |
| `concurrency_groups` | 长短任务分池 | miles |
| `async def` 方法 + await ObjectRef | IO 并发 | slime HTTP poster |
| `async_*` 返回 ref 列表 | 流水线接缝 | slime `actor_group` |

### 8. 数据面（L11 / L13）

| API | 用途 | 证据 |
| --- | --- | --- |
| `ray.put` + 容器包 ref | 延迟加载 | slime `Box` |
| `_tensor_transport="nixl"` | 大张量绕 plasma | slime |
| 可插拔 object store | Ray/Mooncake 后端 | miles |
| `ray.util.queue.Queue` | 背压队列 | NeMo-RL / verl |
| 自写 MessageQueue actor | 自定义协议 | verl fully-async |
| ZMQ / HTTP 传张量 | 离开 Ray 数据面 | TQ / AReaL |

### 9. 权重同步（L09 / L14）

| API | 用途 | 证据 |
| --- | --- | --- |
| `ray.util.collective` | Ray 托管 NCCL 组 | verl checkpoint_engine |
| 具名 actor 发 NCCL id | 跨 actor 拓扑 | verl `NCCLIDStore` |
| torch.distributed 跨 actor 组 | train→engine 广播 | slime / NeMo-RL |
| 可重建 process group | actor 重启后恢复 | slime `reloadable_process_group` |
| 24 行 Lock actor | 更新互斥 | slime |
| HTTP 回调更新 | 服务化 | AReaL |

### 10. 环境与作业（L09 / L15 / L19）

| API | 用途 | 证据 |
| --- | --- | --- |
| `runtime_env={"env_vars"}` | 按 actor 注入 | 几乎所有 |
| `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES=1` | 自管 CVD | slime / verl |
| `RAY_USE_UVLOOP=0` | 修 async 间歇故障 | slime |
| `NCCL_CUMEM_ENABLE=0` | 与 SGLang 一致 | slime |
| `JobSubmissionClient` / `ray job submit` | 作业生命周期 | NeMo-RL / 脚本 |
| `ray_bootstrap`（rank0 head） | 平台内自举 | AReaL |

## 三、按框架的源码入口

| 想读懂… | 先读 | 对应篇 |
| --- | --- | --- |
| 最小编排 | `slime/slime/ray/` 1,228 行 | 一 / 二 / 三 |
| 声明式分发 | `verl/single_controller/{base,ray}/` | 四 |
| 教科书分层 | `NeMo-RL/nemo_rl/distributed/` | 一 / 五 |
| 薄 Ray 服务化 | `AReaL/areal/infra/{launcher,scheduler}/` | 五 |
| 无 Ray 对照 | `prime-rl/src/prime_rl/orchestrator/` | 四 |
| 数据面 | `TransferQueue/transfer_queue/` | 三 |

### slime 推荐阅读序

| 序 | 文件 | 行数 | 看什么 |
| ---: | --- | ---: | --- |
| 1 | `ray_actor.py` | 10 | rendezvous 基类 |
| 2 | `utils.py` | 75 | env 注入 + Lock |
| 3 | `train_actor.py` | 126 | `get_gpu_ids` → local rank |
| 4 | `placement_group.py` | 253 | 单 PG + bundle 重排 |
| 5 | `actor_group.py` | 269 | 扇出、async_train 返回 ref |
| 6 | `rollout.py` | 495 | 引擎管理与数据转换 |

## 四、可以不学的

```
ray.serve   ray.train   ray.data   ray.tune   RLlib   ActorPool   Compiled Ray DAG
```

主流 RL 框架零使用。编排层只做三件事：**拉起、排序、出错能看见。**

## 五、系列索引

| 篇 | 主题 |
| --- | --- |
| [一](/blog/posts/rlinfra-with-ray-01-actors-and-resources/) | actor、资源账本、三个静默陷阱 |
| [二](/blog/posts/rlinfra-with-ray-02-placement-and-rank/) | PG、GPU 身份、rank 稳定 |
| [三](/blog/posts/rlinfra-with-ray-03-async-engines-data-plane/) | 异步、引擎子进程、数据面 |
| [四](/blog/posts/rlinfra-with-ray-04-paradigms-and-production/) | 生产 API、五种编排范式 |
| [五](/blog/posts/rlinfra-with-ray-05-multinode-concurrency-process/) | 多机角色、并发池、进程监督 |
| 本篇 | API 对账与源码地图 |

配套练习仓库仍保留 20 课可运行脚本与完整实验记录格式；本附录适合当读源码时的速查表。
