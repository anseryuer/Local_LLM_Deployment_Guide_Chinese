# 本地部署大语言模型的中文教学

> 最后一次大修：2026 年 8 月 30 日
> 本文优先解释计算过程和硬件限制。具体型号、软件版本和命令可能继续变化。

Hello！大家好，这里是 Anser！

这是一篇关于本地部署大语言模型（LLM）的教程。

内容包括本地聊天、代码与文档处理，也包括为几十到几百名用户部署内部推理服务。阅读本文不要求机器学习或高性能计算基础。

LLM 的硬件和软件仍在快速更新。如果你发现事实错误或过时内容，欢迎提交 Issue 或 PR。

赞助我：[Buy me a coffee](https://buymeacoffee.com/ansertbyu)

也感谢 [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) 中公开硬件测试、运行参数和故障排查过程的各位。

---

## 目录

- [快速选择部署工具](#快速选择部署工具)
- [第一部分：LLM 到底在干什么？](#第一部分llm-到底在干什么)
  - [Token、Transformer 与自回归生成](#tokentransformer-与自回归生成)
  - [Prefill 和 Decode](#prefill-和-decode)
  - [KV cache：模型为什么不用每次从头算？](#kv-cache模型为什么不用每次从头算)
  - [Prefix caching：复用相同前缀的 KV cache](#prefix-caching复用相同前缀的-kv-cache)
  - [MTP 与 speculative decoding](#mtp-与-speculative-decoding)
- [第二部分：决定速度的到底是什么？](#第二部分决定速度的到底是什么)
  - [容量、带宽和算力](#容量带宽和算力)
  - [Latency 与 Throughput](#latency-与-throughput)
  - [模型到底要占多少内存？](#模型到底要占多少内存)
  - [MoE：总参数很大，激活参数很小](#moe总参数很大激活参数很小)
- [第三部分：精度与量化](#第三部分精度与量化)
  - [FP4 为什么必须缩放？](#fp4-为什么必须缩放)
  - [MXFP4 与 NVFP4](#mxfp4-与-nvfp4)
  - [不同 GPU 架构支持什么精度？](#不同-gpu-架构支持什么精度)
  - [训练和推理应该选什么精度？](#训练和推理应该选什么精度)
- [第四部分：如何为 LLM 配一台电脑](#第四部分如何为-llm-配一台电脑)
- [第五部分：如何选择模型](#第五部分如何选择模型)
  - [先确定能运行的候选范围](#先确定能运行的候选范围)
  - [排行榜的适用范围](#排行榜的适用范围)
  - [LMArena](#lmarena)
  - [Artificial Analysis](#artificial-analysis)
  - [建立自己的 private benchmark](#建立自己的-private-benchmark)
  - [Reasoning、RLVR 与真实成本](#reasoningrlvr-与真实成本)
  - [模型许可证](#模型许可证)
- [第六部分：部署方法](#第六部分部署方法)
- [第七部分：vLLM、PagedAttention 与高并发](#第七部分vllmpagedattention-与高并发)
- [第八部分：怎样提高 KV cache 命中率](#第八部分怎样提高-kv-cache-命中率)
- [第九部分：三套实际方案](#第九部分三套实际方案)
- [常见误区](#常见误区)
- [延伸阅读与资料来源](#延伸阅读与资料来源)

---

## 快速选择部署工具

先按照使用场景选择推理程序：

| 使用场景 | 程序 |
|---|---|
| 第一次部署，希望使用图形界面 | [LM Studio](https://lmstudio.ai/) |
| 想用一条命令启动模型，给别的软件调用 | [Ollama](https://ollama.com/) |
| Windows、macOS、Linux 都想跑，或者只有 CPU/核显 | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| 有 NVIDIA/AMD GPU，想给多人提供高吞吐 API | [vLLM](https://github.com/vllm-project/vllm) |
| 只使用 NVIDIA GPU，模型和硬件固定，可以投入时间编译和调优 | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| 公司需要经过验证的容器、监控、生命周期和支持 | [NVIDIA NIM](https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html) |
| Intel CPU、核显或独显，且需要 Intel 官方路线 | [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) |

第一次测试可以使用 7B～14B 模型的 Q4 或 Q5 量化版本，并把上下文长度设为 4K～8K。这个配置先把权重和 KV cache 占用控制在较小范围，便于分别排查模型兼容性、GPU 后端和内存容量。基础配置运行正常后，再增加模型大小和上下文长度。

---

## 第一部分：LLM 到底在干什么？

### Token、Transformer 与自回归生成

LLM 的输入和输出单位是 Token，而不是字符或单词。

比如 `unbelievable` 可能被切成一个或几个 Token；中文的一个汉字有时是一个 Token，有时多个字会合成一个 Token。不同模型使用的 tokenizer 也不同，所以两个模型都显示 50 token/s，并不代表它们每秒生成的中文字符一定一样多。

Tokenizer 通常不是神经网络。它把文本转换成整数 ID 序列；模型的 embedding layer 再把每个 ID 映射为向量。

目前绝大多数聊天 LLM 的主体依然属于 decoder-only Transformer。Transformer 最初发表于 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。3Blue1Brown 的 [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) 提供了不要求数学基础的可视化说明。

![Transformer 架构示意图](image-3.png)

自回归生成按以下步骤进行：

1. 给模型一段已有 Token；
2. 模型预测下一个 Token 的概率；
3. 按照 temperature、top-p 等规则选出一个；
4. 把这个新 Token 接回末尾；
5. 再预测下一个。

因此，普通 LLM 每次 forward pass 只确定一个新 Token。生成 1000 个 Token，通常需要约 1000 次顺序执行的解码步骤；后面的步骤必须等待前面的 Token 确定，无法全部并行。

### Prefill 和 Decode

一次 LLM 请求分为两个阶段：

**Prefill（预填充）**：模型一次性读取你的 system prompt、聊天记录、文档和问题，为这些输入 Token 计算中间状态，并生成第一个新 Token。

**Decode（解码）**：之后一个 Token 一个 Token 地继续生成答案。

Prefill 和 Decode 对硬件的使用方式不同：

- Prefill 同时处理输入中的许多 Token。线性层可以执行较大的矩阵乘法，GPU 的矩阵单元利用率较高，所以这个阶段通常首先受计算吞吐限制。Attention 的计算量还会随输入长度近似按平方增长，具体增长方式取决于 full attention、sliding-window attention 或其他稀疏结构。
- Decode 每一步通常只加入一个 Token。batch size 为 1 时，线性层接近矩阵-向量乘法：为了生成一个 Token，需要读取各层的大部分权重，但每个权重只参与很少的运算，所以这个阶段通常首先受内存带宽限制。

以一个 BF16 权重矩阵为例，读取一个权重需要 2 bytes，一次乘加约产生 2 FLOPs。batch size 为 1 时，忽略 cache 后的算术强度大约只有 1 FLOP/byte。H100 的 FP16/BF16 Tensor Core 计算吞吐与 HBM 带宽之比是数百 FLOPs/byte，因此这种工作负载远未达到计算上限，内存带宽会先被占满。batch size 增大后，同一份权重可以同时服务多个 Token，算术强度近似随 batch 增长，工作负载才可能转为 compute-bound。

因此，评估单用户生成速度时要同时看模型实际权重大小和内存带宽；评估长输入处理或高并发吞吐时，还要看低精度矩阵计算吞吐与 batch 调度效率。

### KV cache：模型为什么不用每次从头算？

在 causal Transformer 中，前面的 Token 看不到未来。第 100 个 Token 的出现，不会回头改变第 1～99 个 Token 已经算出的 Key 和 Value。

生成第 101 个 Token 时，不需要重新计算前 100 个 Token 在各层的 Key 和 Value。这些保存在内存中的张量称为 **KV cache**。

没有 KV cache 时，第 `t` 个生成步骤需要重新计算前 `t-1` 个 Token 在每一层的 Key 和 Value。有 KV cache 时，这些张量只在对应 Token 第一次进入模型时计算一次；以后直接从显存或内存读取。

KV cache 的容量随上下文长度、并发序列数和每个元素的字节数线性增长。

对于标准 MHA/GQA，可按下式估算：

```text
KV cache 字节数 ≈ 2 × 层数 × KV head 数 × head dimension
                 × 每个元素字节数 × 已缓存 Token 数 × 并发序列数
```

前面的 `2` 是 K 和 V 两份。

例如一个 32 层、8 个 KV heads、head dimension 为 128 的 GQA 模型，用 BF16 保存 KV，单个 32K 上下文大约需要：

```text
2 × 32 × 8 × 128 × 2 bytes × 32768 ≈ 4 GiB
```

该结果不包括模型权重、运行时 workspace 和其他请求的 KV cache。

因此，20GB 权重不能可靠地部署在只剩 4GB 空间的 24GB 显卡上并同时使用很长上下文。以上述模型为例，单个 32K BF16 KV cache 已经约为 4GiB；128K 时约为 16GiB。并发 8 个相同长度的序列还要再乘 8。

现代模型使用 GQA、MQA 或 MLA 减少每个 Token 保存的 Key/Value 数量。部分运行时还支持 FP8、INT8 或 NVFP4 KV cache。低精度 KV cache 能减少容量和带宽需求，但可能影响长上下文准确率；如果运行时缺少对应低精度 attention kernel，它也可能只节省容量而不提高速度。

### Prefix caching：复用相同前缀的 KV cache

KV cache 可以在同一次生成中使用，也可以跨请求复用。

假设 100 个员工都在询问同一本 200 页的公司手册：

```text
[完全相同的 system prompt]
[完全相同的公司手册]
[每个人不同的问题]
```

前两部分的 Token 前缀完全一致，那么它们对应的 KV cache 也一致。服务器第一次算完后，后续请求可以直接复用这部分缓存，只计算各自不同的问题。

这种跨请求复用称为 **prefix caching（前缀缓存）**。它减少命中部分的 prefill 计算，不减少新输出 Token 的 Decode 计算。vLLM 的 [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) 实现了这一功能。

复用条件通常是模型、tokenizer、Token ID 序列和位置完全相同。文本语义相似不构成命中；时间戳、随机 ID、JSON 字段顺序或聊天模板中的任一差异，都会使差异位置之后的前缀无法复用。

### MTP 与 speculative decoding

普通自回归生成每次只确认一个 Token，因此输出长度为 `N` 时至少需要 `N` 个顺序解码步骤。

**Multi-Token Prediction（MTP）** 的训练目标，是让额外的预测模块学习预测更远的未来 Token，而不只学习紧邻的下一个 Token。公开工作主要使用它完成两件事：

1. 训练时给模型更多学习信号，可能提升主模型能力；
2. 推理时使用 MTP 模块提出未来若干 Token，供 speculative decoding 验证。

Speculative decoding 的流程如下：

1. 较小的 draft model 或 MTP head 提出接下来若干 Token；
2. 目标模型用一次并行 forward 验证这些候选 Token；
3. 接受符合主模型分布的部分，拒绝后从正确位置继续。

正确实现的验证算法可以保持目标模型的输出分布。draft model 只负责提出候选，最终接受哪些 Token 仍由目标模型决定。加速来自目标模型一次并行验证多个候选，而不是降低目标模型精度。

**MTP 是训练目标和模型结构；speculative decoding 是推理算法。** 两者可以组合使用，但 speculative decoding 也可以使用独立 draft model、EAGLE 等方法生成候选。

DeepSeek-V3 的公开权重包括单独的 MTP 模块；vLLM 也提供 [MTP speculative decoding](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/) 支持。实际加速取决于候选接受率、每次验证的候选数、batch size、模型结构和 kernel 开销；支持 MTP 本身不能保证固定的加速倍数。

MTP 不一定等于“再运行一个完整的小模型”。独立 draft model 会增加一份 draft 权重和 KV cache；内置 MTP 通常只增加若干预测层或 head。两种方法都在增加候选生成和验证计算，但目标模型一次读取权重后可以并行验证多个位置。如果一次验证接受了 `k` 个 Token，目标模型权重的读取成本就近似由 `k` 个输出 Token 分摊；代价是 draft 计算、被拒绝候选的计算和额外 KV 管理。

这解释了为什么 speculative decoding 常用于单请求、memory-bound 的 Decode。此时矩阵单元原本没有达到峰值吞吐，可以用剩余计算能力减少每个已接受 Token 对应的权重读取次数。计算能力与显存带宽之比较高的 GPU，例如 RTX 4090，可能比计算能力较低但带宽相近的 GPU 获得更大收益。高 batch serving 已经可以通过多个请求复用权重，工作负载更接近 compute-bound；此时 draft 开销可能抵消收益。

公开实测可以说明数量级，但不能代替本机 A/B 测试：[Qwen3.8-27B MTP 社区记录](https://github.com/sudoingX/qwen38-mtp)在相同模型上报告了 RTX 4090 从 47.7 提高到 76.3 token/s，以及另一种 Q4 配置从 36.1 提高到 74.8 token/s；两组都使用 `n-max=2`，记录的接受率范围不同。该项目同时报告 speculative decoding 的单流收益在 `--parallel 4` 时消失。[Sequoia](https://arxiv.org/abs/2402.12374)则从更一般的 speculative decoding 角度说明，收益会随硬件的 compute/bandwidth ratio、draft tree、接受率和 offload 方式变化。

因此测试 MTP 时至少固定模型与量化、prompt、输出长度、temperature、并发数和运行时版本，然后记录 baseline 与 MTP 的 token/s、接受率、TTFT、显存占用和功耗。代码和 JSON 等可预测结构可能有较高接受率；开放式文本可能更低。增加 draft token 数也不保证更快，因为被拒绝候选仍消耗计算。

---

## 第二部分：决定速度的到底是什么？

### 容量、带宽和算力

买硬件时需要分别计算以下三个指标：

**容量（GB）**：决定能否同时存放模型权重、KV cache、activation 和运行时 workspace。容量不足时必须减小模型或上下文、降低精度、减少并发，或者把数据分布到其他设备；峰值 TFLOPS 不能解决容量不足。

**内存带宽（GB/s 或 TB/s）**：每秒可以从内存向计算单元传输多少权重和 KV 数据。单用户、batch size 很小的 Decode 通常首先受它限制。

**计算能力（FLOPS/TOPS）**：每秒可以完成多少运算。Prefill、训练和高 batch 的矩阵乘法通常首先受计算吞吐限制。

**互联带宽**决定显卡或机器之间交换 activation、梯度和 MoE Token 的速度。模型跨显卡、跨机器或跨 MoE 专家时，PCIe、NVLink、Infinity Fabric 或网络都可能成为瓶颈。

在 batch size 为 1、Decode 受权重读取带宽限制时，可以用以下公式估算理论上限：

```text
理论 Decode 速度上限（token/s）≈ 可用内存带宽 ÷ 每 Token 需要读取的权重字节数
```

例如一个 8B 密集模型，量化后的有效大小是 5GB，设备真实可用带宽假设是 250GB/s：

```text
250 ÷ 5 ≈ 50 token/s
```

50 token/s 是只计算权重读取的上限。实际速度还要扣除 attention、KV cache 读取、量化解码、kernel 启动和调度开销。设备标称 250GB/s 也不代表推理程序能持续利用全部带宽，所以实测结果应低于这个数字。

#### 完整数值例子：H200 运行 FP8 27B 密集模型

NVIDIA 公布的 H200 SXM 规格是 141GB HBM3e、4.8TB/s 显存带宽和 3,958 TFLOPS FP8 Tensor Core 吞吐；官方表格注明该 FP8 数字使用结构化稀疏。下面按不使用稀疏的约 1,979 TFLOPS 计算。假设 27B 模型所有权重都以 1 byte/parameter 的 FP8 保存，忽略 scale 和未量化层，则权重约为 27GB。

先算 batch size 为 1 的 Decode。每生成一个 Token，需要读取约 27GB 权重，并完成约 `2 × 27B = 54 GFLOPs` 的线性层运算：

```text
带宽上限 = 4,800 GB/s ÷ 27 GB ≈ 178 token/s
计算上限 = 1,979,000 GFLOPs/s ÷ 54 GFLOPs ≈ 36,648 token/s
实际理论上限 = min(178, 36,648) = 178 token/s
```

两个上限相差约 206 倍，所以这个理想化的单流 Decode 明确是 memory-bound。141GB 容量使 27GB 权重可以常驻显存，并留下约 114GB 给 KV cache、workspace 和并发请求；容量本身不会把 178 token/s 的带宽上限提高。

再看权重被多个 Token 复用的情况。FP8 权重每个参数 1 byte，每个 Token 对每个参数约做 2 FLOPs。如果一次矩阵乘法同时处理 `M` 个 Token，线性层的算术强度近似为 `2M FLOPs/byte`。H200 在上述假设下的机器计算/带宽比是：

```text
1,979 TFLOPS ÷ 4.8 TB/s ≈ 412 FLOPs/byte
临界 M ≈ 412 ÷ 2 ≈ 206 Token
```

当 `M=64` 时，权重带宽对应的吞吐上限约为 `4,800 × 64 ÷ 27 = 11,378 token/s`，仍低于 36,648 token/s 的计算上限，线性层仍偏 memory-bound。当 `M=256` 时，带宽上限提高到约 45,511 token/s，超过计算上限，线性层转为 compute-bound。

Prefill 的线性层会同时处理整段输入。单个 2,048 Token prompt 的投影层和 MLP，其 `M` 已远高于上述约 206 Token 的临界值，所以在这个简化 roofline 模型中会受 FP8 计算吞吐限制。实际 Prefill 还包含 attention；full attention 的计算量随序列长度近似按平方增长，因此实际每秒处理的输入 Token 会低于 36,648，并随模型结构和输入长度变化。

这个例子使用的是硬件峰值和简化的 `2 × 参数量` 计算量。真实结果还要扣除 scale、embedding、未量化层、attention、KV cache、activation 读写、kernel 效率和调度开销。判断方法不变：分别计算带宽上限与计算上限，较小的那个才是当前负载的上限。

批量增大后，一次读入的权重可以服务更多 Token，算术强度上升，工作负载可能从 bandwidth-bound 逐渐变成 compute-bound。因此，单用户测试结果不能直接预测 100 个并发用户下的吞吐。

硬件规格见 NVIDIA [H200 官方页面](https://www.nvidia.com/en-us/data-center/h200/)。计算过程可参考 NVIDIA 对 [prefill 与 decode](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) 以及 [roofline 模型](https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/) 的说明。

### Latency 与 Throughput

测试前需要确定主要指标及其约束。

**Latency（延迟）**描述单个请求的等待时间：

- TTFT（Time To First Token）：点发送后多久看到第一个字；
- ITL/TPOT（Inter-Token Latency / Time Per Output Token）：之后每个 Token 间隔多久；
- End-to-end latency：整个回答多久完成。

**Throughput（吞吐）**关心整台服务器单位时间做多少工作：总 output tokens/s、requests/s，以及在某个延迟 SLA 下每张 GPU 每秒能处理多少 Token。

如果只给自己聊天，你一般希望 batch 小、TTFT 低、每个对话 token/s 高。

如果几十到几百人共用一个模型，主要指标应包括总吞吐、最大并发、P95/P99 延迟和每百万 Token 的成本。在延迟 SLA 允许的范围内增加 continuous batch，可以让一次权重读取服务更多 Token，从而提高总吞吐。

这些指标存在直接冲突。增大 batch 可以让一次权重读取服务更多请求，提高总 tokens/s；但请求可能需要等待调度器组成 batch，TTFT 随之上升。例如调度器额外等待 20ms 可能提高满载吞吐，却会给每个请求增加最多约 20ms 排队延迟。具体收益取决于到达率、输入长度分布和 GPU 利用率，必须在目标 P95/P99 延迟约束下测试。

### 模型到底要占多少内存？

纯权重容量的下限为：

```text
模型权重 GB ≈ 参数量（B）× 每参数 bit 数 ÷ 8
```

所以 8B 模型：

| 理论精度 | 纯权重理论大小 |
|---|---:|
| FP32 | 32GB |
| FP16/BF16 | 16GB |
| FP8/INT8 | 8GB |
| 4-bit | 4GB |

实际文件还包括 scale、zero point、embedding、metadata 和未量化层。GGUF 的 `Q4_K_M` 也不等于严格的 4 bit/parameter；初步预算可按 4.5～5 bit/parameter 计算，下载后应以文件大小和运行时实际占用为准。

完整内存预算应该是：

```text
模型权重 + KV cache + 临时 activation / workspace + 推理框架开销 + 操作系统和其他程序
```

不能把显卡标称容量全部分配给权重。CUDA context、模型 activation、临时 workspace 和 KV cache 都需要显存。没有模型配置时，可以先给这些项目预留 10%～20%；确定层数、KV heads、head dimension、context 和并发后，应改用前面的 KV cache 公式计算，而不是继续使用固定百分比。

训练除了权重，还可能需要梯度、optimizer states、master weights 和 activation。以常见的混合精度 Adam 训练作近似估算，仅参数、梯度和优化器状态就可能达到每参数 16 bytes 左右，还没有包括 activation；BF16 推理权重只有每参数 2 bytes。LoRA/QLoRA、gradient checkpointing、ZeRO/FSDP 能减少占用，因此推理所需内存不能用于推断训练所需内存。

### MoE：总参数很大，激活参数很小

Mixture of Experts（MoE）模型会把部分 FFN 层拆成很多“专家”，再用 router 为每个 Token 选择少数几个专家。

![MoE 工作方式示意图](image-5.png)

因此 MoE 有两个参数数字：

- **总参数（total parameters）**：所有专家加起来，决定权重至少要占多少内存；
- **激活参数（active parameters）**：一个 Token 实际经过的参数，和单 Token 计算量、理想情况下的权重读取量更相关。

比如 Qwen3-Next-80B-A3B 是约 80B 总参数、每 Token 激活 3B；2026 年 8 月发布的 [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) 是 125B 主模型参数、每 Token 激活 6B，另外还有 51B N-gram embedding。内存需要容纳 80B 或 125B 主模型的权重以及随 checkpoint 提供的额外模块，而每 Token 的 MoE 计算只经过其中 3B 或 6B 激活参数。

因此，128GB、256GB 或更大容量的 LPDDR5X/统一内存系统可以容纳总参数较大的量化 MoE。其内存带宽低于 HBM GPU，但每个 Token 只访问少数专家时，单用户 Decode 的权重读取量可能显著低于同总参数的密集模型。

例如，80B 总参数模型按平均 4.5 bit/parameter 计算，纯权重约为 45GB，因此可以装入 64GB 以上的可用内存。若每 Token 激活 3B 参数，对应的激活权重下限约为 `3B × 4.5 / 8 = 1.69GB`。在 256GB/s 内存系统上，只按这部分权重读取计算的上限约为 151 token/s。实际还要读取 attention、shared expert、embedding、KV cache 和量化 scale，并承担路由与 kernel 开销，所以真实速度会明显低于 151 token/s。这个估算只能用于解释为什么该类硬件可能达到可交互速度，不能代替 benchmark。

激活参数少不等于 MoE 不受带宽限制，原因包括：

- attention、shared expert 和其他 dense 层仍参与计算；
- router 的不规则访问会降低实际效率；
- 不同 Token 可能选择不同专家，高 batch 会增加同一解码步骤访问的专家数量；
- 专家分布在多张卡上时会产生 all-to-all 通信；
- 如果专家权重没有常驻 GPU 可直接访问的内存，而是每层都要从系统 RAM 通过 PCIe 传输，传输上限会由 PCIe 决定。例如 PCIe 4.0 x16 的单向理论带宽约为 32GB/s，远低于高端 GPU 的数百 GB/s 到数 TB/s 显存带宽。

MoE 的实际吞吐取决于 fused MoE kernel、expert parallel、负载均衡和互联带宽。评测时需要同时固定总参数、激活参数、batch size、并行方式和运行时版本。

---

## 第三部分：精度与量化

### 浮点数和整数到底差在哪？

降低位宽不会使模型质量按固定比例下降。结果取决于数值格式、block 大小、scale、校准数据、量化算法、模型结构和任务；位宽同时影响权重文件大小、内存流量和可用的硬件指令。

| 格式 | 典型用途 | 特点 |
|---|---|---|
| FP32 | 科学计算、部分累加和 master weight | 每元素 4 bytes；8-bit exponent、23-bit mantissa |
| TF32 | NVIDIA Tensor Core 上的矩阵乘法 | 使用 FP32 的 8-bit exponent 和 10-bit mantissa；输入与输出仍以 FP32 表示 |
| FP16 | 训练、推理 | 每元素 2 bytes；5-bit exponent、10-bit mantissa，动态范围小于 BF16 |
| BF16 | 训练、推理 | 每元素 2 bytes；8-bit exponent、7-bit mantissa，指数范围与 FP32 相同 |
| FP8 | 训练、推理 | 每元素 1 byte；E4M3 提供较多尾数位，E5M2 提供较大动态范围 |
| INT4 / GGUF Q4 | 消费级本地推理 | 原始权重每元素 0.5 byte，另需 scale 等 metadata；多数执行 W4A16 |
| MXFP4 | OCP 标准微缩放 FP4 | 32 个 E2M1 值共享一个 E8M0 scale |
| NVFP4 | NVIDIA 的 FP4 recipe | 16 个 E2M1 值共享 E4M3 scale，另有全 tensor FP32 scale |

**权重存储精度和矩阵乘法的执行精度是两个独立参数。** “权重以 4 bit 保存”不等于“GPU 用原生 FP4 做矩阵乘法”。

没有原生 FP4 矩阵指令的 GPU 通常把 4-bit 权重读入寄存器或 shared memory，反量化成 FP16/BF16，再执行 W4A16 运算。这样可以减少权重容量和内存流量，但矩阵乘法仍使用 FP16/BF16 指令，并增加反量化开销。

### FP4 为什么必须缩放？

FP4 E2M1 只有 1 个 sign bit、2 个 exponent bit 和 1 个 mantissa bit。它能表达的非负数大致只有：

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

如果一组权重的绝对值范围是 0.001～200，直接转换为未缩放的 E2M1 会让小值舍入为 0，并让超过 6 的值饱和或溢出。

低精度量化需要先计算 scale，把一组数映射到 FP4 可以表示的范围：

```text
量化：q = round_to_fp4(x / scale)
还原：x_hat = q × scale
```

整个 tensor 共用一个 scale 所需的 scale metadata 最少，但一个较大的离群值会增大 scale，使其他数值量化后集中在 0 和相邻的少数表示值。Block scaling 将 tensor 划分为小块，并为每个 block 单独计算 scale。

例如一个 block 的最大绝对值为 6 时，可以使用 scale=1；如果最大绝对值为 0.06，可以使用 scale=0.01，把 0.06 映射为 FP4 中的 6。若这两组数共用 scale=1，0.06 会被舍入为 0，信息直接丢失。较小 block 可以让不同范围的数据使用不同 scale，通常降低误差；代价是需要存储更多 scale，并增加量化、内存布局和 kernel 处理开销。

### MXFP4 与 NVFP4

两者的元素主体都是 E2M1 FP4，区别主要在缩放方法。

OCP [Microscaling Formats (MX) 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)规定 MXFP4：

- 每 32 个 E2M1 元素共享一个 scale；
- scale 使用 E8M0；
- E8M0 没有 mantissa，scale 取 2 的整数次幂；每个 32 元素 block 使用 1 byte scale。

```text
x ≈ fp4_value × block_scale_E8M0
```

NVIDIA 的 [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) 使用两级缩放：

- 每 16 个 E2M1 元素共享一个 FP8 E4M3 block scale；
- 整个 tensor 再共享一个 FP32 global scale。

```text
x ≈ fp4_value × block_scale_E4M3 × global_scale_FP32
```

E4M3 scale 有 3 个 mantissa bits，不限于 2 的整数次幂；NVFP4 的 block 也从 32 个元素减为 16 个。两项设计增加了可选 scale 的密度，通常降低量化误差，同时增加 scale metadata 和 recipe/kernel 的实现要求。

NVIDIA 的 NVFP4 训练 recipe 还使用二维 16×16 weight scaling、Random Hadamard Transform 和 stochastic rounding。二维 scaling 使按行和按列使用的量化权重保持一致；Hadamard Transform 用于重新分布离群值；stochastic rounding 减少反复量化时的系统性舍入偏差。因此，比较 FP4 模型时需要同时确认元素格式、scale 类型、block 大小、量化 recipe 和执行 kernel，不能只比较“4 bit”这个数字。

### 不同 GPU 架构支持什么精度？

下面是面向 LLM 用户的简化表。具体型号和软件栈仍然可能有差异：

| 硬件架构 | 原生低精度矩阵能力 | 部署含义 |
|---|---|---|
| NVIDIA Turing（RTX 20/T4） | FP16、INT8、INT4 Tensor Core | 可使用 INT4/INT8 weight-only 或 FP16；没有 BF16/FP8 原生矩阵加速 |
| NVIDIA Ampere（RTX 30/A100） | FP16、BF16、TF32、INT8、INT4 | 可原生执行 BF16/FP16 Tensor Core 矩阵乘法；没有原生 FP8/FP4 |
| NVIDIA Ada（RTX 40/L4/L40） | 增加 FP8 Tensor Core | FP8 可有计算优势；FP4 通常仍是 weight-only 路线 |
| NVIDIA Hopper（H100/H200） | FP8、BF16、FP16、TF32、INT8 | FP8 有原生矩阵加速；**没有原生 FP4 Tensor Core** |
| NVIDIA Blackwell（B100/B200、RTX 50/RTX PRO Blackwell、GB10） | FP4/FP6/FP8、BF16/FP16 等 | NVFP4/MXFP4 才能获得原生 4-bit 矩阵加速 |
| AMD CDNA2（MI200） | FP16、BF16、INT8 等 | 训练/推理以 16-bit 和整数为主 |
| AMD CDNA3（MI300/MI325） | 增加 FP8/BF8 | 可使用原生 FP8；没有 CDNA4 的原生 MXFP4 |
| AMD CDNA4（MI350/MI355） | MXFP4、MXFP6、MXFP8、OCP FP8 | 原生支持标准 microscaling 低精度 |
| Apple Silicon / AMD RDNA 核显 | 依赖 Metal/MLX、Vulkan/HIP 等优化 kernel | 4-bit 多数表示权重存储与 weight-only 推理，不要套用 NVIDIA 峰值表 |

资料可参考 NVIDIA 的 [Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/)、[Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) 与 [Blackwell NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) 说明，以及 AMD 的 [CDNA 架构页面](https://www.amd.com/en/technologies/cdna.html)。

#### H100 运行 NVFP4 权重

H100 可以通过软件 kernel 读取 NVFP4 权重，但它没有原生 FP4 Tensor Core。常见兼容路径是在 kernel 内把 FP4 权重反量化到 BF16/FP16，做 W4A16；某些路径也可能转换到其他受支持精度。因此它可能节省显存和带宽，但不能获得 Blackwell 的原生 FP4 矩阵吞吐，并可能慢于使用 H100 原生 FP8 指令的 checkpoint。NVIDIA NIM 也把 pre-Blackwell 的 NVFP4 路径称为 emulation，并提示它可能慢于同卡原生精度。

在 H100 上应比较同一模型的官方 FP8、BF16 和 NVFP4 实测结果；在 Blackwell 上再把原生 NVFP4/MXFP4 作为主要候选。

### 训练和推理应该选什么精度？

| 场景 | 优先测试的精度 | 原因与限制 |
|---|---|---|
| 从零预训练 | BF16 + FP32 累加/optimizer state | BF16 与 FP32 具有相同的 8-bit exponent；矩阵输入减半存储，累加与优化器保留更高精度 |
| Hopper/Ada/CDNA3 大规模训练 | 经过验证的 FP8 mixed precision | 省内存和计算，但要用动态/块缩放 recipe |
| Blackwell/CDNA4 大规模训练 | NVFP4 或 MXFP4 mixed precision（需严格验证） | 可能进一步提高吞吐，但不是所有层都适合 4-bit |
| LoRA/QLoRA 微调 | 4-bit frozen base + BF16/FP16 compute + 16-bit adapter | 大幅减小显存，训练参数仍保留较高精度 |
| 消费级单用户推理 | GGUF Q4_K_M/Q5_K_M 或 AWQ/GPTQ INT4 | 原始 4-bit 权重约为 BF16 的 1/4；质量损失需要按任务实测 |
| Hopper 生产推理 | FP8 或具有对应 kernel 的 INT4 weight-only | 在固定模型、batch 和延迟约束下比较吞吐 |
| Blackwell 生产推理 | 经过任务评测的 NVFP4/MXFP4/FP8 | 可使用原生 FP4/FP6/FP8 矩阵指令；按任务准确率决定最低可用精度 |
| KV cache | 先测 BF16/FP16，再测 FP8/INT8/FP4 | 从 16-bit 降到 8-bit 可把 KV 容量减半；需要检查长上下文准确率和 attention kernel |

消费级本地推理可以先测试 Q4 和 Q5。Q2/Q3 进一步减少内存，但通常会增加数学、代码、工具调用和长上下文任务的准确率损失；小模型对量化误差通常更敏感。“较大模型的 Q2 一定优于较小模型的 Q5”没有普遍保证，需要在目标任务上分别测量成功率。

训练中的不同 tensor 通常使用不同精度。现代 mixed-precision recipe 会给敏感层、累加、梯度和 optimizer state 保留更高精度。理论 FP4 FLOPS 不包括量化、缩放、数据移动和非矩阵算子开销；矩阵尺寸或 batch 较小时也可能无法达到峰值吞吐。

### GGUF、GPTQ、AWQ、FP8 又是什么？

- **Safetensors**：安全、常见的权重文件容器，本身不说明量化方法；
- **GGUF**：llama.cpp 生态的模型与 metadata 容器，常见 Q4_K_M、Q5_K_M、IQ4 等量化；
- **GPTQ**：经典的 post-training weight-only 量化方法/生态；
- **AWQ**：利用 activation-aware 信息保护重要权重的 weight-only 量化；
- **bitsandbytes NF4**：QLoRA 中常见的 4-bit 格式；
- **FP8/NVFP4 checkpoint**：通常需要相应框架、scale 布局和硬件 kernel；
- **EXL2**：ExLlama 生态的可变 bit 量化格式。

下载 4-bit 模型前，需要确认运行时能否加载、目标 GPU 是否有对应 kernel、执行路径是 W4A16 还是 W4A4、KV cache 使用什么精度，以及量化报告是否公开了校准数据、评测任务和相对未量化模型的结果。

---

## 第四部分：如何为 LLM 配一台电脑

### 先问自己四个问题

1. 我要跑多大的模型，密集还是 MoE？
2. 我要多长上下文，同时几个人用？
3. 我更在意单对话速度，还是总吞吐？
4. 我要推理、LoRA 微调，还是从零训练？

根据模型权重、KV cache 和并发计算内存需求后，再选择硬件。只按显卡型号或峰值 FLOPS 购买，无法确认目标模型能否完整装入内存。

### 独立显卡路线

NVIDIA GPU 的主要软件优势是 CUDA 生态。vLLM、TensorRT-LLM、FlashAttention 和多种量化 kernel 通常首先提供 CUDA 实现，新模型架构发布后也经常先获得 CUDA 支持。

挑卡时大致按这个顺序：

1. 显存容量能不能装下目标权重和 KV cache；
2. 显存带宽；
3. 该架构是否原生支持你要用的精度；
4. 软件栈是否支持；
5. 功耗、散热、价格和保修。

对于需要约 20GB 权重、还要分配 KV cache 的模型，24GB RTX 3090 可能比新一代 16GB 显卡更合适，因为 16GB 显卡必须把部分权重放在系统 RAM。若两张卡都能完整容纳权重和 KV cache，则需要比较显存带宽、支持精度、矩阵计算吞吐和运行时 kernel；新架构在 Prefill 和高 batch 场景通常具有更高计算吞吐。

显存不足时可以把部分层放在 CPU 和系统 RAM 中。这样可以运行超过显存容量的模型，但每个 Token 都要执行 CPU 层，或通过 PCIe 传输中间数据，因此速度取决于 CPU 内存带宽、PCIe 带宽和切分位置。系统 RAM 容量不能视为同等性能的显存容量。

非原厂扩容至 48GB 的游戏卡通常涉及显存焊接、BIOS 修改和散热改造，并可能失去原厂保修。购买前需要验证驱动识别、ECC/显存错误、满载温度、长期稳定性和售后责任；这些风险与原厂 48GB 专业卡不同。

AMD 独显和 Instinct 可通过 ROCm、vLLM、SGLang 和 llama.cpp 运行部分模型。购买前需要按具体 GPU、操作系统、模型架构和量化 kernel 核对支持矩阵；“支持 ROCm”不能说明某个模型、量化格式或 attention kernel 已受支持。

### 大容量统一内存路线

统一内存系统让 CPU 和 GPU 访问同一组物理内存，模型不需要在独立的系统 RAM 与显存之间逐层复制。它可以用较低成本提供 128GB 以上的 GPU 可访问内存。限制是 LPDDR5X 带宽通常低于数据中心 HBM，软件也必须为相应 GPU 后端和模型结构提供有效 kernel。因此，这类系统主要用于大容量 MoE 的单用户或低并发推理，而不是直接替代高并发 HBM GPU 服务器。

截至 2026-08-30，相关平台的公开规格如下：

| 平台 | 内存与带宽 | 可用软件与场景 | 限制 |
|---|---|---|---|
| Apple M5 Ultra Mac Studio | 最高 512GB unified memory，1.2TB/s | MLX、Metal、llama.cpp；可容纳超过 100B 参数的量化模型 | 2026-08-25 发布，9 月开始供货；CUDA-only 工具不能直接使用 |
| Apple M5 Max | 最高 128GB，最高 614GB/s | 便携/桌面单用户，中大型量化模型 | 软件主要走 MLX/Metal/llama.cpp |
| NVIDIA DGX Spark（GB10） | 128GB coherent LPDDR5X，273GB/s | CUDA/Blackwell 开发、FP4、约 200B 以内模型实验 | 带宽远低于 HBM 卡；“1 PFLOP FP4”包含特定精度/稀疏条件 |
| AMD Strix Halo / Ryzen AI Max+ 395 | 最高 128GB LPDDR5X，256GB/s | Windows/Linux，llama.cpp Vulkan/HIP 与部分 ROCm 路线 | RDNA 3.5 不是 CDNA4，不具有 MI350 的原生 MXFP4 矩阵能力 |
| Intel Crescent Island | 最高 480GB LPDDR5X，350W PCIe 卡 | 未来的大容量数据中心推理 | 仍是面向客户采样/生态建设的产品，不等于现在能买到的家用卡 |

官方规格来源：[Apple M5 Ultra](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/)、[DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)、[Ryzen AI Halo](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)、[Intel Crescent Island](https://newsroom.intel.com/client-computing/intel-outlines-architectures-for-agentic-ai-at-hot-chips-2026)。

Intel 官方公布的产品名是 **Crescent Island**；没有查到名为 Crescent Lake 的对应官方产品。

以 Qwen3-Next-80B-A3B 为例，按 4.5 bit/parameter 估算的纯权重约为 45GB；128GB 统一内存扣除 45GB 权重后，还剩约 83GB 供操作系统、运行时、未量化层和 KV cache 使用。每 Token 激活 3B 参数，对应的激活权重读取下限约为 1.69GB。购买前还需要用目标运行时的实测 token/s 和内存占用验证，因为专家 kernel、共享层和量化格式会使实际结果偏离这两个下限。

同时服务约 100 名用户时，应按请求到达率、平均输入/输出长度和延迟 SLA 计算所需总 tokens/s。高并发会增加 Prefill 计算量、KV cache 占用和同时被访问的专家数量。此时多张数据中心 GPU 的 HBM 带宽、矩阵吞吐、expert parallel 与 serving 软件通常比单个统一内存 SoC 提供更高吞吐。两类系统面向的负载不同。

### 多显卡与多机

多卡能解决容量问题，但不会自动线性加速。

- **Tensor Parallel（TP）**：一层矩阵分到多卡，每层都要通信；
- **Pipeline Parallel（PP）**：不同层放在不同卡；流水线未填满或排空时会产生空闲阶段；
- **Expert Parallel（EP）**：MoE 的专家分到不同卡，需要 all-to-all；
- **Data Parallel（DP）**：每组卡复制一份模型，服务不同请求，适合提升吞吐。

两张卡可以分担模型容量，但每层跨卡同步会增加 latency。PCIe 4.0 x16 单向理论带宽约为 32GB/s；H100 SXM 的 NVLink 聚合双向带宽约为 900GB/s。并行策略需要频繁交换 activation 或专家 Token 时，互联差异会直接影响扩展效率。

10GbE 的单向理论带宽约为 1.25GB/s，远低于显存内部的数百 GB/s 至数 TB/s。它可以用于请求级分流、通信量较低的 pipeline 或特定 offload；需要每层同步或 all-to-all 的并行方式会受到网络带宽和延迟限制，四台机器的性能不能按设备数量直接相加。

Thunderbolt 5、RDMA 和高速 NIC 可以提高互联带宽并降低部分通信开销；Apple 新款 Mac Studio 也支持通过 Thunderbolt 5/RDMA 组集群。最终吞吐仍取决于运行时支持、模型切分方式、每 Token 通信量和网络延迟，不能只相加各机器的内存容量和带宽。

---

## 第五部分：如何选择模型

### 先确定能运行的候选范围

模型排名没有意义，除非模型能在目标硬件、运行时和延迟要求下工作。筛选顺序可以固定为：

1. **容量**：实际权重文件、KV cache、workspace 和并发序列能否同时放入可用内存；
2. **运行时**：框架是否支持该模型架构、量化格式、attention、MTP 和 tool calling；
3. **速度**：用权重大小和内存带宽估算单流 Decode 上限，再测 TTFT、ITL 和目标并发下的 throughput；
4. **任务能力**：在自己的数据上比较成功率，而不是先按参数量或公开总分决定；
5. **许可证与成本**：确认商用、再分发和衍生模型条款，并计算每个成功任务的 GPU 时间、电力或 API 成本。

例如一张 24GB 显卡若给 KV cache、workspace 和桌面程序留下 6GB，可用于权重的预算约为 18GB。27B 模型按平均 4.5 bit/parameter 计算，纯权重约为 15.2GB，加入 scale 和未量化层后可能仍能完整放入显存；70B 模型的纯权重已经约为 39.4GB，不在候选范围内。如果该卡实测显存带宽约为 1TB/s、模型实际权重占 17GB，则单流 Decode 的权重读取上限约为 `1,000 ÷ 17 = 59 token/s`，实测应低于该数字。

在比较模型前，还需要写出服务目标。例如个人聊天可以要求 P95 TTFT 小于 2 秒、单流输出不低于 30 token/s；内部服务可以要求 32 并发下 P95 TTFT 小于 5 秒、总吞吐不低于 500 output token/s。没有这些约束，“速度快”无法对应具体选择。

### 排行榜的适用范围

选择模型至少要同时看：任务、dense/MoE、thinking/instruct、总参数与激活参数、context、tokenizer、运行时支持、量化、许可证，以及你的硬件上的 TTFT、token/s、并发和功耗。

排行榜用于产生候选名单，不用于直接决定生产模型。固定、公开的 benchmark 会面临两类问题：测试题可能进入预训练或微调数据；即使没有直接泄漏，训练方也可以针对公开任务的题型、prompt、评分器和总分反复选择数据、checkpoint 与推理参数。多次围绕同一测试集做选择，本身就会逐步过拟合该测试集。

这不表示公开 benchmark 没有用。数学、代码、长上下文、工具调用和人类偏好测量的是不同能力，多个独立结果可以用于缩小范围。需要检查 benchmark 的题目来源、更新时间、评分方法、是否允许 reasoning、每题采样次数和置信区间。[LiveBench](https://arxiv.org/abs/2406.19314)通过频繁更新题目和使用可自动核对的答案降低污染风险，但仍不能保证其任务分布等于你的实际工作。

### LMArena

[LMArena](https://lmarena.ai/) 的核心方法不是固定选择题，而是匿名展示两个模型对同一真实用户 prompt 的回答，让用户选择 A、B、平局或都不好，再用成对偏好数据计算排名。[Chatbot Arena 论文](https://arxiv.org/abs/2403.04132)介绍了其 pairwise comparison 方法。它的优点是问题来自真实用户，评分直接反映人类看到完整回答后的偏好，并且没有一套长期固定的标准答案。

它测量的仍然是“LMArena 用户在该 prompt 分布下更喜欢哪个回答”，不是所有任务的正确率。结果会受到回答长度、格式、措辞、拒答方式、用户语言、题目类别和投票人群影响；它也不测量你的硬件能否运行模型、量化后的质量、许可证、TTFT、并发吞吐或任务成本。模型训练方还可以针对 Arena 中常见的回答风格优化。因此 LMArena 适合发现候选模型和比较通用对话偏好，不能替代代码单元测试、企业文档抽取准确率或特定写作风格的内部评测。

### Artificial Analysis

[Artificial Analysis](https://artificialanalysis.ai/models/) 同时提供 Intelligence Index、首 Token 延迟、输出速度、缓存价格、reasoning token 使用量和每任务成本，可用于初步筛选 API 模型。

它的 [methodology](https://artificialanalysis.ai/methodology) 说明了各 benchmark 的构成、权重和成本算法，并区分“每百万 Token 的标价”和“完成一个任务的成本”。

Artificial Analysis Index 是多个 benchmark 的加权组合，其任务分布和权重不一定等于你的业务分布。它可以用于缩小候选范围，最终选择仍应由业务 eval 决定。

模型技术报告用于确认架构、context、训练方法和许可证。Artificial Analysis、LMArena、固定 benchmark 和技术报告测量的内容不同，不能相互替代。

### 建立自己的 private benchmark

最终选择应使用没有公开发布、与实际任务一致的 private benchmark。可以先收集 30～200 个真实任务，删除个人和公司敏感信息，为每一题预先规定评分方法，并保留一部分从未用于调 prompt 或选模型的 held-out 集合。

| 任务类型 | 评分方法示例 |
|---|---|
| 代码修改 | 在隔离环境运行单元测试，记录通过数、编译错误和超时 |
| 结构化抽取 | 比较字段值、JSON schema、漏项率和多余字段率 |
| 数学与规则推理 | 使用标准答案或程序 verifier；把最终答案和过程格式分开评分 |
| RAG/企业问答 | 核对引用位置、事实正确率、无依据陈述和拒答正确率 |
| 工具调用 | 检查工具选择、参数 schema、调用次数和最终任务是否完成 |
| 客服、总结、翻译 | 使用固定 rubric，由不知道模型名称的评审者盲评 |
| 小说与开放式写作 | 盲评情节一致性、人物声音、风格遵循、约束满足和重复程度 |

每次评测应记录完整配置：模型和 checkpoint、量化、chat template、system prompt、temperature、reasoning effort、context、运行时版本和硬件。性能至少分别测量 512、4K 和目标长输入下的 TTFT，单流 ITL，以及并发 1、4、16、目标峰值时的总 throughput 和 P95/P99 延迟。质量结果还应配套记录平均输入/输出/reasoning Token、失败与重试次数、峰值内存和功耗。

private benchmark 也会被自己过拟合。调 prompt、选择量化或修改 agent 流程时只使用开发集；最终比较使用 held-out 集。可以每月替换 10%～20% 题目并保留历史版本，既观察新任务，也检查模型升级是否让旧任务退步。样本量较小时报告每类任务的原始成功数，例如“代码 37/50、抽取 44/50”，不要只报一个加权总分。

选择流程可以分成两轮：先淘汰容量、许可证、格式正确率或任务成功率不达标的模型；再在合格模型中比较满足延迟 SLA 时的 throughput、功耗和每个成功任务成本。这样不会让一个便宜但经常失败的模型仅凭 Token 单价胜出。

### Reasoning、RLVR 与真实成本

当前主流模型通常提供 thinking/reasoning 模式，部分模型也提供可关闭推理的 instruct 模式。

很多模型通过 reinforcement learning 学会在回答前进行更长的推理。RLVR（Reinforcement Learning with Verifiable Rewards）使用可以自动检查的结果作为奖励，例如数学最终答案、代码单元测试、逻辑约束和输出格式。DeepSeek-R1 等工作显示了数学和代码任务上的提升；OpenAI 也公开说明部分 reasoning 能力会随训练时 RL compute 和测试时 thinking compute 增长。

RLVR 的直接训练信号在哪些任务上最可靠，reasoning 的收益通常就在哪些任务上最容易测量。数学题可以核对答案，代码可以运行测试，工具调用可以检查 schema 和环境状态；小说质量、语气、审美和开放式建议没有唯一正确答案，不能直接使用同一种 verifier。专门研究写作训练的 [Writing-Zero](https://arxiv.org/abs/2506.00103)正是因为普通 RLVR 与非可验证写作任务之间存在这一缺口，才另外设计 pairwise generative reward model 和训练方法。

因此不能从“模型是 reasoning model”直接推出“打开 reasoning 后小说一定写得更好”。规划长篇结构、检查人物关系或满足明确写作约束可能受益，但文风、创意和人物声音未必提高；更长的推理还可能增加重复、解释性文字、延迟和成本。模型厂商也可能混合使用 SFT、偏好训练、RLVR、模型评审和专门写作数据，所以不能只根据模型标签推断训练内容。

对于小说、翻译、总结、分类和普通聊天，应在同一模型上比较 non-thinking、低 reasoning effort 和高 reasoning effort。写作任务可以固定 50 个 prompt 和输出长度，由不知道配置的评审者做成对盲评；同时记录平均输出 Token、端到端延迟和每个被选中回答的成本。如果 reasoning 版本的胜率没有提高，就不应为额外 trajectory 付费或占用本地 GPU。

只比较每百万 Token 的单价会遗漏每个任务消耗的 Token 数：

> 一个输出价格 $1 / 1M token 的模型，不一定真的只有 $10 / 1M token 模型的十分之一成本。

原因是前者可能为了得到同样正确率，生成十几倍甚至几十倍的 reasoning trajectory。

真实成本大致应该按任务算：

```text
任务成本 = 非缓存输入 Token × 输入价
         + 缓存输入 Token × 缓存价
         +（reasoning Token + 可见答案 Token）× 输出价
```

自部署不产生按 Token 计费的 API 账单，但 reasoning 仍消耗计算资源：更长的 trajectory 会增加端到端延迟、GPU 时间和电力，并在固定硬件上降低可服务的并发数。

因此比较 reasoning 模型时至少记录任务成功率、平均 reasoning + answer tokens、end-to-end latency、每个成功任务的成本，以及不同 reasoning effort 下的能力曲线。

提高 reasoning effort 通常会增加测试时计算量，并可能提高复杂数学、代码和规划任务的成功率，但收益不会随 Token 数线性增长。模型也可能产生重复或无效推理。抽取、翻译和分类任务应分别测试低 reasoning effort 或 non-thinking 模式，因为这些任务的额外推理可能只增加延迟和成本。

### 模型许可证

“权重可以下载”不等于符合 OSI 开源定义，也不等于许可证允许不受限制的商业使用。

检查商业用途、用户规模限制、attribution、蒸馏/再分发、地区与用途限制，以及训练数据、代码和权重是否使用不同许可证。

Apache-2.0、MIT 和各类自定义 community license 的义务不同，应阅读权重仓库中对应版本的许可证原文。正式商业部署还应由法律人员核对商用、再分发、attribution、使用范围和衍生模型条款。

---

## 第六部分：部署方法

### 路线 A：LM Studio

[LM Studio](https://lmstudio.ai/) 提供模型搜索、GGUF 下载、上下文配置、聊天界面和本地 OpenAI-compatible API，使用时不需要命令行。

基本流程：安装后在 Discover 搜索模型，先选 Q4_K_M 或 Q5_K_M；context 先设 4K/8K；确认 GPU offload；运行正常后再逐渐增加 context。

LM Studio 适合图形界面的单机测试和本地 API。自动选择的 context、GPU offload 和量化未必具有最高吞吐；需要 continuous batching、分布式并行或数十个并发请求时，应同时测试 vLLM/SGLang。

### 路线 B：Ollama

[Ollama](https://ollama.com/) 通过模型 manifest 管理下载、运行参数和本地 API。

```bash
ollama run qwen3:8b
ollama list
```

它默认提供本地 HTTP API，可连接 Open WebUI、编辑器插件和其他程序。

Ollama 支持 macOS、Windows 和 Linux，并提供统一的模型管理与 API。部分底层参数和新模型特性可能晚于 llama.cpp/vLLM 暴露；下载前应在模型 manifest 中核对具体量化、context 和模板，而不是只看模型标签。

### 路线 C：llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) 支持 CPU、CUDA、Metal、Vulkan、HIP 等后端，并定义和使用 GGUF 模型格式。

```bash
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M \
  -ngl 99 \
  -c 8192 \
  --port 8080
```

Windows 下可执行文件可能叫 `llama-server.exe`。

- `-ngl 99`：尽量把所有层 offload 到 GPU；
- `-c 8192`：最大 context 先设 8K；
- `--port 8080`：启动本地服务。

如果出现 out-of-memory，先减小 context 或 GPU layers。若速度明显低于同类硬件公开结果，检查启动日志确认 CUDA、Metal 或 Vulkan 后端已经加载，并核对实际 offload 的层数。

llama.cpp 可用于单用户、边缘设备、Apple Silicon、AMD APU 和没有 CUDA 的系统，也可以启动 HTTP 服务。几十到几百个并发用户的部署还应测试 vLLM/SGLang，因为它们提供 continuous batching、分布式并行和面向服务的 KV cache 调度。

### 路线 D：vLLM

[vLLM](https://docs.vllm.ai/) 是面向批量和并发请求的 LLM serving engine。它提供 continuous batching、PagedAttention、prefix caching、tensor/pipeline/expert parallel、量化和 OpenAI-compatible API。

vLLM 的原生部署环境是 Linux。Windows 可以通过 WSL2 使用，但需要分别检查 GPU passthrough、网络端口映射和 Windows/WSL 文件路径。

```bash
python -m venv .venv
source .venv/bin/activate
pip install vllm

vllm serve Qwen/Qwen3-8B \
  --dtype auto \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --port 8000
```

调用 OpenAI-compatible API：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "你好！请介绍一下你自己。"}]
  }'
```

两张 GPU 的 tensor parallel 可使用 `--tensor-parallel-size 2`。测试时先固定短 context 和单 GPU 基线，再分别启用量化、TP、speculative decoding、长 context 和高并发；每次只改变一个变量并记录 TTFT、ITL、吞吐和显存占用。

### SGLang、TensorRT-LLM、NIM 与 OpenVINO

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) 是另一个 serving framework，提供 structured generation、radix cache、MoE、speculative decoding 和分布式部署。vLLM 与 SGLang 的模型和硬件支持范围随版本变化。

团队部署应在相同模型、量化、prompt length、output length、并发和延迟限制下测试两者。不同测试若没有固定这些变量，throughput 数字不能直接比较。

#### TensorRT-LLM

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) 是 NVIDIA GPU 专用的 LLM 推理工具包，支持 in-flight batching、paged KV cache、量化、speculative decoding 和多 GPU/多节点部署。

TensorRT-LLM 仅面向 NVIDIA GPU，适用于模型和硬件固定、可以投入 engine build 与版本匹配工作的部署。vLLM 直接加载 Hugging Face 模型的路径通常更短；TensorRT-LLM 可针对固定配置生成和选择更专门的 kernel。最终性能需要在同一负载下测试。

#### NVIDIA NIM

NIM 没有过时。2026 年的 NIM LLM 提供经过验证的容器、模型 profile、健康检查、监控、安全更新和企业支持。当前 NIM LLM 2.x 使用直接对齐上游版本的 vLLM 作为核心 inference backend；NIM 的主要功能是打包、配置验证和运维支持，而不是定义新的 Transformer 推理算法。

个人或研究环境可以直接使用 vLLM。需要 CVE 修复、兼容性验证、稳定更新周期和厂商支持的企业环境可以评估 NIM。

#### OpenVINO

[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) 当前仍在维护，并支持 continuous batching、prefix caching、speculative decoding、paged attention 和 Intel CPU/GPU/NPU。

Intel 硬件应测试 OpenVINO 的官方执行路径。ONNX Runtime 仍是通用推理运行时，但新 LLM 架构的模型转换、量化和生成式服务功能需要逐项验证，不应仅根据 ONNX 模型能加载就推断其吞吐。

### 已经过时或不再推荐的项目

**IPEX-LLM**：项目已经在 2026-01-28 被 Intel 官方归档。仓库明确写明不再维护、不再修复，并提示存在已知安全问题。新部署不要继续选择它；Intel 路线优先看 OpenVINO GenAI、llama.cpp 的 Intel 后端或当前 Intel 官方推荐栈。

**旧的 NIM 1.x 多 backend 说明**：很多旧文章会说 NIM 自动在 TensorRT-LLM、vLLM 等 backend 间选择。NIM 2.x 已改成 one-container/one-backend，当前文档要以对应版本为准。

**把 Hugging Face Transformers 直接当生产服务器**：Transformers 可作为开发和研究基线，但单独调用 `model.generate()` 不提供生产服务所需的请求调度、continuous batching、跨请求 KV 管理、监控和限流。

---

## 第七部分：vLLM、PagedAttention 与高并发

传统做法常为每个请求预留一块连续的最大 KV cache。

请求的输入与输出长度不固定。如果按照最大长度为每个请求预留连续 KV cache，短请求会留下大量未使用空间；如果运行中不断申请不同大小的连续空间，则会产生内存碎片。两种情况都会减少同一时刻可以保留的序列数量。

PagedAttention 使用与操作系统虚拟内存相似的 block 映射：

- 把一个请求的 KV cache 切成固定 Token 数的小 block；
- 逻辑上连续的上下文，可以映射到物理上不连续的显存 block；
- 新 Token 生成时按需分配 block；
- 共享前缀的请求可以指向同一批物理 block；
- 分叉写入时使用类似 copy-on-write 的机制。

```text
请求 A 的逻辑 KV： [0] [1] [2] [3]
                       │   │   │   │
物理显存 block：     [17] [04] [91] [22]

请求 B 的相同前缀： [0] [1] [2] [8]
                       │   │   │   │
                      [17] [04] [91] [36]
```

PagedAttention 不改变模型的 attention 定义。它改变 KV cache 的物理存储和地址映射，使内存按 block 分配，并把浪费主要限制在每个序列最后一个未填满的 block。可用显存因此可以容纳更多活跃序列，提高 continuous batch 的大小和总 throughput。

**Continuous batching** 会在每个调度步骤移除已经完成的序列，并加入新的等待请求。静态 batching 必须等待同一 batch 中最长的序列结束，其他已完成序列占用的 batch slot 在此期间无法处理新请求；continuous batching 可以避免这部分空闲。

vLLM 原始 [PagedAttention 论文](https://arxiv.org/abs/2309.06180)在论文所用模型、硬件和基线下报告了同等延迟时 2～4 倍的 serving throughput 提升；早期博客在其他旧基线下报告过更大的倍数。这些数字不能直接外推到不同模型、运行时版本和 GPU，当前部署需要使用相同负载重新测试。

部署时常见的优化顺序：

1. 模型和 KV cache 全部放进目标 GPU；
2. 用真实输入/输出长度测 baseline；
3. 按业务需要调整 `max-model-len`，不要直接使用模型声明的最大值，因为 KV cache 预留和可支持并发会受其影响；
4. 开 prefix caching；
5. 调整并发与 batch token 上限；
6. 再试 FP8/INT8 KV、量化、chunked prefill；
7. 最后尝试 TP/PP/EP、speculative decoding 与 prefill/decode disaggregation。

优化 throughput 时应同时记录 TTFT、ITL、P50/P95/P99、GPU memory、cache hit rate 和错误率。单独一个 tokens/s 数字不能说明是单请求生成速度、所有请求的总速度，还是以更高延迟换取的吞吐。

---

## 第八部分：怎样提高 KV cache 命中率

Prefix cache 的命中条件是完全一致的 Token 前缀。可以采用以下配置提高命中率：

1. **稳定内容放前面，动态内容放后面。** 先 system prompt、工具 schema、共享文档，最后再放用户问题。
2. **把时间戳、request ID 和随机数放在共享前缀之后。** 这些动态字段之后的 Token 都无法复用。
3. **固定聊天模板和 tool schema。** JSON 字段顺序、空格、工具排列都尽量 canonicalize。
4. **同一份长文档不要每次换格式。** 预处理一次并保存相同 Token 序列。
5. **做 cache-aware routing。** 多个 worker 时，把相同 tenant、conversation 或 document 的请求尽量路由到持有对应 KV 的 worker。
6. **减少 worker 重启和模型切换。** 两者都会清除或替换已有缓存。
7. **为高频共享前缀分配足够的 KV 空间。** 容量不足会频繁 eviction，使实际命中率下降。
8. **记录缓存指标。** 分开记录 prefill tokens、cached tokens、TTFT、hit rate 和 eviction。

提高共享前缀长度的布局：

```text
[固定 system prompt]
[固定工具定义]
[固定公司知识/文档]
[稳定的历史对话]
[本轮用户问题]
```

动态字段截断共享前缀的布局：

```text
[当前时间 + 随机 request id]
[本轮问题]
[固定 system prompt]
[固定公司文档]
```

共享 KV cache 可能产生跨租户侧信道和数据隔离问题。多租户服务应使用 cache salt/namespace、租户隔离和访问控制；不同客户的敏感前缀不应进入同一缓存命名空间。vLLM 提供 `cache_salt` 等机制，具体行为应以部署版本的文档和安全测试为准。

---

## 第九部分：三套实际方案

### 方案一：我只想在自己的电脑上聊天

- 工具：LM Studio 或 Ollama；
- 模型：先从 7B～14B instruct、Q4/Q5 开始；
- context：先 8K；
- 目标：能完整放进显存/统一内存，单请求 decode 快；
- 指标：TTFT、单用户 token/s、回答质量。

### 方案二：我有一台 128GB 统一内存机器，想跑大 MoE

- 工具：Apple 优先 MLX/llama.cpp/LM Studio；AMD Strix Halo 优先实测 llama.cpp Vulkan/HIP/ROCm；DGX Spark 可用 CUDA/vLLM/TensorRT-LLM；
- 模型：看总参数确保权重能装下，再看 active parameters 估计 Decode；
- 量化：通常先 Q4/Q5 或官方 FP8/FP4 支持格式；
- 指标：真实 token/s、prompt processing speed、功耗、长上下文稳定性；
- 注意：新 MoE 架构可能权重装得下，但 runtime 还没有高效 fused kernel。

### 方案三：几十到几百人共用一个内部模型

- 操作系统：Linux；
- 工具：vLLM 或 SGLang；仅使用 NVIDIA 且配置固定时，可以同时测试 TensorRT-LLM；
- API：OpenAI-compatible endpoint 前加认证、限流和观测；
- 并行：先单机多 GPU TP/EP，再考虑多机；
- 优化：continuous batching、PagedAttention、prefix caching、cache-aware routing、chunked prefill；
- 指标：总 output tokens/s、requests/s、P95 TTFT、P95 ITL、每成功任务成本；
- 压测：使用真实 prompt/output 长度分布。128 input / 128 output 的结果不能代表长文档输入、长 reasoning 输出或混合长度请求。

如果 workload 里有大量共享长文档，prefix cache 会非常重要；如果回答特别长，speculative decoding、低 bit 权重和 Decode 带宽更重要；如果输入特别长，prefill 算力和 chunked prefill 更重要。

---

## 常见误区

### 模型文件 20GB，是否一定能在 24GB 显存上运行？

不一定。24GB 还需要容纳 KV cache、activation、workspace 和 runtime；前文示例中的单个 32K BF16 KV cache 已约为 4GiB。

### 量化成 4-bit，是否一定达到 FP16 的四倍速度？

不一定。4-bit 权重的原始容量约为 FP16 的 1/4，但还包括 scale；W4A16 在矩阵乘法前需要反量化，且实际吞吐受内存带宽、kernel 和 GPU 指令支持限制。

### H100 能加载 NVFP4，是否表示它有原生 FP4 加速？

不是。Hopper 原生支持 FP8 Tensor Core；原生 FP4 Tensor Core 从 Blackwell 开始提供。

### MoE 激活 3B，是否只需要 3B 模型的内存？

不是。内存仍要容纳全部专家和共享层；激活参数主要影响每 Token 的计算量和理想权重读取量。

### 四张显卡的容量和性能能否直接相加？

容量可以通过模型切分利用，但 TP、PP 和 EP 会引入不同的同步与通信开销，吞吐不会按卡数自动线性增长。

### 模型支持 1M context，是否应把服务上限直接设为 1M？

不应默认这样设置。context 上限会影响 KV cache 预算和可支持并发；长输入也会增加 prefill 时间。应根据实际输入长度分布设置，并用目标任务测试长上下文 recall。

### 每百万 Token 单价较低的模型，是否一定具有较低的任务成本？

不一定。reasoning 长度、tokenizer、缓存率、重试次数和任务成功率都会改变每个成功任务的成本。

### Prefix caching 是否会跳过整段回答的计算？

不会。Prefix caching 只跳过命中前缀的 prefill；新输出的 Decode 仍要逐 Token 计算。

### 排行榜第一是否一定适合自己的任务？

不一定。最终选择需要使用目标数据、prompt、量化、运行时和硬件评测。

---

## 延伸阅读与资料来源

以下资料以论文、项目官方文档和芯片厂商文档为主：

- Transformer：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- H200：[NVIDIA 官方规格](https://www.nvidia.com/en-us/data-center/h200/)
- PagedAttention：[论文](https://arxiv.org/abs/2309.06180)与 [vLLM 介绍](https://vllm.ai/blog/2023-06-20-vllm)
- vLLM prefix caching：[Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- vLLM speculative decoding：[Speculators 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/)
- DeepSeek-V3 MTP 与 FP8 权重：[官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)
- Speculative decoding 与硬件 compute/bandwidth ratio：[Sequoia](https://arxiv.org/abs/2402.12374)
- 消费级 GPU 的 MTP A/B 记录：[Qwen3.8-27B MTP 社区 benchmark](https://github.com/sudoingX/qwen38-mtp)
- OCP MX formats：[Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- NVIDIA NVFP4：[Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- NVIDIA 各精度 GEMM 实测说明：[Transformer Engine Speedups](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html)
- AMD CDNA4：[架构说明](https://www.amd.com/en/technologies/cdna.html)
- Artificial Analysis：[模型比较](https://artificialanalysis.ai/models/)与[方法](https://artificialanalysis.ai/methodology)
- 人类成对偏好评测：[Chatbot Arena 论文](https://arxiv.org/abs/2403.04132)
- 降低公开测试集污染：[LiveBench](https://arxiv.org/abs/2406.19314)
- Reasoning 与 test-time compute：[OpenAI o1 说明](https://openai.com/index/learning-to-reason-with-llms/)
- RL reasoning：[DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)
- 非可验证写作任务的奖励设计：[Writing-Zero](https://arxiv.org/abs/2506.00103)
- llama.cpp：[GitHub](https://github.com/ggml-org/llama.cpp)
- vLLM：[文档](https://docs.vllm.ai/)
- SGLang：[GitHub](https://github.com/sgl-project/sglang)
- TensorRT-LLM：[文档](https://nvidia.github.io/TensorRT-LLM/)
- NVIDIA NIM LLM：[官方概览](https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html)
- OpenVINO GenAI：[GitHub](https://github.com/openvinotoolkit/openvino.genai)
- IPEX-LLM 归档状态：[官方仓库](https://github.com/intel/ipex-llm)

部署前先确定模型、上下文长度、并发和延迟目标，再计算权重与 KV cache 容量，并使用真实任务测量准确率、TTFT、ITL、throughput 和功耗。

模型能够启动只说明当前配置满足最低运行条件。是否适合长期部署，还取决于输出质量、延迟、吞吐、稳定性和总成本。
