# Agent 核心架构与技术概念 八股文

## 目录
1. [ReAct](#1-react)
2. [Plan-and-Execute](#2-plan-and-execute)
3. [Tool Calling / Function Calling](#3-tool-calling--function-calling)
4. [RAG (检索增强生成)](#4-rag-检索增强生成)
5. [Memory (记忆系统)](#5-memory-记忆系统)
6. [Multi-Agent (多智能体)](#6-multi-agent-多智能体)
7. [Agent Loop (智能体循环)](#7-agent-loop-智能体循环)
8. [Prompt Engineering (提示工程)](#8-prompt-engineering-提示工程)
9. [Embedding (嵌入)](#9-embedding-嵌入)
10. [Vector Database (向量数据库)](#10-vector-database-向量数据库)
11. [LangChain 架构](#11-langchain-架构)
12. [LlamaIndex 架构](#12-llamaindex-架构)

---

## 1. ReAct

### 定义
ReAct（**Re**asoning + **Act**ing）是一种将推理（Reasoning）和行动（Acting）交织进行的 Agent 范式，由 Yao et al. (2022) 提出。核心思想是让 LLM 在每一步先进行思考（Thought），然后决定采取什么行动（Action），观察行动结果（Observation），再进行下一轮思考，形成 `Thought → Action → Observation` 的循环。

### 原理
```
循环执行：
1. Thought: 分析当前状态，推理下一步应该做什么
2. Action: 选择并执行一个工具/动作
3. Observation: 获取工具返回的结果
4. 重复直到任务完成（输出 Final Answer）
```

- 推理链作为"内心独白"（inner monologue），帮助模型追踪任务状态
- 行动步骤让模型能与外部环境交互获取信息
- 每轮的 Observation 作为下一轮 Thought 的输入上下文

### 优点
- ✅ **可解释性强**：Thought 步骤提供了清晰的推理过程，便于调试
- ✅ **灵活性高**：可根据中间结果动态调整策略
- ✅ **简单有效**：实现简单，适合大多数单步推理任务
- ✅ **减少幻觉**：通过实际工具调用获取信息，而非凭空生成

### 缺点
- ❌ **容易陷入循环**：可能重复执行相同工具调用
- ❌ **缺乏全局规划**：每步都是局部决策，缺乏对整体任务的规划
- ❌ **token 消耗大**：每轮都要携带完整历史，上下文逐渐膨胀
- ❌ **错误累积**：早期错误推理可能影响后续决策

### 典型实现
LangChain 中的 `create_react_agent`，OpenAI 的 Function Calling 模式本质上也是一种 ReAct 变体。

---

## 2. Plan-and-Execute

### 定义
Plan-and-Execute（先规划后执行）是一种将任务分解为"规划阶段"和"执行阶段"的 Agent 架构。先用一个 Planner LLM 生成完整的任务计划，然后用 Executor 逐步执行每个子任务，必要时可以重新规划（Re-planning）。

### 原理
```
1. Planning Phase:
   - 输入: 用户任务描述
   - Planner LLM 将任务分解为有序子任务列表
   - 输出: [Step 1, Step 2, ..., Step N]

2. Execution Phase:
   - Executor 逐步执行每个子任务
   - 每个子任务可以使用不同的工具
   - 执行结果累积为中间状态

3. Re-planning (可选):
   - 根据执行结果判断是否需要调整计划
   - 如果发现新信息或计划不可行，重新生成计划
```

### 优点
- ✅ **全局视野**：规划阶段能看到任务全貌，步骤间有逻辑连贯性
- ✅ **可预测性强**：执行前就能看到完整计划，便于预估成本和时间
- ✅ **适合复杂任务**：多步骤、需要多工具协作的任务
- ✅ **错误隔离**：单个子任务失败不影响整体规划

### 缺点
- ❌ **规划成本高**：一次规划的 LLM 调用可能消耗较多 token
- ❌ **不够灵活**：如果计划有误，整体执行可能偏离目标
- ❌ **依赖 LLM 能力**：规划质量完全取决于 LLM 的推理能力
- ❌ **延迟较高**：需要先完成规划才能开始执行

### 典型实现
- LangChain 的 `PlanAndExecute`
- LangGraph 中的 Plan-and-Execute 模板
- BabyAGI（任务自动分解与执行）

---

## 3. Tool Calling / Function Calling

### 定义
Tool Calling（工具调用）是 LLM 与外部工具/API 交互的标准化接口机制。LLM 根据用户需求，选择合适的工具并生成调用参数，由外部系统执行后返回结果给 LLM。Function Calling 是 Tool Calling 的一种特定形式，由 OpenAI 率先推广。

### 原理
```
1. 工具定义阶段：
   - 开发者向 LLM 声明可用工具的 Schema（名称、描述、参数类型）
   - 通常使用 JSON Schema 格式定义

2. 调用决策阶段：
   - LLM 分析用户请求，判断是否需要调用工具
   - 如果需要，生成结构化的工具调用请求（函数名 + 参数）
   - 返回特殊的 tool_call 消息而非普通文本

3. 执行与返回阶段：
   - 应用层捕获 tool_call，执行实际的函数/API 调用
   - 将执行结果以 tool_result 消息格式返回给 LLM
   - LLM 基于工具返回结果生成最终回复
```

### 优点
- ✅ **标准化接口**：统一了 LLM 与外部世界的交互方式
- ✅ **减少幻觉**：通过工具获取真实数据，而非模型编造
- ✅ **可扩展性**：可以不断添加新工具
- ✅ **安全性**：参数经过 Schema 验证，格式可控

### 缺点
- ❌ **并行调用限制**：大多数模型不支持原生并行工具调用
- ❌ **参数格式错误**：LLM 有时生成不符合 Schema 的参数
- ❌ **工具选择偏差**：可能选错工具或生成错误的参数组合
- ❌ **延迟增加**：每次工具调用都增加一个往返

### Tool Calling vs Function Calling
| 维度 | Tool Calling | Function Calling |
|------|-------------|-----------------|
| 范围 | 更广，包括搜索、代码执行等任何工具 | 特指函数/API 调用 |
| 标准 | Anthropic/Mistral 等使用 | OpenAI 率先推广 |
| 本质 | Function Calling 是 Tool Calling 的子集 | |
| 接口 | 可能支持多种输出格式 | 通常是 JSON 结构化输出 |

---

## 4. RAG (检索增强生成)

### 定义
RAG（**R**etrieval **A**ugmented **G**eneration，检索增强生成）是一种将外部知识检索与 LLM 生成相结合的技术范式。核心思想是：在生成回答前，先从外部知识库中检索相关文档，将检索结果作为上下文注入 prompt，让 LLM 基于检索到的真实信息来生成回答。

### 原理
```
完整 RAG Pipeline:
1. 索引阶段 (Indexing):
   - 文档加载 → 文档分割(Chunking) → Embedding 向量化 → 存入 Vector DB

2. 检索阶段 (Retrieval):
   - 用户查询 → Query Embedding → 向量相似度搜索 → Top-K 相关文档片段

3. 增强阶段 (Augmentation):
   - 将检索到的文档片段 + 用户原始问题 → 组装成增强 Prompt

4. 生成阶段 (Generation):
   - LLM 基于增强 Prompt 生成最终回答
```

### 高级 RAG 技术
- **Query Rewriting**: 改写用户查询以提高检索效果
- **Hybrid Search**: 结合关键词搜索(BM25) + 向量搜索
- **Reranking**: 对检索结果进行二次排序
- **Self-RAG**: LLM 自主决定是否需要检索、何时检索
- **GraphRAG**: 结合知识图谱进行检索
- **Corrective RAG (CRAG)**: 检索后评估质量，必要时重新检索

### 优点
- ✅ **减少幻觉**：基于真实检索结果生成，有据可依
- ✅ **知识实时更新**：无需重新训练模型，更新知识库即可
- ✅ **可溯源**：可以标注信息来源，增强可信度
- ✅ **领域适应性强**：同一模型 + 不同知识库即可适配不同领域

### 缺点
- ❌ **检索质量瓶颈**：检索不准确会导致生成质量下降
- ❌ **Chunking 策略敏感**：分块大小、重叠程度直接影响效果
- ❌ **上下文窗口限制**：检索结果过多可能超出 LLM 上下文长度
- ❌ **延迟增加**：需要先检索再生成，比直接生成慢
- ❌ **维护成本**：需要持续维护和更新知识库

---

## 5. Memory (记忆系统)

### 定义
Agent Memory 是赋予 LLM Agent 持久化和检索信息能力的系统，使 Agent 能够记住历史交互、学习经验并在未来任务中利用这些记忆。类比人类记忆，分为短期记忆、长期记忆和向量记忆等。

### 5.1 短期记忆 (Short-term Memory / Working Memory)

#### 定义
短期记忆对应当前对话上下文，存储在 LLM 的上下文窗口中。包括对话历史、当前任务状态等临时信息。

#### 原理
- 直接存储在 LLM 的 context window 中
- 每次推理时完整传入
- 受限于模型的最大上下文长度（如 4K、128K 等）

#### 实现方式
- 对话历史数组（messages list）
- 滑动窗口：只保留最近 N 轮对话
- 摘要压缩：对早期对话进行摘要后保留

#### 优点：实现简单、访问延迟为零、信息精确
#### 缺点：受限于上下文窗口、成本随 token 线性增长、会话结束后丢失

### 5.2 长期记忆 (Long-term Memory)

#### 定义
长期记忆是跨会话持久化存储的信息，使 Agent 能记住用户偏好、历史交互摘要、学习到的经验等。

#### 原理
```
存储:
- 将重要信息提取并序列化
- 存入数据库（关系型/键值/文档数据库）

检索:
- 基于规则匹配（如用户ID → 偏好设置）
- 基于语义相似度搜索
- 基于时间衰减的优先级排序
```

#### 常见实现
- **Memory Store**: key-value 存储用户偏好
- **Conversation Database**: 存储历史对话摘要
- **Experience Replay**: 存储成功/失败经验供 Agent 学习
- **MemGPT**: 模拟操作系统的分层记忆管理

#### 优点：跨会话持久化、可存储海量信息、支持个性化
#### 缺点：需要额外存储基础设施、检索可能引入不相关信息、需要记忆管理策略（遗忘/压缩）

### 5.3 向量记忆 (Vector Memory)

#### 定义
向量记忆是利用 Embedding 向量进行语义化存储和检索的记忆系统。将信息转换为高维向量后存入向量数据库，查询时通过向量相似度搜索找到最相关的记忆。

#### 原理
```
1. 写入: 信息 → Embedding Model → 向量 → Vector DB
2. 查询: 查询文本 → Embedding Model → 查询向量 → 
         相似度搜索(余弦/内积/欧氏距离) → Top-K 结果
3. 注入: 将检索到的记忆注入到 Agent 的上下文中
```

#### 优点：语义检索能力强、支持模糊匹配、可扩展性好
#### 缺点：依赖 Embedding 模型质量、向量搜索有误差、需要额外的存储和计算资源

### 5.4 多层记忆架构
现代 Agent 系统通常采用多层记忆架构（如 MemGPT、Generative Agents）：
```
┌─────────────────────────────────┐
│  Working Memory (工作记忆)       │  ← 当前上下文
├─────────────────────────────────┤
│  Recency Cache (近期缓存)        │  ← 最近交互摘要
├─────────────────────────────────┤
│  Semantic Memory (语义记忆)      │  ← 向量化长期记忆
├─────────────────────────────────┤
│  Archival Storage (归档存储)     │  ← 全量历史数据
└─────────────────────────────────┘
```

---

## 6. Multi-Agent (多智能体)

### 定义
Multi-Agent 是指多个 Agent 协作完成复杂任务的系统架构。每个 Agent 具有独立的角色、能力、记忆和工具，通过通信机制进行协调。

### 原理
```
核心组件:
1. Agent 定义: 角色、目标、能力、可用工具
2. 通信机制: Agent 之间如何传递消息和共享状态
3. 协调策略: 如何决定任务分配和执行顺序
4. 共享状态: 全局状态管理（对话历史、任务进度等）
```

### 协作模式

| 模式 | 描述 | 代表框架 |
|------|------|---------|
| **协作式 (Collaborative)** | 多个 Agent 平等协作，各司其职 | CrewAI, AutoGen |
| **层级式 (Hierarchical)** | 有 Manager Agent 分配和监督子 Agent | LangGraph Supervisor |
| **辩论式 (Debate)** | 多个 Agent 对同一问题辩论，综合结论 | |
| **流水线式 (Pipeline)** | Agent 按顺序处理，前一个的输出是后一个的输入 | |
| **竞争式 (Competitive)** | 多个 Agent 独立完成同一任务，选择最佳结果 | |

### 优点
- ✅ **专业化分工**：每个 Agent 专注特定领域，效果更好
- ✅ **可扩展**：新增能力只需添加新 Agent
- ✅ **容错性**：单个 Agent 失败不影响整体
- ✅ **复杂任务**：能处理单 Agent 难以胜任的复杂任务

### 缺点
- ❌ **协调复杂**：Agent 间的通信和状态同步开销大
- ❌ **成本高**：多 Agent 同时运行消耗大量 token
- ❌ **调试困难**：问题可能出在任何一个 Agent 或交互环节
- ❌ **一致性问题**：不同 Agent 可能产生矛盾信息

### 典型框架
- **CrewAI**: 角色扮演式多 Agent 协作
- **AutoGen (Microsoft)**: 对话式多 Agent 框架
- **LangGraph**: 基于图的 Agent 编排
- **MetaGPT**: 模拟软件公司的多 Agent 系统
- **ChatDev**: 模拟开发团队的多 Agent 系统

---

## 7. Agent Loop (智能体循环)

### 定义
Agent Loop 是 Agent 执行任务时的核心运行机制，指 LLM 在一个循环中不断"感知-思考-行动"直到完成任务的过程。它是所有 Agent 范式（ReAct、Plan-and-Execute 等）的底层运行模式。

### 原理
```
Agent Loop 伪代码:
def agent_loop(task):
    messages = [system_prompt, user_message(task)]
    while True:
        response = llm.generate(messages)  # LLM 推理
        
        if response.has_tool_call:         # 判断是否需要工具
            result = execute_tool(response.tool_call)
            messages.append(tool_result(result))
            continue                       # 继续循环
        else:
            return response.content        # 任务完成，返回最终答案
```

### 关键组件
1. **终止条件**：何时退出循环（最终答案、最大迭代次数、错误阈值）
2. **消息管理**：如何维护和更新上下文消息列表
3. **错误处理**：工具调用失败时的重试/回退策略
4. **流式输出**：支持 streaming 的 Agent Loop 实现

### 变体
- **ReAct Loop**: Thought-Action-Observation 循环
- **Plan-Execute Loop**: 先规划后逐步执行
- **Reflection Loop**: 执行后反思并改进
- **Recursive Loop**: Agent 可以委托子 Agent

### 优点
- ✅ **通用性强**：几乎所有 Agent 框架都基于此模式
- ✅ **可观测性**：循环的每一步都可以被记录和调试
- ✅ **可控性**：通过限制最大迭代次数防止无限循环

### 缺点
- ❌ **无界风险**：如果没有合适的终止条件，可能无限循环
- ❌ **上下文膨胀**：循环次数越多，上下文越长，成本越高
- ❌ **延迟累积**：每轮 LLM 调用都有延迟，多轮累积可观

---

## 8. Prompt Engineering (提示工程)

### 定义
Prompt Engineering 是设计和优化输入给 LLM 的文本（Prompt），以引导模型产生期望输出的技术和方法论。在 Agent 系统中，Prompt Engineering 是连接用户意图与 LLM 能力的桥梁。

### 核心技术

#### 8.1 零样本提示 (Zero-shot Prompting)
直接给模型任务描述，不提供示例。

#### 8.2 少样本提示 (Few-shot Prompting)
提供少量输入-输出示例来引导模型。

#### 8.3 链式思考 (Chain-of-Thought, CoT)
引导模型展示推理过程，如"让我们一步一步思考"。

#### 8.4 自洽性 (Self-Consistency)
生成多条推理路径，选择出现频率最高的答案。

#### 8.5 思维树 (Tree of Thoughts, ToT)
让模型在每步生成多个候选思路，评估后选择最佳路径继续。

#### 8.6 System Prompt 设计
- 角色设定（Persona）
- 行为约束（Constraints）
- 输出格式要求（Output Format）
- 工具使用说明（Tool Instructions）
- 安全边界（Safety Guidelines）

### 在 Agent 中的应用
- 定义 Agent 的角色和能力边界
- 设计工具调用的 Prompt 模板
- 构建 RAG 的查询改写和答案生成 Prompt
- 多 Agent 间的通信协议设计

### 优点
- ✅ **成本低**：不需要训练模型，只需优化输入
- ✅ **即时生效**：修改 Prompt 即可看到效果
- ✅ **灵活多变**：同一模型不同 Prompt 可产生截然不同的效果

### 缺点
- ❌ **脆弱性**：微小的 Prompt 变化可能导致输出质量剧变
- ❌ **缺乏理论指导**：很大程度上依赖经验试错
- ❌ **模型依赖**：不同模型需要不同的 Prompt 策略
- ❌ **安全风险**：恶意 Prompt 可能绕过安全限制（Prompt Injection）

---

## 9. Embedding (嵌入)

### 定义
Embedding（向量嵌入）是将文本、图像等非结构化数据转换为固定维度的稠密向量表示的技术。这些向量在高维空间中编码了语义信息，使得语义相似的内容在向量空间中距离更近。

### 原理
```
输入: "机器学习是人工智能的一个子领域"
       ↓ Embedding Model (如 text-embedding-3-small)
输出: [0.023, -0.156, 0.892, ..., 0.045]  (1536维向量)
```

### 核心概念
1. **语义相似度**: 两个语义相近的文本，其 Embedding 向量的余弦相似度高
2. **维度**: 通常 384~3072 维，维度越高信息容量越大但计算成本也越高
3. **归一化**: 常用 L2 归一化，将向量映射到单位超球面上

### 主流 Embedding 模型
| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 性价比高，效果好 |
| OpenAI text-embedding-3-large | 3072 | 最高性能 |
| BGE-M3 | 1024 | 开源，支持多语言 |
| Cohere embed-v3 | 1024 | 搜索/分类多任务 |
| Jina Embeddings v3 | 1024 | 开源，长文本支持好 |
| nomic-embed-text | 768 | 开源，轻量级 |

### 在 Agent 中的应用
- **RAG**: 文档向量化存储 + 语义检索
- **Memory**: 向量化存储 Agent 的记忆
- **Tool Selection**: 通过语义匹配选择最合适的工具
- **Long-term Memory Retrieval**: 从历史记录中检索相关信息

### 优点
- ✅ 捕获深层语义信息，超越关键词匹配
- ✅ 支持多语言、跨模态的统一表示
- ✅ 向量运算效率高，适合大规模检索

### 缺点
- ❌ 信息压缩可能丢失细节
- ❌ 对长文本的表示能力有限
- ❌ 领域适配可能需要微调

---

## 10. Vector Database (向量数据库)

### 定义
Vector Database（向量数据库）是专门用于存储、索引和检索高维向量的数据库系统。它支持高效的近似最近邻（ANN）搜索，是 RAG、Agent Memory 等系统的核心基础设施。

### 原理
```
核心操作:
1. 插入 (Insert): 向量 + 元数据 → 存储并建立索引
2. 搜索 (Search): 查询向量 → ANN 索引 → 返回 Top-K 最近邻
3. 更新 (Update): 修改向量或元数据
4. 删除 (Delete): 移除向量

索引算法:
- HNSW (Hierarchical Navigable Small World): 最常用，平衡精度和速度
- IVF (Inverted File Index): 适合大规模数据
- PQ (Product Quantization): 降低存储和计算成本
- Flat: 暴力搜索，100% 精度但最慢
```

### 主流向量数据库对比
| 数据库 | 类型 | 特点 |
|--------|------|------|
| **Pinecone** | 云托管 | 易用，自动扩展，价格较高 |
| **Weaviate** | 开源/云 | 支持混合搜索，GraphQL API |
| **Milvus/Zilliz** | 开源/云 | 高性能，支持多种索引，分布式 |
| **ChromaDB** | 开源 | 轻量级，适合原型开发 |
| **Qdrant** | 开源/云 | Rust 编写，性能优异 |
| **pgvector** | PostgreSQL 扩展 | 无需额外基础设施 |
| **Redis** | 内存数据库 | 超低延迟，支持向量搜索 |
| **FAISS** | 库（非数据库） | Meta 开发，GPU 加速，单机高性能 |

### 与传统数据库的区别
| 维度 | 传统数据库 | 向量数据库 |
|------|-----------|-----------|
| 查询方式 | 精确匹配 (SQL) | 近似相似度搜索 |
| 索引结构 | B-Tree, Hash | HNSW, IVF, PQ |
| 结果 | 确定性 | 概率性 (近似最近邻) |
| 典型负载 | 结构化数据 | 高维稠密向量 |

### 优点
- ✅ 高效的语义搜索能力
- ✅ 支持大规模向量检索（百万~十亿级）
- ✅ 支持元数据过滤 + 向量搜索的混合查询
- ✅ 向生态不断成熟，与主流框架深度集成

### 缺点
- ❌ 需要额外的存储和计算资源
- ❌ 向量索引的构建和维护有成本
- ❌ 不适合精确查询场景
- ❌ 向量相似度不等于业务相关性

---

## 11. LangChain 架构

### 定义
LangChain 是一个用于构建 LLM 应用的开源框架，提供了模块化的组件和编排工具，帮助开发者快速构建从简单聊天机器人到复杂 Agent 系统的各种应用。

### 核心架构
```
LangChain 生态:
┌─────────────────────────────────────────────┐
│  LangChain (核心库)                          │
│  ├── Models (模型抽象层)                      │
│  ├── Prompts (提示模板)                       │
│  ├── Chains (链式调用)                        │
│  ├── Agents (智能体)                         │
│  ├── Memory (记忆管理)                        │
│  ├── Retrievers (检索器)                     │
│  └── Tools (工具接口)                        │
├─────────────────────────────────────────────┤
│  LangGraph (Agent 编排)                      │
│  ├── State Graph (状态图)                    │
│  ├── Nodes & Edges (节点与边)                │
│  ├── Checkpointing (检查点)                  │
│  └── Human-in-the-loop (人机协作)            │
├─────────────────────────────────────────────┤
│  LangSmith (可观测性平台)                     │
│  ├── Tracing (追踪)                         │
│  ├── Evaluation (评估)                       │
│  └── Prompt Hub (提示管理)                   │
├─────────────────────────────────────────────┤
│  LangServe (部署)                            │
│  └── 将 Chain/Agent 部署为 REST API           │
└─────────────────────────────────────────────┘
```

### 关键组件

#### Models
- 统一接口抽象 LLM 和 Chat Model
- 支持 OpenAI, Anthropic, Google, 本地模型等
- 通过 Provider 抽象层切换模型

#### Chains
- **LCEL (LangChain Expression Language)**: 基于管道的声明式组合
- 支持并行、条件分支、重试等
- 已取代旧版 Chain 类，更灵活

#### Agents
- 基于 LangGraph 实现
- 支持 ReAct、Plan-and-Execute 等多种模式
- 支持工具调用、人机交互、断点恢复

#### Memory
- ConversationBufferMemory: 完整历史
- ConversationSummaryMemory: 摘要记忆
- VectorStoreRetrieverMemory: 向量检索记忆

### 优点
- ✅ **生态丰富**：集成 100+ 工具和数据源
- ✅ **模块化设计**：组件可自由组合
- ✅ **LangGraph 强大**：支持复杂的有状态 Agent 编排
- ✅ **社区活跃**：文档完善，社区支持好
- ✅ **LangSmith 可观测性**：内置追踪和评估工具

### 缺点
- ❌ **抽象层过重**：过度封装可能导致性能开销和调试困难
- ❌ **API 变化频繁**：版本迭代快，向后兼容性差
- ❌ **学习曲线**：概念多，新版本变化大
- ❌ **简单任务过重**：简单场景不需要这么复杂的框架

---

## 12. LlamaIndex 架构

### 定义
LlamaIndex（原名 GPT Index）是一个专为 LLM 数据连接和 RAG 设计的开源框架。它专注于将外部数据与 LLM 高效连接，提供数据摄入、索引、查询和编排的完整工具链。

### 核心架构
```
LlamaIndex 架构:
┌─────────────────────────────────────────┐
│  Data Connectors (数据连接器)             │
│  └── 支持 100+ 数据源 (文件/API/数据库)    │
├─────────────────────────────────────────┤
│  Data Ingestion (数据摄入)                │
│  ├── Document Loaders                     │
│  ├── Text Splitters (文本分割)            │
│  └── Transformers (转换器)                │
├─────────────────────────────────────────┤
│  Indexing (索引构建)                       │
│  ├── Vector Store Index                  │
│  ├── Tree Index                          │
│  ├── Summary Index                       │
│  ├── Knowledge Graph Index               │
│  └── SubQuestion Query Engine            │
├─────────────────────────────────────────┤
│  Query Engine (查询引擎)                   │
│  ├── Retriever (检索器)                   │
│  ├── Response Synthesizer (响应合成器)    │
│  └── Router (路由)                       │
├─────────────────────────────────────────┤
│  Agents & Workflows (代理与工作流)         │
│  ├── Function Calling Agent              │
│  ├── Workflow 编排                       │
│  └── 多 Agent 协作                       │
└─────────────────────────────────────────┘
```

### 索引类型
| 索引类型 | 适用场景 | 原理 |
|---------|---------|------|
| Vector Store | 语义检索 | 向量相似度搜索 |
| Tree | 层级问答 | 构建文档树，逐层摘要 |
| Summary | 全局摘要 | 对整个文档摘要 |
| Knowledge Graph | 结构化关系 | 实体关系图谱 |
| SubQuestion | 复杂问题分解 | 将大问题拆分为子问题 |
| Keyword | 关键词匹配 | BM25 等传统检索 |

### 与 LangChain 的对比
| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| **核心定位** | 数据连接 & RAG | 通用 LLM 应用框架 |
| **RAG 能力** | ⭐⭐⭐⭐⭐ (核心优势) | ⭐⭐⭐⭐ |
| **Agent 能力** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (核心优势) |
| **索引类型** | 丰富（向量/树/摘要/KG） | 主要依赖向量 |
| **编排能力** | Workflow (较新) | LangGraph (成熟) |
| **学习曲线** | RAG 场景较低 | 整体较高 |
| **数据处理** | 内置完善 | 需要额外集成 |

### 优点
- ✅ **RAG 专家**：RAG 场景的最佳选择
- ✅ **索引类型丰富**：支持多种索引策略
- ✅ **数据连接器丰富**：开箱即用的数据源支持
- ✅ **评估工具**：内置 RAG 评估指标
- ✅ **与 LangChain 互补**：可结合使用

### 缺点
- ❌ **Agent 能力较弱**：Agent 编排不如 LangChain/LangGraph
- ❌ **生态较窄**：主要聚焦 RAG 场景
- ❌ **大规模生产**：部分功能在大规模场景下需要优化
- ❌ **版本变化**：v0.10+ 重构较大

---

## 附录：概念关系图

```
┌──────────────────────────────────────────────────────┐
│                    Agent 系统全景图                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐    │
│  │ Prompt  │───→│   LLM    │───→│ Agent Loop   │    │
│  │Enginneer│   │(核心推理) │    │ (ReAct/P&E)  │    │
│  └─────────┘    └────┬─────┘    └──────┬───────┘    │
│                      │                  │             │
│              ┌───────┼──────────┐      │             │
│              ▼       ▼          ▼      ▼             │
│         ┌────────┐ ┌───────┐ ┌──────────────────┐   │
│         │  Tool  │ │Memory │ │   Multi-Agent     │   │
│         │Calling │ │短期/长期│ │   (协作/层级)      │   │
│         └────────┘ │向量记忆│ └──────────────────┘   │
│                    └───┬───┘                         │
│                        │                             │
│              ┌─────────┼─────────┐                   │
│              ▼                   ▼                   │
│         ┌─────────┐      ┌─────────────┐            │
│         │RAG 系统  │      │  Vector DB  │            │
│         │(检索增强) │←────→│(向量数据库)  │            │
│         └────┬────┘      └─────────────┘            │
│              │                                       │
│         ┌────┴────┐                                  │
│         │Embedding│                                  │
│         │(向量嵌入)│                                  │
│         └─────────┘                                  │
│                                                      │
│  框架层: LangChain ←→ LlamaIndex                     │
│          (通用编排)    (RAG 专家)                      │
└──────────────────────────────────────────────────────┘
```

---

## 参考资源
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- Wang et al., "Plan-and-Solve Prompting" (2023)
- LangChain 官方文档: https://docs.langchain.com
- LlamaIndex 官方文档: https://docs.llamaindex.ai
- Awesome-AI-Memory: https://github.com/bigai-nlco/Awesome-AI-Memory
- Awesome-Context-Engineering: https://github.com/Meirtz/Awesome-Context-Engineering
