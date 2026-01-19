# AIgorithm Agent 项目架构文档

## 项目概述

这是一个专为教育内容问答设计的 RAG (Retrieval-Augmented Generation) 系统，特别针对算法和数据结构教材的知识检索进行了优化。系统结合了向量搜索 (FAISS)、词汇搜索 (BM25)、知识图谱功能，并支持多个 LLM 提供商，以提供准确、引用规范的算法相关问题的答案。

### 核心特性

- **双重检索**: 结合语义搜索 (FAISS) 和词汇搜索 (BM25)
- **知识图谱**: 基于算法概念的先修知识检索
- **多 LLM 支持**: DeepSeek、SiliconFlow、OpenAI 兼容 API
- **智能引用**: 自动生成带引用的答案，支持书名和章节信息
- **PEV Agent**: Planner-Executor-Verifier 三段式推理框架

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         API 层 (FastAPI)                        │
│  /faiss_search | /bm25_search | /rag_answer | /qa_search        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        PEV Agent 框架                           │
│  Planner → Executor → Verifier (支持重规划和迭代优化)             │
└───────┬─────────────┬─────────────┬──────────────────────────────┘
        │             │             │
┌───────▼──────┐ ┌───▼────┐ ┌─────▼──────────────────────────────┐
│ FAISS 检索   │ │BM25检索│ │      知识图谱 QA 系统             │
│ (语义相似度) │ │(词汇匹配)│ │  · 实体关系匹配                   │
└───────┬──────┘ └───┬────┘ │  · 先修知识推理                   │
        │            │      │  · 概念属性查询                   │
        └──────┬─────┘      └────────────────────────────────────┘
               │
┌───────▼───────────────────────────────────────────────────────┐
│                    文档存储层                                   │
│  refined_document_chunks.json | faiss2.index                   │
│  knowledge_graph/*.json | qa_bank/*.json                       │
└───────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. API 服务器 (`src/api_server.py`)

**功能**: FastAPI REST API 服务器，提供 RAG 能力的外部访问接口

**主要端点**:

| 端点 | 方法 | 功能 |
|------|------|------|
| `/faiss_search` | POST | 向量相似度搜索 |
| `/bm25_search` | POST | BM25 词汇搜索 |
| `/rag_answer` | POST | 组合 RAG 问答（带引用） |
| `/pre_knowledge_search` | POST | 基于知识图谱的先修知识检索 |
| `/qa_search` | POST | 高级题库搜索 |
| `/smart_answer` | POST | **智能推荐入口（自动决策返回内容）** |

**请求/响应格式**:
```python
# 请求
{
    "query": "什么是快速排序算法？",
    "top_k": 10
}

# 响应
{
    "answer": "快速排序是一种分治算法...",
    "citations": ["《算法导论》第七章", ...]
}
```

---

### 2. 主 RAG 管道 (`src/main.py`)

**功能**: 核心 RAG 实现，双重检索 + 结构化答案生成

**关键函数**:

| 函数 | 功能 |
|------|------|
| `rag_answer(query, top_k=10)` | 主 RAG 接口 |
| `extract_body_and_refs(ans)` | 解析结构化输出，提取正文和引用 |
| `renumber_citations(citations)` | 重新编号引用 |

**数据流**:
```
用户查询 → FAISS检索 + BM25检索 → 结果去重合并 →
LLM生成答案 → 引用格式化 → 结构化输出
```

**响应格式**:
```
【正文】
答案内容...[1][2]

【参考文献】
[1] 《算法导论》第七章 快速排序
[2] 《算法详解》卷一 第三章
```

---

### 3. PEV Agent 框架 (`src/agent.py`)

**功能**: 高级 PEV (Planner-Executor-Verifier) 智能体框架，支持多步推理

#### 3.1 架构组件

**Planner Agent (规划器)**:
- 将复杂查询分解为子任务
- 选择合适的检索工具 (FAISS/BM25/知识图谱)
- 生成执行计划

**Executor Agent (执行器)**:
- 执行检索计划
- 综合多个检索结果
- 生成初步答案

**Verifier Agent (验证器)**:
- 评估答案质量
- 检查覆盖度和一致性
- 决定是否需要重新规划

#### 3.2 关键类和方法

```python
class PEVAgent:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.verifier = VerifierAgent()

    def run(self, query: str, max_iterations: int = 3):
        # 迭代规划和执行
        for iteration in range(max_iterations):
            plan = self.planner.plan(query)
            answer = self.executor.execute(plan)
            if self.verifier.verify(answer):
                break
            query = self.verifier.refine_query(query)
```

#### 3.3 特性
- 多迭代规划，支持重规划
- 结构化日志和错误处理
- 引用验证和重新编号
- 鲁棒的回退机制

---

### 4. 检索系统 (`src/retrieval/retrieval.py`)

**功能**: 双重检索引擎，结合向量和词汇搜索

#### 4.1 FAISS 检索
```python
def faiss_search(query: str, top_k: int = 10):
    # 1. 使用 SiliconFlow API 生成查询向量
    embedding = get_embedding_siliconflow(query)
    # 2. FAISS 向量相似度搜索
    distances, indices = faiss_index.search(embedding, top_k)
    # 3. 返回文档块及其元数据
    return format_results(distances, indices)
```

#### 4.2 BM25 检索
```python
def bm25_search(query: str, top_k: int = 10):
    # 1. jieba 分词
    tokens = jieba.lcut(query)
    # 2. BM25 评分
    scores = bm25_model.get_scores(tokens)
    # 3. 返回高分文档
    return top_documents(scores, top_k)
```

#### 4.3 先修知识检索
```python
PREDEFINED_CONCEPTS = [
    "时间复杂度", "空间复杂度", "数组", "链表",
    "栈", "队列", "树", "图", "哈希表", ...
]

def search_prerequisite_knowledge(concept: str):
    # 基于知识图谱获取概念的先修知识
    return knowledge_graph.get_prerequisites(concept)
```

---

### 5. 知识图谱 QA 系统 (`src/knowledge_graph_qa.py`)

**功能**: 基于实体关系图谱的问答系统

#### 5.1 支持的问题类型

| 问题类型 | 示例 | 处理方式 |
|----------|------|----------|
| 概念定义 | "什么是快速排序？" | 实体属性查询 |
| 方法说明 | "如何实现二叉树遍历？" | 关系路径遍历 |
| 属性查询 | "快速排序的时间复杂度是多少？" | has_attribute 关系 |
| 列举问题 | "有哪些排序算法？" | 类型关系遍历 |

#### 5.2 核心类
```python
class KnowledgeGraphQA:
    def __init__(self, graph_path: str):
        self.entity_relations = load_graph(graph_path)

    def answer(self, question: str) -> str:
        # 1. 关键词提取
        keywords = extract_keywords(question)
        # 2. 实体匹配
        entities = match_entities(keywords)
        # 3. 关系查询
        relations = query_relations(entities)
        # 4. 答案生成
        return generate_answer(relations)
```

---

### 6. 嵌入系统 (`src/embedder.py`)

**功能**: 文档分块和向量嵌入生成

#### 6.1 文档分块策略
```python
def chunk_document(content: str):
    chunks = []
    # 1. 章节级分割 (第N章)
    chapters = split_by_chapter(content)
    # 2. 标题层级分割 (1.2, 3.4.5)
    for chapter in chapters:
        sections = split_by_headings(chapter)
        # 3. 练习题特殊处理
        exercises = extract_exercises(sections)
        # 4. 维护逻辑路径
        chunks = build_logical_path(sections, exercises)
    return chunks
```

#### 6.2 块优化
- 最大块长度: 300 字符 (对于长块进行切分)
- 保持层级结构信息
- 维护逻辑路径 (章节标题路径)

---

### 7. 高级题库检索 (`src/retrieval/qa_retrieval/qa_retrieval_advanced.py`)

**功能**: 智能问答库搜索系统

#### 7.1 多维评分系统
```python
def calculate_score(query, qa_pair):
    score = (
        semantic_similarity(query, qa_pair.question) * 0.5 +  # 语义相似度
        keyword_match_score(query, qa_pair.question) * 0.3 +  # 关键词匹配
        answer_relevance(query, qa_pair.answer) * 0.2        # 答案相关性
    )
    return score
```

#### 7.2 搜索模式
- **keyword_only**: 仅关键词匹配
- **semantic_only**: 仅语义相似度 (TF-IDF)
- **hybrid**: 混合模式 (默认)

---

### 8. 工具函数 (`src/utils.py`)

**功能**: 引用管理和验证辅助函数

| 函数 | 功能 |
|------|------|
| `renumber_citations(citations)` | 将引用号转换为连续序列 |
| `validate_citations(text)` | 检查引用一致性 |
| `extract_citations_from_text(text)` | 从文本中解析引用 |

---

## 数据结构

### 目录结构

```
data/
├── faiss/
│   ├── faiss2.index                 # FAISS 向量索引 (47.4 MB)
│   └── refined_document_chunks.json # 文档块元数据 (15.5 MB)
│       ├── chunk_id                 # 块唯一标识
│       ├── content                  # 块内容
│       ├── metadata                 # 元数据 (书名、章节)
│       ├── path_titles              # 标题路径
│       └── logical_level            # 逻辑层级
│
├── knowledge_graph/
│   ├── test_new.json                # 主知识图谱 (11.4 MB)
│   ├── pre_knowledge_graph.json     # 先修知识关系 (160 KB)
│   ├── id2chunk.json                # 块映射 (784 KB)
│   └── knowledge_graph.json         # 图结构 (61 KB)
│
├── qa_bank/
│   ├── answered_questions.json      # 问答对 (359 KB)
│   └── 算法导论mini.json            # CLRS 习题 mini 版 (30 KB)
│
└── raw/                             # 源语料数据
```

### 文档块结构 (JSON)
```json
{
    "chunk_id": "algo_guide_ch7_sec1_001",
    "content": "快速排序是一种分治算法...",
    "metadata": {
        "book": "算法导论",
        "chapter": "第七章",
        "section": "7.1 快速排序的描述"
    },
    "path_titles": ["第七章", "快速排序", "7.1 快速排序的描述"],
    "logical_level": 3
}
```

### 知识图谱结构 (JSON)
```json
{
    "entity": "快速排序",
    "relations": [
        {"type": "is_a", "target": "排序算法"},
        {"type": "has_attribute", "target": "时间复杂度: O(n log n)"},
        {"type": "has_prerequisite", "target": "分治策略"},
        {"type": "has_component", "target": "分区操作"}
    ]
}
```

---

## 配置系统 (`config/base2.yaml`)

### LLM 提供商配置
```yaml
llm:
  deepseek:
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com"

  siliconflow:
    model: "deepseek-ai/DeepSeek-V3"
    base_url: "https://api.siliconflow.cn"

  dmx_api:
    models:
      - "gpt-4o-mini"
      - "gemini-2.5-flash"

  ollama:
    model: "deepseek-r1:latest"
    base_url: "http://localhost:11434"
```

### 嵌入提供商配置
```yaml
embedding:
  siliconflow:
    model: "Pro/BAAI/bge-m3"
    dimension: 1024

  openai:
    model: "text-embedding-3-small"
    dimension: 1536

  tongyi:
    model: "text-embedding-v2"
    dimension: 1536
```

---

## 关键算法

### 1. 双重检索融合算法
```python
def dual_retrieval(query, top_k=10):
    # 并行检索
    faiss_results = faiss_search(query, top_k)
    bm25_results = bm25_search(query, top_k)

    # 按 chunk_id 去重合并
    merged = {}
    for result in faiss_results + bm25_results:
        chunk_id = result['chunk_id']
        if chunk_id not in merged:
            merged[chunk_id] = result
        else:
            # 合并分数
            merged[chunk_id]['score'] = max(merged[chunk_id]['score'], result['score'])

    # 返回 top-k 结果
    return sorted(merged.values(), key=lambda x: x['score'], reverse=True)[:top_k]
```

### 2. 文档分块算法
```python
def intelligent_chunking(content):
    chunks = []
    # 按章节分割
    chapters = re.split(r'第[一二三四五六七八九十\d]+章', content)

    for chapter in chapters:
        # 按标题层级分割
        sections = re.split(r'^\d+\.\d+', chapter, flags=re.MULTILINE)

        for section in sections:
            # 长块切分
            if len(section) > 300:
                sub_chunks = split_long_chunk(section, max_length=300)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)

    return chunks
```

### 3. 引用重编号算法
```python
def renumber_citations(citations):
    # 去重
    unique_refs = {}
    for idx, ref in enumerate(citations):
        ref_key = (ref['book'], ref['chapter'])
        if ref_key not in unique_refs:
            unique_refs[ref_key] = len(unique_refs) + 1

    # 重新映射
    renumbered = []
    for idx, ref in enumerate(citations):
        ref_key = (ref['book'], ref['chapter'])
        new_idx = unique_refs[ref_key]
        renumbered.append({**ref, 'index': new_idx})

    return renumbered
```

---

## 开发指南

### 环境搭建
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务
uvicorn api_server:app --host 0.0.0.0 --port 8001

# 或使用 app/main.py
python app/main.py
```

### CLI 测试
```bash
# 测试主 RAG 系统
python src/main.py

# 测试知识图谱 QA
python src/knowledge_graph_qa.py

# 测试 PEV Agent
python src/agent.py
```

### 测试文件
- `test/test_agent.py`: PEV 框架测试
- `test/xdy_test*.py`: 各种系统测试
- `script/*.ipynb`: 组件测试 Jupyter 笔记本

---

## 依赖项

### 核心依赖
| 包 | 版本 | 用途 |
|----|------|------|
| fastapi | - | Web 框架 |
| uvicorn | - | ASGI 服务器 |
| faiss-cpu | - | 向量相似度搜索 |
| camel-ai | - | LLM 编排 |
| jieba | - | 中文分词 |
| pyyaml | - | 配置管理 |
| numpy | - | 数值计算 |
| requests | - | HTTP 客户端 |

### AI/ML 栈
| 类型 | 提供商 |
|------|--------|
| LLM | DeepSeek, SiliconFlow, OpenAI, Gemini |
| 嵌入 | SiliconFlow BGE-M3, OpenAI, 通义千问 |
| 向量存储 | FAISS |

---

## 系统能力总结

| 能力 | 描述 |
|------|------|
| 算法问答 | 专门针对算法和数据结构问题优化 |
| 多模态搜索 | 结合语义和词汇搜索 |
| 知识感知 | 理解先修知识关系 |
| 引用规范 | 提供带规范引用的结构化答案 |
| 可扩展性 | 模块化设计，易于扩展 |
| 多语言 | 优化中文教育内容，支持英文 |

---

---

## 智能推荐系统 (`controller.py`)

### 概述

智能推荐系统已实现，通过 `src/controller.py` 提供基于 LLM 的推荐判断功能。系统会根据用户查询自动决定是否需要推荐前置知识点和题库，无需用户手动选择。

### 架构流程

```
用户 Query
    │
    ▼
┌─────────────────────────────────────────┐
│      recommend_controller()              │
│      智能推荐判断器 (LLM 驱动)            │
│  分析 query，决定返回哪些模块             │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    返回内容决策                          │
├─────────────────┬─────────────────┬─────────────────────┤
│   场景 A         │   场景 B         │    场景 C           │
│  仅返回答案      │  答案 + 前置知识  │ 答案 + 前置知识 + 题库│
└─────────────────┴─────────────────┴─────────────────────┘
```

### 核心函数

#### 1. `recommend_controller(user_query: str) -> Dict[str, Any]`

**功能**: 智能推荐判断器，使用 LLM 分析用户查询

**返回值**:
```python
{
    "recommend_prerequisite": bool,  # 是否推荐前置知识点
    "recommend_qa_bank": bool,       # 是否推荐题库
    "detected_concepts": List[str],  # 检测到的概念列表
    "reason": str                    # 决策原因
}
```

**判断逻辑** (通过 LLM Prompt):
- **前置知识点推荐**: 用户询问算法/数据结构概念，且这些概念有依赖的基础知识
  - 例如: "什么是快速排序？" → 推荐"分治策略"、"递归"等
- **题库推荐**: 用户问题是实践导向的或明确表示想练习
  - 例如: "如何实现快速排序？" → 推荐练习题

#### 2. `format_prerequisite_results(raw_results: List) -> List[Dict]`

**功能**: 格式化前置知识点检索结果

**参数**: `pre_knowledge_search` 的原始结果 `[(concept, chunk), ...]`

**返回**: 格式化的前置知识点列表（最多 5 个，每条限制 500 字符）

```python
[
    {
        "concept": "分治策略",
        "content": "..."
    },
    ...
]
```

#### 3. `format_qa_results(raw_results: List) -> List[Dict]`

**功能**: 格式化题库检索结果

**参数**: `qa_retriever.search` 的原始结果 `[(item, score), ...]`

**返回**: 格式化的题库列表（最多 3 个，答案限制 800 字符）

```python
[
    {
        "question": "...",
        "answer": "...",
        "chapter": "...",
        "score": 0.92
    },
    ...
]
```

### API 端点: `/smart_answer`

**位置**: `src/api_server.py:121`

**请求**:
```json
{
    "query": "什么是快速排序？",
    "top_k": 10
}
```

**响应**:
```json
{
    "answer": "快速排序是一种分治算法...",
    "citations": ["《算法导论》第七章"],
    "decision": {
        "recommend_prerequisite": true,
        "recommend_qa_bank": false,
        "detected_concepts": ["快速排序"],
        "reason": "用户询问算法概念定义，适合推荐前置知识"
    },
    "prerequisites": [
        {"concept": "分治策略", "content": "..."},
        {"concept": "递归", "content": "..."}
    ]
}
```

### LLM Prompt 设计

系统使用精心设计的 Prompt 引导 LLM 输出严格 JSON：

```python
RECOMMEND_PROMPT = """你是一个智能教育助手的推荐判断器...

输出格式必须是严格的 JSON（不要有任何其他文字）：

{
  "recommend_prerequisite": true/false,
  "recommend_qa_bank": true/false,
  "detected_concepts": ["概念1", "概念2"],
  "reason": "简要说明决策原因"
}
...
"""
```

### 错误处理

- LLM 调用失败时返回默认值（全部不推荐）
- JSON 解析失败时有降级处理
- 决策原因字段包含错误信息（供调试）

### 决策矩阵

| recommend_prerequisite | recommend_qa_bank | 返回内容 |
|:---------------------:|:----------------:|----------|
|         False         |      False       | `answer` + `citations` |
|         True          |      False       | `answer` + `citations` + `prerequisites` |
|         False         |      True        | `answer` + `citations` + `related_questions` |
|         True          |      True        | `answer` + `citations` + `prerequisites` + `related_questions` |

### 配置

- **LLM 配置**: 使用 `config/base2.yaml` 中定义的 LLM
- **Temperature**: 0.1 (低温度保证决策稳定性)
- **Max Tokens**: 512

### 模块依赖

```
controller.py
├── camel.agents.ChatAgent      # LLM 交互
├── camel.models.OpenAIModel    # 模型封装
├── config/base2.yaml           # 配置加载
└── (调用方)
    ├── agent.py                # agent_framework
    ├── retrieval/retrieval.py  # pre_knowledge_search
    └── qa_retrieval/           # qa_retriever.search
```

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| 1.0 | 初始版本 | 基础 RAG 系统 |
| 1.1 | - | 添加知识图谱支持 |
| 1.2 | - | PEV Agent 框架 |
| 1.3 | - | 高级题库检索 |
| 1.4 | 2025-12-30 | 智能推荐系统 (`controller.py` + `/smart_answer`) |

---

*文档更新时间: 2025-12-30*
*项目路径: /home/guoziyang/AIgorithm_Agent*
