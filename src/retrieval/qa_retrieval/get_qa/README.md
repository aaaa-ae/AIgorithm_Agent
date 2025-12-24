# 算法题目抽取和重写系统

## 概述

本系统从教材的 chunk 化 JSON 数据中，自动抽取出所有"算法题目"，并将这些题目重写为"信息完整、无需依赖上下文即可独立解答的标准算法题"。

## 核心功能

### 四阶段处理流程

1. **题目检测**：判断 chunk 是否可能包含题目
2. **题目分类**：判断是"单题"、"多题列表"还是"不含题目"
3. **原子题目抽取**：从 chunk 中抽取出"原子题目列表"
4. **题目重写**：对每一道原子题目进行重写，使其成为可独立解答的完整题目

## 文件说明

### 主要文件

- `algorithm_question_extractor.py` - 核心抽取和重写系统
- `analyze_questions.py` - 题目质量分析和过滤工具
- `debug_extractor.py` - 调试和测试脚本
- `README.md` - 本说明文档

### 输入输出

- **输入文件**：`/home/guoziyang/AIgorithm_Agent/data/faiss/refined_document_chunks.json`
- **输出文件**：
  - `extracted_algorithm_questions.json` - 所有抽取的题目
  - `quality_algorithm_questions.json` - 过滤后的高质量题目

## 使用方法

### 1. 运行完整的抽取流程

```bash
cd /home/guoziyang/AIgorithm_Agent/src/retrieval/qa_retrieval/get_qa
python3 algorithm_question_extractor.py
```

### 2. 分析题目质量

```bash
python3 analyze_questions.py
```

### 3. 调试特定chunk

```bash
python3 debug_extractor.py
```

### 4. 自定义使用

```python
from algorithm_question_extractor import AlgorithmQuestionExtractor

# 创建抽取器
extractor = AlgorithmQuestionExtractor(
    data_file="path/to/refined_document_chunks.json",
    output_file="path/to/output.json"
)

# 运行处理
stats = extractor.run(rewrite_questions=True)

# 预览结果
extractor.preview_results(num_samples=10)
```

## 处理结果统计

根据运行结果：

- 总处理 chunks：11,587 个
- 检测到包含题目的 chunks：4,261 个
- 抽取到的题目总数：3,098 道
  - 单题模式：1,872 道
  - 多题模式：747 道
- 质量过滤后保留：1,562 道（50.4%）

## 题目数据结构

```json
{
  "question_id": "唯一标识符",
  "title": "题目标题/编号",
  "description": "题目描述",
  "context": "上下文信息",
  "source": "来源信息",
  "rewrite_prompt": "重写提示",
  "metadata": {
    "chunk_id": "来源chunk ID",
    "classification": "single/multiple",
    "rewritten": "是否已重写"
  }
}
```

## 题目重写模板

重写后的题目包含：

1. **背景信息**：算法背景、相关定义
2. **题目描述**：完整的题目要求
3. **输入格式**：具体的输入格式说明
4. **输出格式**：具体的输出格式说明
5. **约束条件**：时间空间约束等
6. **示例**：输入输出示例（如果有）

## 质量控制

### 过滤标准

- 描述长度 ≥ 50 字符
- 不包含"答案"、"提示"、"参考"等非题目关键词
- 标题不为纯数字
- 有实际的题目描述内容

### 常见问题

1. **误判为题目**：包含"练习"关键词但实际不是题目
2. **题目描述不完整**：缺少关键信息，依赖上下文
3. **格式问题**：包含大量HTML标签或格式符号

## 调试技巧

### 1. 查看特定chunk的处理过程

```python
debugger = DebugExtractor("data_file.json")
debugger.debug_chunk(chunk_id=2029)  # 调试特定chunk
```

### 2. 查找包含特定模式的chunks

```python
# 查找包含"练习"的chunks
chunks = debugger.find_example_chunks("练习", max_results=5)

# 查找包含具体题号的chunks
chunks = debugger.find_example_chunks("18.2-1", max_results=3)
```

### 3. 批量测试样本

```python
test_chunk_ids = [2029, 2030, 2031, 2032]
debugger.test_extraction_on_samples(test_chunk_ids)
```

## 扩展功能

### 1. 集成LLM进行智能重写

当前系统使用模板进行基础重写，可以扩展为调用LLM API：

```python
def _llm_rewrite(self, question: Question) -> str:
    prompt = self._build_rewrite_prompt(chunk, title, description)
    # 调用LLM API进行智能重写
    response = llm_client.generate(prompt)
    return response
```

### 2. 添加题目难度分类

可以根据题目内容自动判断难度等级：

```python
def classify_difficulty(self, question: Question) -> str:
    # 简单、中等、困难
    pass
```

### 3. 添加知识点标签

为每道题自动添加算法类型标签：

```python
def extract_algorithm_tags(self, question: Question) -> List[str]:
    # 动态规划、图算法、排序等
    pass
```

## 注意事项

1. **编码问题**：确保所有文件使用UTF-8编码
2. **内存使用**：处理大量chunks时注意内存占用
3. **正则表达式**：题目编号模式可能需要根据具体数据调整
4. **质量标准**：过滤标准可以根据实际需求调整

## 性能优化建议

1. **批处理**：可以分批处理chunks，避免内存溢出
2. **缓存机制**：对重复处理的结果进行缓存
3. **并行处理**：使用多进程加速处理大量chunks
4. **增量更新**：只处理新增或修改的chunks

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue到项目仓库
- 发送邮件到开发团队
- 参与项目讨论群