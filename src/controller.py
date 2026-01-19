"""
智能推荐控制器

基于 LLM 判断用户查询是否需要推荐前置知识点和题库
"""
import json
from typing import Dict, Any, List
import yaml
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from camel.agents import ChatAgent
from camel.models import OpenAIModel


def load_config():
    # 获取项目根目录 - 向上查找直到找到 config 目录
    current = Path(__file__).parent
    while current != current.parent:
        config_path = current / "config" / "base2.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        current = current.parent

    # 如果找不到，尝试从当前目录向上查找
    import os
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "base2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


def create_llm_client():
    """创建 LLM 客户端"""
    config = load_config()
    llm_provider = config["llm"][config["llm"]["use"]]

    model_name = llm_provider["model"]
    api_key = llm_provider["api_key"]
    api_base = llm_provider["api_base"].rstrip("/")

    model_config = {
        "temperature": 0.1,  # 推荐判断使用低温度
        "max_tokens": 512
    }

    model = OpenAIModel(
        model_type=model_name,
        model_config_dict=model_config,
        api_key=api_key,
        url=api_base
    )
    return model


# LLM 客户端
_LLM_CLIENT = None


def get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = create_llm_client()
    return _LLM_CLIENT


# 推荐 Prompt
RECOMMEND_PROMPT = """你是一个智能教育助手的推荐判断器。你的职责是分析用户问题，判断是否需要推荐额外的学习内容。

你需要判断两个方面：

1. **前置知识点推荐** (recommend_prerequisite)：
   - 如果用户询问的是一个具体的算法概念或数据结构，且这些概念有依赖的基础知识
   - 例如：用户问"什么是快速排序？"，可能需要推荐"分治策略"、"递归"等前置知识
   - 判断标准：问题是否涉及算法/数据结构的概念理解

2. **题库推荐** (recommend_qa_bank)：
   - 如果用户的问题类型是"如何实现"、"怎么写"、"怎么做"等实践导向的问题
   - 或者用户明确表示想要练习、做题
   - 判断标准：问题是否暗示用户想要动手实践或寻找练习题

输出格式必须是严格的 JSON（不要有任何其他文字）：

{
  "recommend_prerequisite": true/false,
  "recommend_qa_bank": true/false,
  "detected_concepts": ["概念1", "概念2"],
  "reason": "简要说明决策原因"
}

示例：

输入：什么是快速排序？
输出：
{
  "recommend_prerequisite": true,
  "recommend_qa_bank": false,
  "detected_concepts": ["快速排序"],
  "reason": "用户询问算法概念定义，适合推荐前置知识"
}

输入：如何实现快速排序？
输出：
{
  "recommend_prerequisite": true,
  "recommend_qa_bank": true,
  "detected_concepts": ["快速排序"],
  "reason": "用户询问实现方法，既需要前置知识，也适合推荐练习题"
}

输入：时间复杂度和空间复杂度有什么区别？
输出：
{
  "recommend_prerequisite": false,
  "recommend_qa_bank": false,
  "detected_concepts": ["时间复杂度", "空间复杂度"],
  "reason": "用户询问概念比较，无需额外推荐"
}

现在请分析以下用户问题：
"""


def recommend_controller(user_query: str) -> Dict[str, Any]:
    """
    智能推荐判断器

    Args:
        user_query: 用户查询字符串

    Returns:
        决策字典，包含：
        - recommend_prerequisite: 是否推荐前置知识点
        - recommend_qa_bank: 是否推荐题库
        - detected_concepts: 检测到的概念列表
        - reason: 决策原因
    """
    try:
        agent = ChatAgent(RECOMMEND_PROMPT, model=get_llm_client())

        response = agent.step(user_query)
        result = json.loads(response.msg.content)

        # 确保返回的字段完整
        return {
            "recommend_prerequisite": result.get("recommend_prerequisite", False),
            "recommend_qa_bank": result.get("recommend_qa_bank", False),
            "detected_concepts": result.get("detected_concepts", []),
            "reason": result.get("reason", "")
        }

    except Exception as e:
        # 出错时返回默认值（不推荐）
        print(f"[推荐判断出错] {str(e)}")
        return {
            "recommend_prerequisite": False,
            "recommend_qa_bank": False,
            "detected_concepts": [],
            "reason": f"判断出错: {str(e)}"
        }


def format_prerequisite_results(raw_results: List) -> List[Dict]:
    """
    格式化前置知识点检索结果

    Args:
        raw_results: pre_knowledge_search 的原始结果

    Returns:
        格式化后的前置知识点列表
    """
    prerequisites = []
    seen = set()

    for concept, chunk in raw_results:
        if concept in seen:
            continue
        seen.add(concept)

        # 限制返回数量（最多5个）
        if len(prerequisites) >= 5:
            break

        prerequisites.append({
            "concept": concept,
            "content": chunk.get("content", "")[:500]  # 限制内容长度
        })

    return prerequisites


def format_qa_results(raw_results: List) -> List[Dict]:
    """
    格式化题库检索结果

    Args:
        raw_results: qa_retriever.search 的原始结果 [(item, score), ...]

    Returns:
        格式化后的题库列表
    """
    formatted = []

    for item, score in raw_results:
        # 限制返回数量（最多3个）
        if len(formatted) >= 3:
            break

        formatted.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", "")[:800],  # 限制答案长度
            "chapter": item.get("chapter", ""),
            "score": round(score, 3)
        })

    return formatted
