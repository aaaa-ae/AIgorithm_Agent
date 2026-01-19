import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import existing modules
from retrieval.retrieval import faiss_search, bm25_search, pre_knowledge_search
from retrieval.qa_retrieval.qa_retrieval_advanced import (
    AdvancedQARetriever,
    AdvancedRetrievalConfig,
)

from camel.agents import ChatAgent
from camel.models import OpenAIModel
from utils import renumber_citations, validate_citations, renumber_citations_both_formats

# 初始化题库检索器
qa_retriever = AdvancedQARetriever(
    AdvancedRetrievalConfig(search_mode="hybrid")
)


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "base2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# Initialize LLM client
CONFIG = load_config()
LLM_PROVIDER = CONFIG["llm"][CONFIG["llm"]["use"]]

def create_llm_client():
    """Create LLM client using existing pattern"""
    model_name = LLM_PROVIDER["model"]
    api_key = LLM_PROVIDER["api_key"]
    api_base = LLM_PROVIDER["api_base"].rstrip("/")

    model_config = {
        "temperature": LLM_PROVIDER.get("temperature"),
        "top_p": LLM_PROVIDER.get("top_p"),
        "max_tokens": LLM_PROVIDER.get("max_tokens")
    }
    model_config = {k: v for k, v in model_config.items() if v is not None}

    model = OpenAIModel(
        model_type=model_name,
        model_config_dict=model_config,
        api_key=api_key,
        url=api_base
    )
    return model

LLM_CLIENT = create_llm_client()


# ================================
# Agent Prompts
# ================================

PLANNER_PROMPT = """你是一个专业的问答任务规划助手。你的职责是分析用户问题，制定详细的执行计划。

你必须：
1. 将复杂问题分解为具体的子任务
2. 为每个子任务选择合适的检索工具（faiss_search 或 bm25_search）
3. 制定验证策略

输出格式必须是严格的 JSON：

{
  "task_decomposition": [
    "子任务1：明确需要查找的信息",
    "子任务2：补充相关背景知识",
    "子任务3：整合答案"
  ],
  "executor_plan": [
    {
      "step_id": 1,
      "action": "faiss_search",
      "query": "具体的检索查询词",
      "top_k": 10
    }
  ],
  "verifier_plan": {
    "check_coverage": true,
    "check_citation": true,
    "check_consistency": true
  }
}

注意：
- action 只能是 "faiss_search", "bm25_search" 或 "synthesize_answer"
- 最后一步必须包含 "synthesize_answer"
- 确保查询词准确反映子任务需求
"""

EXECUTOR_SYNTHESIZE_PROMPT = """你是一个专业的答案合成助手。请基于检索到的证据生成准确、有用的答案。

要求：
1. 严格基于提供的 observations 生成答案
2. 不得添加任何未在 observations 中的信息
3. 保持逻辑清晰，层次分明
4. 在适当位置标注引用编号 [1], [2] 等,不要都放在最后

输出格式：
{
  "answer": "生成的答案内容，包含引用标注",
  "citations": [1, 2, 3]
}

Observations：
{observations}
"""

VERIFIER_PROMPT = """你是一个专业的答案质量验证助手。请评估答案是否满足用户需求。

评估维度：
1. 充分性：是否完全回答了用户问题
2. 准确性：内容是否基于检索到的证据
3. 引用正确性：引用是否准确对应
4. 一致性：答案内部是否自洽

输出严格 JSON：
{
  "is_sufficient": true/false,
  "missing_aspects": ["缺少的方面1", "缺少的方面2"],
  "has_hallucination": true/false,
  "has_citation_error": true/false,
  "next_action": "accept" 或 "replan"
}

用户问题：{user_query}
候选答案：{final_answer}
使用引用：{used_citations}
检索证据：{observations}
"""


# ================================
# Logging Utility
# ================================

def log_agent(agent_name: str, message: str, data: Any = None):
    """Print structured logging for debugging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{agent_name}] {message}")
    if data is not None:
        if isinstance(data, dict):
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(data)


# ================================
# Core Agent Functions
# ================================

def run_planner(user_query: str) -> Dict[str, Any]:
    """
    Run the Planner agent to decompose the query and create execution plan

    Args:
        user_query: Original user query

    Returns:
        Planner's output as dictionary with task_decomposition, executor_plan, verifier_plan
    """
    log_agent("PLANNER", "Starting query planning", f"Query: {user_query}")

    try:
        agent = ChatAgent(PLANNER_PROMPT, model=LLM_CLIENT)

        user_prompt = f"用户问题：{user_query}\n\n请生成执行计划："
        response = agent.step(user_prompt)

        # Parse JSON response
        result = json.loads(response.msg.content)
        log_agent("PLANNER", "Planning completed", result)
        return result

    except Exception as e:
        log_agent("PLANNER", f"Error occurred: {str(e)}")
        # Fallback plan
        return {
            "task_decomposition": ["直接检索答案"],
            "executor_plan": [
                {"step_id": 1, "action": "faiss_search", "query": user_query, "top_k": 10},
                {"step_id": 2, "action": "bm25_search", "query": user_query, "top_k": 10},
                {"step_id": 3, "action": "synthesize_answer"}
            ],
            "verifier_plan": {
                "check_coverage": True,
                "check_citation": True,
                "check_consistency": True
            }
        }


def run_executor(executor_plan: List[Dict], user_query: str = "") -> Dict[str, Any]:
    """
    Execute the plan generated by the Planner

    Args:
        executor_plan: List of execution steps
        user_query: Original user query (for fallback LLM answer)

    Returns:
        Dictionary with observations, final_answer, and used_citations
    """
    log_agent("EXECUTOR", "Starting execution", f"Steps: {len(executor_plan)}")

    observations = []
    all_docs = []
    final_answer = ""
    used_citations = []

    try:
        for step in executor_plan:
            step_id = step.get("step_id", 0)
            action = step.get("action", "")

            if action == "faiss_search":
                query = step.get("query", "")
                top_k = step.get("top_k", 10)
                docs = faiss_search(query, top_k)

                observation = {
                    "step_id": step_id,
                    "action": action,
                    "query": query,
                    "results": docs
                }
                observations.append(observation)
                all_docs.extend(docs)

                log_agent("EXECUTOR", f"Step {step_id}: FAISS search", f"Query: {query}, Results: {len(docs)}")

            elif action == "bm25_search":
                query = step.get("query", "")
                top_k = step.get("top_k", 10)
                docs = bm25_search(query, top_k)

                observation = {
                    "step_id": step_id,
                    "action": action,
                    "query": query,
                    "results": docs
                }
                observations.append(observation)
                all_docs.extend(docs)

                log_agent("EXECUTOR", f"Step {step_id}: BM25 search", f"Query: {query}, Results: {len(docs)}")

            elif action == "synthesize_answer":
                # Synthesize answer from observations
                log_agent("EXECUTOR", "Step {step_id}: Synthesizing answer")

                # Deduplicate documents by chunk_id
                unique_docs = {}
                for doc in all_docs:
                    chunk_id = doc.get("chunk_id")
                    if chunk_id not in unique_docs:
                        unique_docs[chunk_id] = doc

                # Format observations for LLM
                formatted_observations = []
                for i, (cid, doc) in enumerate(unique_docs.items(), 1):
                    obs_str = f"[{i}] 标题：{doc.get('title', '')}\n"
                    obs_str += f"章节：{doc.get('path_titles', [])}\n"
                    obs_str += f"内容：{doc.get('content', '')}"
                    formatted_observations.append(obs_str)

                observations_text = "\n\n".join(formatted_observations)

                # Call LLM to synthesize
                agent = ChatAgent(EXECUTOR_SYNTHESIZE_PROMPT, model=LLM_CLIENT)
                user_prompt = f"Observations：\n{observations_text}"
                response = agent.step(user_prompt)

                # Parse synthesis result
                synthesis_result = json.loads(response.msg.content)
                final_answer = synthesis_result.get("answer", "")
                raw_citations = synthesis_result.get("citations", list(range(1, len(unique_docs) + 1)))

                # 重新编号引用，确保从1开始连续
                final_answer, used_citations = renumber_citations(final_answer, raw_citations)

                log_agent("EXECUTOR", "Answer synthesized", f"Raw citations: {raw_citations}")
                log_agent("EXECUTOR", "Renumbered citations", f"New citations: {used_citations}")

        return {
            "observations": observations,
            "final_answer": final_answer,
            "used_citations": used_citations
        }

    except Exception as e:
        log_agent("EXECUTOR", f"Error occurred: {str(e)}")

        # Fallback: 直接调用 LLM 回答问题
        if user_query:
            try:
                log_agent("EXECUTOR", "Using fallback LLM to answer directly")
                fallback_prompt = f"""请直接回答以下问题。如果不确定，请诚实地说明。

问题：{user_query}

请提供准确、有用的答案。"""
                agent = ChatAgent("你是一个专业的问答助手。", model=LLM_CLIENT)
                response = agent.step(fallback_prompt)
                fallback_answer = response.msg.content

                return {
                    "observations": observations,
                    "final_answer": fallback_answer,
                    "used_citations": [],
                    "fallback": True
                }
            except Exception as fallback_error:
                log_agent("EXECUTOR", f"Fallback LLM also failed: {str(fallback_error)}")

        return {
            "observations": observations,
            "final_answer": "抱歉，生成答案时遇到错误。",
            "used_citations": []
        }


def run_verifier(user_query: str, final_answer: str, used_citations: List[int],
                 observations: List[Dict]) -> Dict[str, Any]:
    """
    Verify the quality and completeness of the generated answer

    Args:
        user_query: Original user query
        final_answer: Generated answer from executor
        used_citations: List of citation indices used
        observations: Raw observations from executor

    Returns:
        Verification result dictionary
    """
    log_agent("VERIFIER", "Starting verification")

    try:
        agent = ChatAgent(VERIFIER_PROMPT, model=LLM_CLIENT)

        # Format observations for verification
        obs_summary = []
        for obs in observations:
            if obs.get("action") in ["faiss_search", "bm25_search"]:
                obs_summary.append({
                    "query": obs.get("query", ""),
                    "result_count": len(obs.get("results", []))
                })

        user_prompt = VERIFIER_PROMPT.format(
            user_query=user_query,
            final_answer=final_answer,
            used_citations=used_citations,
            observations=json.dumps(obs_summary, ensure_ascii=False)
        )

        response = agent.step(user_prompt)
        result = json.loads(response.msg.content)

        log_agent("VERIFIER", "Verification completed", result)
        return result

    except Exception as e:
        log_agent("VERIFIER", f"Error occurred: {str(e)}")
        # Fallback to accept
        return {
            "is_sufficient": True,
            "missing_aspects": [],
            "has_hallucination": False,
            "has_citation_error": False,
            "next_action": "accept"
        }


def agent_framework(user_query: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Main controller that orchestrates the PEV framework

    Args:
        user_query: User's input query
        max_iterations: Maximum number of replanning cycles (default: 3)

    Returns:
        Final answer and metadata
    """
    log_agent("CONTROLLER", "Starting PEV framework", f"Query: {user_query}")

    iteration = 0
    missing_aspects = []

    while iteration < max_iterations:
        iteration += 1
        log_agent("CONTROLLER", f"Iteration {iteration}")

        # 1. Planning phase
        if iteration == 1:
            plan_input = user_query
        else:
            # Add missing aspects to query
            plan_input = f"{user_query}\n\n需要补充的方面：{', '.join(missing_aspects)}"

        planner_result = run_planner(plan_input)
        executor_plan = planner_result.get("executor_plan", [])

        # 2. Execution phase
        executor_result = run_executor(executor_plan, user_query)
        final_answer = executor_result.get("final_answer", "")
        used_citations = executor_result.get("used_citations", [])
        observations = executor_result.get("observations", [])

        # 3. Verification phase
        verifier_result = run_verifier(
            user_query, final_answer, used_citations, observations
        )

        # 4. Decision
        next_action = verifier_result.get("next_action", "accept")

        if next_action == "accept":
            log_agent("CONTROLLER", "Answer accepted", f"Total iterations: {iteration}")
            return {
                "final_answer": final_answer,
                "used_citations": used_citations,
                "iterations": iteration,
                "verification": verifier_result,
                "observations": observations
            }
        else:
            missing_aspects = verifier_result.get("missing_aspects", [])
            log_agent("CONTROLLER", "Replanning needed", f"Missing: {missing_aspects}")

            if iteration >= max_iterations:
                log_agent("CONTROLLER", "Max iterations reached, returning current answer")
                return {
                    "final_answer": final_answer + "\n\n注意：可能存在以下未充分覆盖的方面：" +
                                   ", ".join(missing_aspects),
                    "used_citations": used_citations,
                    "iterations": iteration,
                    "verification": verifier_result,
                    "observations": observations
                }

    return {
        "final_answer": "未找到有效答案",
        "used_citations": [],
        "iterations": iteration,
        "verification": {},
        "observations": []
    }


# ================================
# CLI Interface
# ================================
def build_citation_display_index(response: Dict[str, Any]) -> Dict[int, str]:
    """
    citation_id -> "《书名》 · title"
    citation_id 按 observations.results 首次出现的 chunk 去重顺序递增
    """
    idx: Dict[int, str] = {}
    seen_chunk_ids = set()
    citation_id = 1

    for obs in response.get("observations", []):
        for r in obs.get("results", []):
            chunk_id = r.get("chunk_id")
            if chunk_id is None or chunk_id in seen_chunk_ids:
                continue

            meta = r.get("metadata", {}) or {}
            book = meta.get("title") or meta.get("source") or "未知来源"
            title = r.get("title") or "未命名条目"

            idx[citation_id] = f"《{book}》-{title}"

            seen_chunk_ids.add(chunk_id)
            citation_id += 1

    return idx


def format_response(response: Dict[str, Any]) -> Tuple[str, List[str]]:
    answer = response.get("final_answer", "")
    used = response.get("used_citations", [])

    # 1. 重新编号 answer 中的引用（支持 [] 和 【】）
    answer, old_to_new_mapping = renumber_citations_both_formats(answer, used)

    # 2. 构建原始编号 -> 文献信息的映射
    display_index = build_citation_display_index(response)

    # 3. 按 used 的顺序获取文献信息
    citations = []
    for old_idx in sorted(display_index.keys()):
        if old_idx in old_to_new_mapping:
            citations.append(display_index[old_idx])

    return answer, citations


def main():
    """Command-line interface for the PEV agent"""
    print("输入您的问题，输入 'exit' 或 'quit' 退出\n")

    while True:
        try:
            query = input("\n请输入您的问题：\n> ").strip()

            if query.lower() in ["exit", "quit", "退出", "q"]:
                print("\n感谢使用！")
                break

            if not query:
                continue

            print("\n正在处理您的问题...")
            print("-" * 40)

            # Run the PEV framework
            response = agent_framework(query)

            answer, citations = format_response(response)
            print("answer:", answer)
            print("citations:", citations)
            # Display result
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n感谢使用！")
            break
        except Exception as e:
            print(f"\n处理出错：{str(e)}")
            print("请重试或联系系统管理员")


if __name__ == "__main__":
    main()