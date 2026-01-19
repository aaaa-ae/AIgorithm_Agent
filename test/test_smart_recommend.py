# -*- coding: utf-8 -*-
"""
测试 api_smart_answer 函数逻辑
"""
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_api_smart_answer():
    """
    测试 api_smart_answer 函数逻辑

    模拟 api_server.py 中的 api_smart_answer 函数的执行流程
    """
    from agent import agent_framework, format_response
    from retrieval.retrieval import pre_knowledge_search
    from retrieval.qa_retrieval.qa_retrieval_advanced import (
        AdvancedQARetriever,
        AdvancedRetrievalConfig,
    )
    from controller import (
        recommend_controller,
        format_prerequisite_results,
        format_qa_results,
    )

    # 初始化 QA retriever
    qa_retriever = AdvancedQARetriever(
        AdvancedRetrievalConfig(
            search_mode="hybrid"
        )
    )

    # 测试查询
    test_queries = [
        "什么是快速排序？",
        "如何实现二叉树的遍历？",
        "数组和链表有什么区别？",
    ]

    print("=" * 60)
    print("api_smart_answer 函数逻辑测试")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {query}")
        print("=" * 60)

        # ---------- 模拟 api_smart_answer 的执行流程 ----------

        # 1. 推荐判断
        print("\n[步骤 1] 推荐判断...")
        decision = recommend_controller(query)
        print(f"  - 推荐前置知识: {decision['recommend_prerequisite']}")
        print(f"  - 推荐题库: {decision['recommend_qa_bank']}")
        print(f"  - 检测概念: {decision['detected_concepts']}")
        print(f"  - 原因: {decision['reason']}")

        # 2. 基础答案 (始终返回)
        print("\n[步骤 2] 生成基础答案...")
        response = agent_framework(query)
        answer, citations = format_response(response)
        print(f"  - 答案长度: {len(answer)} 字符")
        print(f"  - 答案预览: {answer[:100]}...")
        print(f"  - 引用数量: {len(citations)}")

        # 构建结果
        result = {
            "answer": answer,
            "citations": citations,
            "decision": {
                "recommend_prerequisite": decision["recommend_prerequisite"],
                "recommend_qa_bank": decision["recommend_qa_bank"],
                "detected_concepts": decision["detected_concepts"],
                "reason": decision["reason"]
            }
        }

        # 3. 根据决策添加额外内容
        if decision["recommend_prerequisite"]:
            print("\n[步骤 3a] 获取前置知识点...")
            prereq_raw = pre_knowledge_search(query)
            prerequisites = format_prerequisite_results(prereq_raw)
            result["prerequisites"] = prerequisites
            print(f"  - 前置知识点数量: {len(prerequisites)}")
            for j, prereq in enumerate(prerequisites[:3], 1):
                print(f"    {j}. {prereq.get('concept', 'N/A')}")
        else:
            print("\n[步骤 3a] 不推荐前置知识点，跳过")

        if decision["recommend_qa_bank"]:
            print("\n[步骤 3b] 获取相关题库...")
            qa_raw = qa_retriever.search(query, top_k=5)
            related_questions = format_qa_results(qa_raw)
            result["related_questions"] = related_questions
            print(f"  - 相关题目数量: {len(related_questions)}")
            for j, qa in enumerate(related_questions[:3], 1):
                print(f"    {j}. {qa.get('question', 'N/A')[:50]}...")
        else:
            print("\n[步骤 3b] 不推荐题库，跳过")

        # 输出最终结果结构
        print("\n[最终结果结构]")
        print(f"  - answer: {len(result.get('answer', ''))} 字符")
        print(f"  - citations: {len(result.get('citations', []))} 条")
        print(f"  - decision.recommend_prerequisite: {result['decision']['recommend_prerequisite']}")
        print(f"  - decision.recommend_qa_bank: {result['decision']['recommend_qa_bank']}")
        print(f"  - decision.detected_concepts: {result['decision']['detected_concepts']}")
        print(f"  - prerequisites: {len(result.get('prerequisites', []))} 条")
        print(f"  - related_questions: {len(result.get('related_questions', []))} 条")


if __name__ == "__main__":
    test_api_smart_answer()
