#!/usr/bin/env python3
"""
运行 agent 框架测试的便捷脚本
"""

import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import agent_framework

def test_agent_framework():
    """测试 agent 框架"""
    # Test query
    test_query = "什么是最优先子结构？"

    print("=" * 60)
    print("PEV Agent Framework Test")
    print("=" * 60)
    print(f"测试问题：{test_query}\n")

    # Run the PEV framework
    result = agent_framework(test_query)

    # Display results
    print("\n" + "=" * 60)
    print("PEV Agent Response:")
    print("=" * 60)
    print(result["final_answer"])
    print("\n迭代次数:", result["iterations"])
    print("验证结果:", result.get("verification", {}))

if __name__ == "__main__":
    test_agent_framework()