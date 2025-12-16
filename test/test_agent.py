#!/usr/bin/env python3
"""
Simple test script for the PEV Agent Framework
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import agent_framework

# Test query
test_query = "最优子结构是什么"
queries = ["", "" ,"",...]
results = []

for test_query in queries:
  result = agent_framework(test_query)
  res = result["final_answer"]
  results.append(agent_framework(test_query))



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