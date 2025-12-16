#!/usr/bin/env python3
"""
修改算法导论题目JSON文件
删除id、type、full_text字段，只保留chapter和question
"""

import json

def modify_json():
    # 读取原文件
    with open('/home/chenyifan/get_QA/算法导论题目.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建新的数据结构
    modified_data = {
        "exercises": [],
        "problems": [],
        "statistics": data.get("statistics", {})
    }

    # 处理练习题
    for exercise in data.get("exercises", []):
        modified_exercise = {
            "chapter": exercise.get("chapter", ""),
            "question": exercise.get("question", "")
        }
        modified_data["exercises"].append(modified_exercise)

    # 处理思考题
    for problem in data.get("problems", []):
        modified_problem = {
            "chapter": problem.get("chapter", ""),
            "title": problem.get("title", ""),
            "content": problem.get("content", "")
        }
        modified_data["problems"].append(modified_problem)

    # 保存修改后的文件
    with open('/home/chenyifan/get_QA/算法导论题目.json', 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

    print(f"修改完成！")
    print(f"练习题数量: {len(modified_data['exercises'])}")
    print(f"思考题数量: {len(modified_data['problems'])}")

    # 显示前几个修改后的条目作为示例
    print("\n修改后的练习题示例：")
    for i, ex in enumerate(modified_data["exercises"][:3]):
        print(f"\n练习题 {i+1}:")
        print(f"  章节: {ex['chapter']}")
        print(f"  问题: {ex['question'][:100]}...")

if __name__ == "__main__":
    modify_json()