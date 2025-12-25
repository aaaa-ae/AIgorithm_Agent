from typing import List, Tuple, Dict, Any
import re
import json


def renumber_citations(final_answer: str, used_citations: List[int]) -> Tuple[str, List[int]]:
    """
    重新编号引用，确保引用编号从1开始连续递增

    Args:
        final_answer: 包含原始引用编号的答案
        used_citations: 使用的引用编号列表（可能不连续）

    Returns:
        Tuple: (重新编号后的答案, 重新编号后的引用列表)
    """
    if not used_citations:
        return final_answer, []

    # 创建映射：原始编号 -> 新编号（从1开始连续）
    original_to_new = {}
    used_sorted = sorted(set(used_citations))  # 去重并排序

    for new_num, original_num in enumerate(used_sorted, 1):
        original_to_new[original_num] = new_num

    # 使用正则表达式更精确地替换引用编号
    # 确保只替换独立的 [数字] 格式
    new_answer = final_answer
    for original_num, new_num in original_to_new.items():
        # 使用正则表达式确保只匹配完整的引用标记
        pattern = r'\[' + str(original_num) + r'\](?!\d)'  # 避免匹配 [10] 时误改 [1]
        new_answer = re.sub(pattern, f'[{new_num}]', new_answer)

    # 生成新的引用列表（连续的 1, 2, 3...）
    new_citations = list(range(1, len(used_sorted) + 1))

    return new_answer, new_citations


def extract_citations_from_text(text: str) -> List[int]:
    """
    从文本中提取所有引用编号

    Args:
        text: 包含引用的文本

    Returns:
        提取到的引用编号列表
    """
    # 使用正则表达式查找所有 [数字] 格式的引用
    citations = re.findall(r'\[(\d+)\]', text)
    # 转换为整数并去重
    return sorted(list(set(int(c) for c in citations)))


def validate_citations(final_answer: str, used_citations: List[int]) -> Dict[str, Any]:
    """
    验证答案中的引用是否正确

    Args:
        final_answer: 生成的答案
        used_citations: 声明使用的引用列表

    Returns:
        验证结果字典
    """
    # 从答案中实际提取的引用
    actual_citations = extract_citations_from_text(final_answer)

    # 找出差异
    missing_in_answer = set(used_citations) - set(actual_citations)
    extra_in_answer = set(actual_citations) - set(used_citations)

    return {
        "is_valid": len(missing_in_answer) == 0 and len(extra_in_answer) == 0,
        "actual_citations": actual_citations,
        "declared_citations": used_citations,
        "missing_in_answer": list(missing_in_answer),
        "extra_in_answer": list(extra_in_answer)
    }


def renumber_citations_both_formats(answer: str, used_citations: List[int]) -> Tuple[str, Dict[int, int]]:
    """
    重新编号 answer 中的引用，支持 [] 和 【】 两种格式

    Args:
        answer: 包含引用的答案文本
        used_citations: 使用的旧引用编号列表

    Returns:
        Tuple: (重新编号后的答案, 旧编号->新编号 的映射)
    """
    if not used_citations:
        return answer, {}

    # 创建映射：旧编号 -> 新编号（从1开始连续）
    old_to_new = {}
    used_sorted = sorted(set(used_citations))
    for new_num, old_num in enumerate(used_sorted, 1):
        old_to_new[old_num] = new_num

    # 替换 [] 格式的引用
    new_answer = answer
    for old_num, new_num in old_to_new.items():
        pattern = r'\[' + str(old_num) + r'\](?!\d)'
        new_answer = re.sub(pattern, f'[{new_num}]', new_answer)

    # 替换 【】 格式的引用
    for old_num, new_num in old_to_new.items():
        pattern = r'【' + str(old_num) + r'】(?!\d)'
        new_answer = re.sub(pattern, f'【{new_num}】', new_answer)

    return new_answer, old_to_new