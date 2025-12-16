#!/usr/bin/env python3
"""
算法导论练习题和思考题提取器
从markdown文件中提取练习题和思考题，并保存为JSON格式
"""

import re
import json
from typing import List, Dict, Any

class ExerciseExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.exercises = []
        self.problems = []

    def load_file(self) -> str:
        """加载markdown文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"文件 {self.file_path} 不存在")
            return ""
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return ""

    def extract_exercises(self, content: str) -> List[Dict[str, Any]]:
        """提取练习题"""
        exercises = []

        # 匹配练习题模式，如 "1.1-1", "2.3-1" 等
        exercise_pattern = r'(\d+\.\d+-\d+)\s+([^.\n]*(?:\.[^.\n]*)*)?\.?\s*([^.\n]*(?:\.[^.\n]*)*)?\.?'

        # 也匹配带星号的练习题，如 "*4.3-5"
        exercise_pattern_star = r'\*(\d+\.\d+-\d+)\s+([^.\n]*(?:\.[^.\n]*)*)?\.?\s*([^.\n]*(?:\.[^.\n]*)*)?\.?'

        lines = content.split('\n')
        current_chapter = ""

        for line in lines:
            line = line.strip()

            # 检测章节标题
            if line.startswith('# 第') and '章' in line:
                current_chapter = line.replace('#', '').strip()
                continue

            # 匹配普通练习题
            match = re.match(exercise_pattern, line)
            if match:
                exercise_id = match.group(1)
                # 提取题目描述（直到第一个句号或行尾）
                question_start = line.find(exercise_id) + len(exercise_id)
                question = line[question_start:].strip()

                if question and not question.startswith('.'):
                    exercises.append({
                        'chapter': current_chapter,
                        'question': question
                    })
                continue

            # 匹配带星号的练习题
            match_star = re.match(exercise_pattern_star, line)
            if match_star:
                exercise_id = match_star.group(1)
                question_start = line.find(exercise_id) + len(exercise_id)
                question = line[question_start:].strip()

                if question and not question.startswith('.'):
                    exercises.append({
                        'chapter': current_chapter,
                        'question': question
                    })

        return exercises

    def extract_problems(self, content: str) -> List[Dict[str, Any]]:
        """提取思考题"""
        problems = []
        lines = content.split('\n')
        current_chapter = ""
        current_problem = None

        for line in lines:
            line = line.strip()

            # 检测章节标题
            if line.startswith('# 第') and '章' in line:
                current_chapter = line.replace('#', '').strip()
                continue

            # 匹配思考题标题，如 "# 1-1 算法运行时间的比较"
            if re.match(r'#\s*\d+-\d+', line):
                # 保存上一个思考题
                if current_problem:
                    problems.append(current_problem)

                # 开始新的思考题
                problem_id = re.search(r'(\d+-\d+)', line).group(1)
                title = line.replace('#', '').strip()

                current_problem = {
                    'id': problem_id,
                    'type': 'problem',
                    'chapter': current_chapter,
                    'title': title,
                    'content': [],
                    'full_text': []
                }
                continue

            # 如果当前有思考题，收集内容
            if current_problem and line and not line.startswith('#'):
                # 跳过一些明显的非内容行
                if line.startswith('[') or line.startswith('Omitted'):
                    continue

                current_problem['content'].append(line)
                current_problem['full_text'].append(line)

        # 保存最后一个思考题
        if current_problem:
            problems.append(current_problem)

        # 清理思考题内容
        for problem in problems:
            problem['content'] = ' '.join(problem['content']).strip()
            problem['full_text'] = '\n'.join(problem['full_text']).strip()

        return problems

    def process(self) -> Dict[str, Any]:
        """处理文件并提取所有题目"""
        print(f"正在处理文件: {self.file_path}")

        # 加载文件
        content = self.load_file()
        if not content:
            return {'exercises': [], 'problems': []}

        print("文件加载成功，开始提取练习题...")
        # 提取练习题
        exercises = self.extract_exercises(content)
        print(f"找到 {len(exercises)} 道练习题")

        print("开始提取思考题...")
        # 提取思考题
        problems = self.extract_problems(content)
        print(f"找到 {len(problems)} 道思考题")

        return {
            'exercises': exercises,
            'problems': problems,
            'statistics': {
                'total_exercises': len(exercises),
                'total_problems': len(problems),
                'total_questions': len(exercises) + len(problems)
            }
        }

    def save_to_json(self, data: Dict[str, Any], output_file: str):
        """保存结果到JSON文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def print_summary(self, data: Dict[str, Any]):
        """打印提取结果摘要"""
        stats = data.get('statistics', {})
        print("\n=== 提取结果摘要 ===")
        print(f"练习题数量: {stats.get('total_exercises', 0)}")
        print(f"思考题数量: {stats.get('total_problems', 0)}")
        print(f"总题目数量: {stats.get('total_questions', 0)}")

        # 按章节统计练习题
        exercise_chapters = {}
        for exercise in data.get('exercises', []):
            chapter = exercise.get('chapter', '未知章节')
            exercise_chapters[chapter] = exercise_chapters.get(chapter, 0) + 1

        print("\n=== 练习题章节分布 ===")
        for chapter, count in sorted(exercise_chapters.items()):
            print(f"{chapter}: {count} 道")


def main():
    """主函数"""
    # 输入和输出文件路径
    input_file = "/home/guoziyang/AIgorithm_Agent/src/get_QA/markdown/算法导论.md"
    output_file = "/home/guoziyang/AIgorithm_Agent/src/get_QA/算法导论题目.json"

    # 创建提取器实例
    extractor = ExerciseExtractor(input_file)

    # 处理文件
    result = extractor.process()

    # 保存结果
    extractor.save_to_json(result, output_file)

    # 打印摘要
    extractor.print_summary(result)


if __name__ == "__main__":
    main()