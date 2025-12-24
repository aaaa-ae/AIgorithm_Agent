#!/usr/bin/env python3
"""
算法题目抽取和重写系统
从教材的 chunk 化 JSON 数据中，自动抽取出所有"算法题目"，
并将这些题目重写为"信息完整、无需依赖上下文即可独立解答的标准算法题"。

重要变更：每个 chunk 视为一个完整的题目单元，不做任何基于正则或编号的拆分。
拆分与否的判断应当被视为一个高层语义决策，不在当前代码中处理。

四阶段处理流程：
1. 题目检测：判断 chunk 是否可能包含题目
2. 题目分类：统一返回"single"，不进行多题识别
3. 题目抽取：直接使用 chunk.title 作为题目的 title，chunk.content 作为题目正文
4. 题目重写：对题目进行重写，使其成为可独立解答的完整题目
"""

import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import jieba
from dataclasses import dataclass
import hashlib


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Question:
    """算法题目数据结构"""
    question_id: str  # 唯一标识
    title: str  # 题目标题/编号
    description: str  # 题目描述
    context: str  # 上下文信息
    source: str  # 来源信息
    rewrite_prompt: str  # 重写时需要的提示信息
    metadata: Dict[str, Any]  # 额外元数据


class AlgorithmQuestionExtractor:
    """算法题目抽取和重写系统"""

    def __init__(self, data_file: str, output_file: str = "algorithm_questions.json"):
        self.data_file = data_file
        self.output_file = output_file
        self.chunks = []
        self.extracted_questions = []

        # 配置参数
        self.question_keywords = [
            '练习', '题目', '问题', '算法题', '编程题', '实验题',
            '提高题', '习题', '例题', '思考题', 'Problems', 'Exercises'
        ]

        # 题目编号正则模式（仅用于检测，不用于拆题）
        self.number_patterns = [
            r'\d+\.\d+-\d+',  # 18.2-1
            r'\d+\.\d+\.\d+',  # 2.3.1
            r'练习\s*\d+',  # 练习 1
            r'习题\s*\d+',  # 习题 1
            r'第\d+\s*题',  # 第1题
            r'\d+\.\d+',  # 4.1
            r'[A-Za-z]\d+',  # A1, B2
        ]

        # 初始化jieba分词
        jieba.initialize()

    def load_data(self) -> None:
        """加载chunk数据"""
        logger.info(f"正在加载数据文件: {self.data_file}")
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"成功加载 {len(self.chunks)} 个chunks")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def stage1_detect_questions(self, chunk: Dict[str, Any]) -> bool:
        """
        第一阶段：高精度题目检测（判别式）
        目标：只保留「可独立求解的标准算法题」，强力过滤：
        - 定义 / 定理 / 证明
        - 章节结构编号
        - 教学性说明 / 示例
        """

        content = chunk.get("content", "")
        title = chunk.get("title", "")
        path_titles = chunk.get("path_titles", [])

        # ===============================
        # 0. 基础长度过滤（过短必不是题）
        # ===============================
        if not content or len(content) < 20:
            return False

        # ===============================
        # 1. 路径级强信号（练习/习题章节）
        # ===============================
        PATH_QUESTION_KEYWORDS = [
            '练习', '习题', '问题', '算法题', '编程题', '实验题'
        ]

        path_hit = any(
            any(k in path_title for k in PATH_QUESTION_KEYWORDS)
            for path_title in path_titles
        )

        # ===============================
        # 2. 严格题号检测（必须）
        #   ⚠️ 明确区分「题号」vs「章节号」
        # ===============================
        QUESTION_ID_PATTERNS = [
            r'(练习|习题|问题)\s*\d+',
            r'第\s*\d+\s*题',
            r'\d+\.\d+-\d+',          # 18.2-1
            r'\d+\.\d+\.\d+\s*[（(]', # 2.3.1(...)
        ]

        has_question_id = any(
            re.search(r'(^|\n)\s*' + p, content)
            for p in QUESTION_ID_PATTERNS
        )

        if not has_question_id:
            return False


        # ===============================
        # 3. 额外补强（标题/路径弱信号兜底）
        # ===============================
        TITLE_KEYWORDS = ['题', '练习', '习题']

        title_hit = any(k in title for k in TITLE_KEYWORDS)

        # 最终判定：
        # - 强规则已全部满足
        # - 路径 or 标题 至少一个提供辅助确认
        return path_hit or title_hit


    def stage2_classify_questions(self, chunk: Dict[str, Any]) -> str:
        """
        第二阶段：题目分类（已废弃，统一返回"single"）
        每个 chunk 视为一个完整的题目单元，不做拆分
        """
        return "single"

    def stage3_extract_questions(self, chunk: Dict[str, Any], classification: str) -> List[Question]:
        """
        第三阶段：题目抽取
        每个 chunk 视为一个完整的题目单元，不做任何拆分
        """
        questions = []

        # 构建上下文信息
        context = self._build_context(chunk)

        # 直接抽取为一个题目
        question = self._extract_single_question(chunk, context)
        if question:
            questions.append(question)

        return questions

    def stage4_rewrite_question(self, question: Question) -> Question:
        """
        第四阶段：题目重写
        将题目重写为可独立解答的完整题目
        """
        # 这里可以调用LLM API进行智能重写
        # 目前先做基础的信息补全和格式化

        rewritten_description = self._basic_rewrite(question)

        # 创建重写后的题目
        rewritten_question = Question(
            question_id=question.question_id + "_rewritten",
            title=question.title,
            description=rewritten_description,
            context="",  # 重写后不需要额外上下文
            source=question.source,
            rewrite_prompt="",
            metadata={
                **question.metadata,
                "rewritten": True,
                "rewrite_timestamp": datetime.now().isoformat()
            }
        )

        return rewritten_question

    def _build_context(self, chunk: Dict[str, Any]) -> str:
        """构建题目上下文信息"""
        context_parts = []

        # 章节信息
        path_titles = chunk.get('path_titles', [])
        if path_titles:
            context_parts.append(f"章节: {' -> '.join(path_titles[-3:])}")

        # 书籍信息
        metadata = chunk.get('metadata', {})
        book_title = metadata.get('title', '')
        if book_title:
            context_parts.append(f"书籍: {book_title}")

        return "\n".join(context_parts)

    def _extract_single_question(self, chunk: Dict[str, Any], context: str) -> Optional[Question]:
        """
        抽取单个题目
        直接使用 chunk.title 作为题目的 title，chunk.content 作为题目正文内容
        不做任何基于正则或编号的拆分
        """
        title = chunk.get('title', '')
        content = chunk.get('content', '')
        chunk_id = chunk['chunk_id']

        # 直接使用 chunk.content 作为题目描述
        description = content

        # 清理描述（仅做基础清理，不拆分）
        description = self._clean_description(description)

        if not description or len(description) < 10:
            return None

        question_id = f"q_{chunk_id}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

        return Question(
            question_id=question_id,
            title=title,
            description=description,
            context=context,
            source=self._build_source_info(chunk),
            rewrite_prompt=self._build_rewrite_prompt(chunk, title, description),
            metadata={
                "chunk_id": chunk_id,
                "classification": "single",
                "original_length": len(content)
            }
        )

    def _clean_description(self, description: str) -> str:
        """清理题目描述"""
        # 移除多余的空白字符
        description = re.sub(r'\s+', ' ', description)

        # 移除开头和结尾的标点符号
        description = description.strip('。，！？；：""''《》')

        # 移除Markdown格式标记
        description = re.sub(r'[#*`_\[\]]+', '', description)

        # 移除图片链接
        description = re.sub(r'!\[.*?\]\(.*?\)', '', description)

        # 移除网址
        description = re.sub(r'https?://\S+', '', description)

        return description.strip()

    def _build_source_info(self, chunk: Dict[str, Any]) -> str:
        """构建来源信息"""
        path_titles = chunk.get('path_titles', [])
        metadata = chunk.get('metadata', {})

        source_parts = []

        if path_titles:
            source_parts.append(" -> ".join(path_titles[-3:]))

        book_title = metadata.get('title', '')
        if book_title:
            source_parts.append(book_title)

        return " | ".join(source_parts)

    def _build_rewrite_prompt(self, chunk: Dict[str, Any], title: str, description: str) -> str:
        """构建题目重写提示"""
        return f"""
        请将以下算法题目重写为信息完整、无需依赖上下文即可独立解答的标准算法题：

        题目编号：{title}
        题目描述：{description}
        上下文：{self._build_context(chunk)}

        重写要求：
        1. 补充必要的背景信息和定义
        2. 明确输入和输出格式
        3. 添加约束条件（如果有）
        4. 提供示例（如果有）
        5. 确保题目完整性和独立性
        """

    def _basic_rewrite(self, question: Question) -> str:
        """基础重写功能（简化版）"""
        # 简化重写，只添加必要的上下文信息
        rewritten = ""

        # 添加背景信息（如果内容不够完整）
        if question.context and len(question.context) > 50:
            rewritten += f"背景：{question.context[:100]}...\n\n"

        # 主要内容就是题目描述
        rewritten += question.description

        return rewritten

    def process_all_chunks(self, rewrite_questions: bool = True) -> Dict[str, Any]:
        """处理所有chunks"""
        logger.info("开始处理所有chunks...")

        stats = {
            "total_chunks": len(self.chunks),
            "detected_chunks": 0,
            "total_questions": 0,
            "rewritten_questions": 0
        }

        for i, chunk in enumerate(self.chunks):
            if i % 1000 == 0:
                logger.info(f"处理进度: {i}/{len(self.chunks)}")

            # 第一阶段：题目检测
            if not self.stage1_detect_questions(chunk):
                continue

            stats["detected_chunks"] += 1

            # 第二阶段：题目分类（统一返回"single"）
            classification = self.stage2_classify_questions(chunk)

            # 第三阶段：题目抽取
            questions = self.stage3_extract_questions(chunk, classification)

            if not questions:
                continue

            # 第四阶段：题目重写
            if rewrite_questions:
                for question in questions:
                    rewritten_question = self.stage4_rewrite_question(question)
                    self.extracted_questions.append(rewritten_question)
                    stats["rewritten_questions"] += 1
            else:
                self.extracted_questions.extend(questions)

            # 更新统计信息
            stats["total_questions"] += len(questions)

        logger.info(f"处理完成！统计信息: {stats}")
        return stats

    def save_results(self) -> None:
        """保存结果（精简版：只保存书名+章节+content）"""
        logger.info(f"正在保存结果到: {self.output_file}")

        result = {
            "metadata": {
                "total_questions": len(self.extracted_questions),
                "extraction_time": datetime.now().isoformat(),
                "source_file": self.data_file
            },
            "questions": []
        }

        for question in self.extracted_questions:
            # 提取书名和章节信息
            book_name = ""
            chapter = ""

            # 从source字段提取书名和章节
            source_parts = question.source.split(" | ")
            if len(source_parts) >= 2:
                chapter = source_parts[0]  # 章节信息
                book_name = source_parts[1]  # 书名
            elif source_parts:
                book_name = source_parts[0]

            # 构建精简的结果
            simplified_question = {
                "book_name": book_name,
                "chapter": chapter,
                "content": question.description
            }

            result["questions"].append(simplified_question)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存，共 {len(self.extracted_questions)} 道题目")

    def run(self, rewrite_questions: bool = True) -> Dict[str, Any]:
        """运行完整的处理流程"""
        try:
            # 加载数据
            self.load_data()

            # 处理所有chunks
            stats = self.process_all_chunks(rewrite_questions)

            # 保存结果
            self.save_results()

            return stats

        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            raise


def main():
    """主函数"""
    # 配置文件路径
    data_file = "/home/guoziyang/AIgorithm_Agent/data/faiss/refined_document_chunks.json"
    output_file = "/home/guoziyang/AIgorithm_Agent/output/extracted_algorithm_questions.json"

    # 创建抽取器
    extractor = AlgorithmQuestionExtractor(data_file, output_file)

    # 运行处理流程
    stats = extractor.run(rewrite_questions=False)  # 不重写，保存原始题目

    # 打印统计信息
    print("\n=== 处理统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()