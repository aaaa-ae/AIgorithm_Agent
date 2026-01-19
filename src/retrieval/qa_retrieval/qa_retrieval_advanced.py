#!/usr/bin/env python3
"""
高级题库检索系统
支持语义搜索、多维度匹配和智能排序
"""

import json
import os
import re
import jieba
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from collections import Counter
import math


@dataclass
class AdvancedRetrievalConfig:
    """高级检索配置"""
    qa_bank_path: Path = (
    Path(__file__)
    .resolve()
    .parents[3]          # 回到 AIgorithm_Agent 根目录
    / "data"
    / "qa_bank"
    / "answered_questions.json"
    )
    top_k: int = 10
    min_score: float = 0.05

    # 搜索模式
    search_mode: str = "hybrid"  # "keyword", "semantic", "hybrid"

    # TF-IDF配置
    use_tfidf: bool = True
    min_word_freq: int = 2



class TFIDFCalculator:
    """TF-IDF计算器"""

    def __init__(self):
        self.documents = []
        self.vocab = set()
        self.idf = {}

    def fit(self, documents: List[str]):
        """训练TF-IDF模型"""
        self.documents = documents
        # 构建词汇表
        word_counts = Counter()
        doc_word_sets = []

        for doc in documents:
            words = set(jieba.lcut(doc))
            doc_word_sets.append(words)
            word_counts.update(words)

        # 过滤低频词
        self.vocab = {word for word, count in word_counts.items() if count >= 2}

        # 计算IDF
        n_docs = len(documents)
        for word in self.vocab:
            df = sum(1 for words in doc_word_sets if word in words)
            self.idf[word] = math.log(n_docs / (df + 1))

    def transform(self, documents: List[str]) -> np.ndarray:
        """将文档转换为TF-IDF向量"""
        vectors = []
        for doc in documents:
            words = jieba.lcut(doc)
            word_count = Counter(words)
            vector = []

            for word in sorted(self.vocab):
                tf = word_count.get(word, 0)
                tfidf = tf * self.idf.get(word, 0)
                vector.append(tfidf)

            vectors.append(vector)

        return np.array(vectors)

    def cosine_similarity(self, query_vec: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # 归一化
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(doc_vectors))

        query_vec = query_vec / query_norm
        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1
        doc_vectors = doc_vectors / doc_norms[:, np.newaxis]

        # 计算余弦相似度
        similarities = np.dot(doc_vectors, query_vec)
        return similarities


class AdvancedQARetriever:
    """高级题库检索器"""

    def __init__(self, config: AdvancedRetrievalConfig = None):
        self.config = config or AdvancedRetrievalConfig()
        self.logger = logging.getLogger(__name__)
        self.qa_data = []
        self.tfidf_calculator = TFIDFCalculator() if self.config.use_tfidf else None
        self.document_vectors = None

        # 初始化jieba
        jieba.initialize()

        self.load_qa_bank()
        if self.tfidf_calculator:
            self.build_index()

    def load_qa_bank(self):
        """加载题库数据"""
        try:
            with open(self.config.qa_bank_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            self.logger.info(f"成功加载题库，共 {len(self.qa_data)} 道题目")
        except Exception as e:
            self.logger.error(f"加载题库失败: {e}")
            raise

    def build_index(self):
        """构建搜索索引"""
        if not self.tfidf_calculator:
            return

        # 准备文档
        documents = []
        for item in self.qa_data:
            doc = item.get('question', '') + ' ' + item.get('chapter', '')
            if self.config.search_mode == 'hybrid':
                doc += ' ' + item.get('answer', '')[:500]  # 答案前500字
            documents.append(doc)

        # 训练TF-IDF
        self.logger.info("正在构建TF-IDF索引...")
        self.tfidf_calculator.fit(documents)
        self.document_vectors = self.tfidf_calculator.transform(documents)

    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = jieba.lcut(text)
        # 过滤停用词和短词
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '对', '为', '等', '中', '可以', '将', '具有'}
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords

    def calculate_keyword_score(self, query: str, text: str) -> float:
        """计算关键词匹配分数"""
        query_keywords = set(self.extract_keywords(query.lower()))
        text_keywords = set(self.extract_keywords(text.lower()))

        if not query_keywords:
            return 0.0

        # 计算Jaccard相似度
        intersection = query_keywords.intersection(text_keywords)
        union = query_keywords.union(text_keywords)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def calculate_semantic_similarity(self, query: str, item_idx: int) -> float:
        """计算语义相似度（使用TF-IDF）"""
        if not self.tfidf_calculator or self.document_vectors is None:
            return 0.0

        # 转换查询为向量
        query_vector = self.tfidf_calculator.transform([query])[0]

        # 计算相似度
        similarities = self.tfidf_calculator.cosine_similarity(
            query_vector,
            self.document_vectors
        )

        return similarities[item_idx]

    def calculate_answer_relevance(self, query: str, answer: str) -> float:
        """计算答案相关性"""
        if not answer:
            return 0.0

        # 只检查答案的前500字
        answer_preview = answer[:500]
        return self.calculate_keyword_score(query, answer_preview) * 0.8

    def calculate_total_score(self, query: str, item_idx: int) -> float:
        item = self.qa_data[item_idx]
        question = item.get("question", "")
        chapter = item.get("chapter", "")
        answer = item.get("answer", "")

        # 1. lexical on question (精确信号)
        keyword_score = self.calculate_keyword_score(query, question)

        # 2. semantic similarity (主排序信号)
        semantic_score = 0.0
        if self.config.search_mode in ["semantic", "hybrid"]:
            semantic_score = self.calculate_semantic_similarity(query, item_idx)

        # 3. answer relevance (弱补充信号)
        answer_score = self.calculate_answer_relevance(query, answer)

        # 4. chapter boost（结构性先验）
        chapter_boost = 0.0
        if chapter and any(k in chapter for k in self.extract_keywords(query)):
            chapter_boost = 0.05

        # 5. hybrid score
        total_score = (
            0.5 * semantic_score +
            0.3 * keyword_score +
            0.2 * answer_score +
            chapter_boost
        )

        return min(total_score, 1.0)


    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """执行搜索"""
        if not query:
            return []

        top_k = top_k or self.config.top_k
        self.logger.info(f"搜索查询: {query} (模式: {self.config.search_mode})")

        # 计算所有题目的分数
        scored_items = []
        for idx, item in enumerate(self.qa_data):
            score = self.calculate_total_score(query, idx)
            if score >= self.config.min_score:
                scored_items.append((item, score))

        # 按分数排序
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # 返回前K个结果
        return scored_items[:top_k]

    def search_with_filters(self, query: str, chapter_filter: str = None,
                          top_k: int = None) -> List[Tuple[Dict, float]]:
        """带过滤器的搜索"""
        # 预过滤
        if chapter_filter:
            filtered_data = [item for item in self.qa_data
                           if chapter_filter.lower() in item.get('chapter', '').lower()]
            # 临时替换数据
            original_data = self.qa_data
            self.qa_data = filtered_data
            results = self.search(query, top_k)
            self.qa_data = original_data
            return results
        else:
            return self.search(query, top_k)

    def get_suggestions(self, query: str, num_suggestions: int = 5) -> List[str]:
        """获取搜索建议"""
        # 简单实现：从题目中提取高频词作为建议
        all_keywords = []
        for item in self.qa_data:
            question = item.get('question', '')
            keywords = self.extract_keywords(question)
            all_keywords.extend(keywords)

        # 统计词频
        keyword_freq = Counter(all_keywords)

        # 找出包含查询词的相似词
        query_keywords = set(self.extract_keywords(query))
        suggestions = []

        for keyword, freq in keyword_freq.most_common():
            if keyword in query_keywords:
                continue
            # 简单的相似度检查
            for qk in query_keywords:
                if qk in keyword or keyword in qk:
                    suggestions.append(keyword)
                    break
            if len(suggestions) >= num_suggestions:
                break

        return suggestions


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='高级题库检索系统')
    parser.add_argument('query', nargs='*', help='搜索关键词')
    parser.add_argument('-k', '--top-k', type=int, default=10, help='返回结果数量')
    parser.add_argument('-m', '--mode', choices=['keyword', 'semantic', 'hybrid'],
                       default='hybrid', help='搜索模式')
    parser.add_argument('-c', '--chapter', help='章节过滤')
    parser.add_argument('-s', '--suggestions', action='store_true', help='显示搜索建议')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 构建查询
    query = ' '.join(args.query) if args.query else input("请输入搜索关键词：")

    # 创建配置
    config = AdvancedRetrievalConfig(search_mode=args.mode)

    # 创建检索器
    try:
        retriever = AdvancedQARetriever(config)
    except Exception as e:
        print(f"错误：{e}")
        return 1

    # 搜索建议
    if args.suggestions:
        suggestions = retriever.get_suggestions(query)
        if suggestions:
            print("\n搜索建议：")
            for s in suggestions:
                print(f"  - {s}")
            print()

    # 执行搜索
    if args.chapter:
        results = retriever.search_with_filters(query, args.chapter, args.top_k)
        print(f"\n搜索结果（章节：{args.chapter}）：")
    else:
        results = retriever.search(query, args.top_k)
        print("\n搜索结果：")

    # 显示结果
    for idx, (item, score) in enumerate(results, 1):
        chapter = item.get('chapter', '未知章节')
        question = item.get('question', '')

        print(f"\n{idx}. 【匹配度: {score:.2%}】{chapter}")
        print(f"题目：{question[:150]}{'...' if len(question) > 150 else ''}")

    if not results:
        print("未找到匹配的题目。")

    return 0


if __name__ == "__main__":
    exit(main())