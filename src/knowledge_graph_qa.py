#!/usr/bin/env python3
"""
知识图谱问答系统
基于实体、关系和属性的知识图谱查询和回答
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import jieba
import os


class KnowledgeGraphQA:
    def __init__(self, data_dir: str = None):
        """
        初始化知识图谱问答系统

        Args:
            data_dir: 知识图谱数据目录
        """
        # 如果未指定data_dir，使用默认的相对路径
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data", "knowledge_graph")
        else:
            self.data_dir = data_dir
        self.entities = {}  # 实体表
        self.relations = defaultdict(list)  # 邻接表
        self.chunks = {}  # 文档块表
        self.entity_index = defaultdict(set)  # 快表

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载知识图谱数据"""
        # 加载chunk数据
        chunk_file = os.path.join(self.data_dir, "id2chunk.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

        # 加载图谱数据
        graph_file = os.path.join(self.data_dir, "test_new.json")
        if os.path.exists(graph_file):
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            for item in graph_data:
                start_node = item.get('start_node', {})
                end_node = item.get('end_node', {})
                relation = item.get('relation', '')

                # 提取实体信息
                if start_node:
                    entity_name = start_node.get('properties', {}).get('name', '')
                    entity_type = start_node.get('properties', {}).get('schema_type', '')
                    chunk_id = start_node.get('properties', {}).get('chunk id', '')

                    if entity_name:
                        self.entities[entity_name] = {
                            'type': entity_type,
                            'chunk_id': chunk_id,
                            'properties': start_node.get('properties', {})
                        }
                        self.entity_index[entity_name.lower()].add(entity_name)

                # 存储关系
                if start_node and end_node and relation:
                    start_name = start_node.get('properties', {}).get('name', '')
                    end_name = end_node.get('properties', {}).get('name', '')

                    if start_name and end_name:
                        self.relations[start_name].append({
                            'relation': relation,
                            'target': end_name,
                            'target_properties': end_node.get('properties', {})
                        })

    def _extract_keywords(self, question: str) -> List[str]:
        """提取问题中的关键词"""
        # 使用jieba分词
        words = jieba.lcut(question)

        # 过滤停用词和短词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '什么', '哪些', '如何', '为什么', '呢', '吗', '啊', '？', '？', '。', '！', '~'}
        keywords = [word.strip() for word in words if len(word.strip()) > 1 and word.strip() not in stop_words]

        return keywords

    def _find_entities(self, keywords: List[str]) -> List[str]:
        """根据关键词查找实体"""
        found_entities = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.entity_index:
                found_entities.extend(list(self.entity_index[keyword_lower]))

        # 模糊匹配
        for entity_name in self.entities.keys():
            for keyword in keywords:
                if keyword in entity_name or entity_name in keyword:
                    if entity_name not in found_entities:
                        found_entities.append(entity_name)

        return list(set(found_entities))

    def _get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """获取实体详细信息"""
        return self.entities.get(entity_name, {})

    def _get_entity_relations(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取实体的关系"""
        return self.relations.get(entity_name, [])

    def _get_chunk_content(self, chunk_id: str) -> str:
        """获取文档块内容"""
        return self.chunks.get(chunk_id, '')

    def _answer_what_is(self, entity_name: str) -> str:
        """回答"什么是..."类型的问题"""
        entity_info = self._get_entity_info(entity_name)
        if not entity_info:
            return f"抱歉，我没有找到关于'{entity_name}'的信息。"

        chunk_id = entity_info.get('chunk_id', '')
        content = self._get_chunk_content(chunk_id)

        # 获取实体的属性
        relations = self._get_entity_relations(entity_name)
        attributes = []
        for rel in relations:
            if rel['relation'] == 'has_attribute':
                attr_name = rel['target_properties'].get('name', '')
                if attr_name:
                    attributes.append(attr_name)

        answer = f"{entity_name}是{entity_info.get('type', '')}"
        if attributes:
            answer += f"，具有属性：{', '.join(set(attributes))}"

        # 如果有文档内容，添加部分内容
        if content:
            # 提取前200字符作为摘要
            content_snippet = content[:200].replace('\n', ' ').strip()
            if len(content) > 200:
                content_snippet += "..."
            answer += f"\n\n相关内容：{content_snippet}"

        return answer

    def _answer_how_to(self, entity_name: str) -> str:
        """回答"如何..."类型的问题"""
        entity_info = self._get_entity_info(entity_name)
        if not entity_info:
            return f"抱歉，我没有找到关于'{entity_name}'的信息。"

        chunk_id = entity_info.get('chunk_id', '')
        content = self._get_chunk_content(chunk_id)

        if not content:
            return f"抱歉，没有找到关于'{entity_name}'的详细说明。"

        # 查找包含"如何"、"方法"、"步骤"等关键词的内容段落
        sentences = re.split(r'[。！？\n]', content)
        relevant_sentences = []

        keywords = ['如何', '方法', '步骤', '应该', '需要', '可以', '实现', '操作', '进行']
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence for keyword in keywords) and len(sentence) > 10:
                relevant_sentences.append(sentence)

        if relevant_sentences:
            answer = f"关于{entity_name}：\n\n"
            for i, sentence in enumerate(relevant_sentences[:3]):  # 最多返回3个相关句子
                answer += f"{i+1}. {sentence}。\n"
        else:
            # 如果没有找到相关的方法说明，返回内容的前300字符
            content_snippet = content[:300].replace('\n', ' ').strip()
            if len(content) > 300:
                content_snippet += "..."
            answer = f"关于{entity_name}的相关信息：\n\n{content_snippet}"

        return answer

    def _answer_attribute_question(self, entity_name: str, attribute: str) -> str:
        """回答属性相关问题"""
        entity_info = self._get_entity_info(entity_name)
        if not entity_info:
            return f"抱歉，我没有找到关于'{entity_name}'的信息。"

        relations = self._get_entity_relations(entity_name)

        # 查找特定属性
        for rel in relations:
            if rel['relation'] == 'has_attribute':
                attr_name = rel['target_properties'].get('name', '')
                if attribute in attr_name or attr_name in attribute:
                    return f"{entity_name}的{attr_name}属性已找到。"

        # 如果没有找到精确匹配，返回包含相关内容的文档
        chunk_id = entity_info.get('chunk_id', '')
        content = self._get_chunk_content(chunk_id)

        if content and attribute in content:
            # 提取包含属性的相关句子
            sentences = re.split(r'[。！？\n]', content)
            for sentence in sentences:
                if attribute in sentence and len(sentence.strip()) > 5:
                    return f"关于{entity_name}的{attribute}：{sentence.strip()}"

        return f"抱歉，没有找到关于{entity_name}的{attribute}属性的信息。"

    def _answer_list_questions(self, entity_type: str = None) -> str:
        """回答列举类问题"""
        entities = []

        if entity_type:
            # 列出特定类型的实体
            for name, info in self.entities.items():
                if info.get('type', '') == entity_type:
                    entities.append(name)
        else:
            # 列出所有实体
            entities = list(self.entities.keys())

        if entities:
            if len(entities) <= 20:
                return f"找到{len(entities)}个相关项目：\n" + "\n".join([f"- {entity}" for entity in entities])
            else:
                return f"找到{len(entities)}个相关项目，前20个：\n" + "\n".join([f"- {entity}" for entity in entities[:20]])
        else:
            return "抱歉，没有找到相关项目。"

    def answer(self, question: str) -> str:
        """
        回答问题

        Args:
            question: 用户问题

        Returns:
            回答字符串
        """
        question = question.strip()
        if not question:
            return "请输入您的问题。"

        # 提取关键词
        keywords = self._extract_keywords(question)

        # 查找相关实体
        entities = self._find_entities(keywords)

        # 分析问题类型并生成回答
        question_lower = question.lower()

        # "什么是"类型问题
        if any(word in question_lower for word in ['什么是', '什么叫', '解释']):
            if entities:
                return self._answer_what_is(entities[0])
            else:
                return f"抱歉，我没有找到关于关键词'{' '.join(keywords)}'的相关信息。"

        # "如何"类型问题
        elif any(word in question_lower for word in ['如何', '怎么', '怎样', '方法']):
            if entities:
                return self._answer_how_to(entities[0])
            else:
                return f"抱歉，我没有找到关于如何处理'{' '.join(keywords)}'的信息。"

        # 属性问题
        elif any(word in question_lower for word in ['的', '有什么', '包含', '具有']):
            if entities:
                # 尝试找到属性关键词
                for keyword in keywords:
                    if keyword not in entities[0]:
                        return self._answer_attribute_question(entities[0], keyword)
                return self._answer_what_is(entities[0])
            else:
                return f"抱歉，我没有找到关于'{' '.join(keywords)}'的属性信息。"

        # 列举问题
        elif any(word in question_lower for word in ['有哪些', '列出', '列表', '所有']):
            # 尝试识别实体类型
            entity_type = None
            type_keywords = {'书籍': '书籍', '算法': '算法', '数据结构': '数据结构', '概念': '概念'}
            for keyword in keywords:
                if keyword in type_keywords:
                    entity_type = type_keywords[keyword]
                    break

            return self._answer_list_questions(entity_type)

        # 通用查询：如果有相关实体，返回基本信息
        elif entities:
            answers = []
            for entity in entities[:10]:  # 最多返回10个实体的信息
                entity_info = self._get_entity_info(entity)
                if entity_info:
                    answer = f"{entity}（{entity_info.get('type', '')}）"
                    chunk_id = entity_info.get('chunk_id', '')
                    if chunk_id and chunk_id in self.chunks:
                        content = self.chunks[chunk_id][:100]
                        answer += f"\n简介：{content.replace(chr(10), ' ')}..."
                    answers.append(answer)

            return "\n\n".join(answers)

        # 默认回答
        return f"抱歉，我没有理解您的问题。您可以询问关于算法、数据结构、概念等的相关问题。例如：\n- 什么是二分查找？\n- 如何实现快速排序？\n- 有哪些排序算法？"

    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        return {
            '实体数量': len(self.entities),
            '关系数量': sum(len(relations) for relations in self.relations.values()),
            '文档块数量': len(self.chunks),
            '实体类型': list(set(info.get('type', '') for info in self.entities.values()))
        }


def main():
    """主函数，提供交互式问答界面"""
    # 初始化问答系统
    qa_system = KnowledgeGraphQA()

    # 显示统计信息
    stats = qa_system.get_statistics()
    print(f"知识图谱已加载：")
    print(f"- 实体数量: {stats['实体数量']}")
    print(f"- 关系数量: {stats['关系数量']}")
    print(f"- 文档块数量: {stats['文档块数量']}")
    print(f"- 实体类型: {', '.join(stats['实体类型'])}")
    print("\n输入您的问题，输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    while True:
        try:
            question = input("\n请输入您的问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用！")
                break

            if not question:
                continue

            # 生成回答
            answer = qa_system.answer(question)
            print(f"\n回答: {answer}")

        except KeyboardInterrupt:
            print("\n\n感谢使用！")
            break
        except Exception as e:
            print(f"\n抱歉，处理问题时出现错误: {e}")


if __name__ == "__main__":
    main()