import json
import yaml
import requests
import numpy as np
import faiss
import jieba
import os
from rank_bm25 import BM25Okapi


# ======================================================
# 1. 加载配置
# ======================================================

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "base2.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

emb_conf = cfg["embedding"]["siliconflow"]


# ======================================================
# 2. Embedding
# ======================================================

def embed_texts_siliconflow(texts, emb_conf):
    url = emb_conf["api_base"] + "/embeddings"
    api_key = emb_conf["api_key"]
    model = emb_conf["model"]

    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return [item["embedding"] for item in r.json()["data"]]


def embed_query(text):
    vec = embed_texts_siliconflow([text], emb_conf)[0]
    return np.array(vec, dtype="float32").reshape(1, -1)


# ======================================================
# 3. 加载 FAISS + Docstore
# ======================================================

FAISS_PATH = os.path.join(PROJECT_ROOT, "data", "faiss", "faiss2.index")
DOCSTORE_PATH = os.path.join(PROJECT_ROOT, "data", "faiss", "refined_document_chunks.json")

index = faiss.read_index(FAISS_PATH)
print("FAISS index loaded:", index.ntotal)

with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    docstore = json.load(f)

print("Docstore loaded:", len(docstore))

DOC_BY_ID = {d["chunk_id"]: d for d in docstore}


# ======================================================
# 4. 构建 BM25
# ======================================================

corpus_texts = [d["content"] for d in docstore]
chunk_ids = [d["chunk_id"] for d in docstore]

tokenized_corpus = [list(jieba.cut(t)) for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 initialized!")


# ======================================================
# 5. 检索接口（对外暴露）
# ======================================================

def faiss_search(query, top_k=10):
    q_vec = embed_query(query)
    D, I = index.search(q_vec, top_k)

    results = []
    for cid in I[0]:
        if cid == -1:
            continue
        doc = DOC_BY_ID.get(int(cid))
        if doc:
            results.append(doc)

    return results


def bm25_search(query, top_k=10):
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)

    idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in idx:
        cid = chunk_ids[i]
        doc = DOC_BY_ID.get(cid)
        if doc:
            results.append(doc)

    return results


# ======================================================
# 6. 前置知识检索器
# ======================================================

# 定义固定的算法概念列表
ALGORITHM_CONCEPTS = [
    "枚举", "递归", "分治", "二分查找", "归并排序", "快速排序", "堆排序",
    "插入排序", "冒泡排序", "选择排序", "计数排序", "桶排序", "基数排序",
    "滑动窗口", "双指针", "前缀和", "贪心算法", "Kruskal", "Prim",
    "最小区间覆盖", "Dijkstra", "Huffman编码", "作业调度",
    "最小字典序构造", "动态规划", "最长公共子序列", "背包问题", "编辑距离",
    "最近点对", "DFS", "BFS", "图连通块", "拓扑排序", "最短路径", "最小步数",
    "回溯", "N皇后", "图着色", "剪枝", "最小生成树", "网络流", "KMP", "局部搜索",
    "模拟退火", "遗传算法"
]

# 加载知识图谱
KNOWLEDGE_GRAPH_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "pre_knowledge_graph.json")

def load_knowledge_graph():
    """加载知识图谱数据"""
    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Knowledge graph not found at {KNOWLEDGE_GRAPH_PATH}")
        return []

# 缓存知识图谱
_knowledge_graph = load_knowledge_graph()

def pre_knowledge_search(query):
    """
    基于算法概念检索前置知识

    Args:
        query: 用户查询字符串

    Returns:
        list of tuples: [(prerequisite_concept_name, chunk_object), ...]
    """
    # 1. 识别查询中出现的算法概念
    matched_concepts = []
    query_lower = query.lower()

    for concept in ALGORITHM_CONCEPTS:
        # 简单字符串匹配，检查概念是否出现在查询中
        if concept in query:
            matched_concepts.append(concept)

    if not matched_concepts:
        return []

    # 2. 查找每个匹配概念的直接前置知识
    prerequisite_chunks = {}

    for edge in _knowledge_graph:
        # 检查边的起点是否是匹配的概念
        start_concept = edge.get("start_node", {}).get("properties", {}).get("name", "")

        if start_concept in matched_concepts:
            # 获取前置知识概念名称和 chunk_id
            prereq_concept = edge.get("end_node", {}).get("properties", {}).get("name", "")
            chunk_id = edge.get("end_node", {}).get("properties", {}).get("chunk id")

            if prereq_concept and chunk_id:
                # 确保每个前置概念只添加一次
                if prereq_concept not in prerequisite_chunks:
                    prerequisite_chunks[prereq_concept] = chunk_id

    # 3. 根据 chunk_id 从文档库中检索文档
    results = []

    for prereq_concept, chunk_id in prerequisite_chunks.items():
        # 转换 chunk_id 为整数（如果存储的是字符串）
        try:
            chunk_id_int = int(chunk_id)
            chunk_object = DOC_BY_ID.get(chunk_id_int)

            if chunk_object:
                results.append((prereq_concept, chunk_object))
        except (ValueError, TypeError):
            # 如果转换失败，尝试直接使用 chunk_id
            chunk_object = None
            for doc in docstore:
                if str(doc.get("chunk_id", "")) == str(chunk_id):
                    chunk_object = doc
                    break

            if chunk_object:
                results.append((prereq_concept, chunk_object))

    return results



def main():
    test_queries = [
        "请讲一下枚举",
        "递归和分治有什么关系？"
    ]

    for q in test_queries:
        print("=" * 60)
        print(f"Query: {q}")

        results = pre_knowledge_search(q)

        if not results:
            print("No prerequisite knowledge found.")
            continue

        print(f"Found {len(results)} prerequisite chunks:")
        for concept, chunk in results:
            print(f"- 前置知识点: {concept}")
            print(f"  chunk_id: {chunk.get('chunk_id')}")
            content_preview = chunk.get("content", "")[:120].replace("\n", " ")
            print(f"  content preview: {content_preview}...")
            print()

if __name__ == "__main__":
    main()
