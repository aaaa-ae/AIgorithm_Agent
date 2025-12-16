import yaml
import re
import os

from camel.agents import ChatAgent
from camel.models import OpenAIModel
from retrieval import faiss_search, bm25_search, pre_knowledge_search


# ======================================================
# 1. 加载 YAML 配置（仅保留 LLM 配置）
# ======================================================

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "base2.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# LLM 配置
llm_provider = cfg["llm"][cfg["llm"]["use"]]


# ======================================================
# 2. 初始化 OpenAIModel（Camel 的 LLM Wrapper）
# ======================================================

def load_llm_client(cfg):
    model_name = llm_provider["model"]
    api_key = llm_provider["api_key"]
    url = llm_provider["api_base"].rstrip("/")

    model_config = {
        "temperature": llm_provider.get("temperature"),
        "top_p": llm_provider.get("top_p"),
        "max_tokens": llm_provider.get("max_tokens")
    }
    model_config = {k: v for k, v in model_config.items() if v is not None}

    model = OpenAIModel(
        model_type=model_name,
        model_config_dict=model_config,
        api_key=api_key,
        url=url
    )
    return model


llm = load_llm_client(cfg)
print("LLM Loaded:", llm_provider["model"])


# ======================================================
# 3. 从模型最终输出中提取正文和参考文献
# ======================================================

def extract_body_and_refs(ans: str):
    """
    从模型最终输出中提取：
    - 正文内容（【正文】...）
    - 参考文献（【参考文献】...）
    """

    # 提取正文
    body_match = re.search(r"【正文】\s*(.*?)\s*【参考文献】", ans, re.S)
    body = body_match.group(1).strip() if body_match else ""

    # 提取参考文献部分
    ref_match = re.search(r"【参考文献】\s*(.*)", ans, re.S)
    refs_raw = ref_match.group(1).strip() if ref_match else ""

    # 把参考文献按行拆开
    refs = [line.strip() for line in refs_raw.split("\n") if line.strip()]

    return body, refs


# ======================================================
# 4. RAG 问答接口（增强版）
# ======================================================

def rag_answer(query, top_k=10):
    # 1. 检索（来自 retrieval.py）
    faiss_docs = faiss_search(query, top_k)
    bm25_docs = bm25_search(query, top_k)

    # 2. 合并 + 去重
    merged = {d["chunk_id"]: d for d in faiss_docs + bm25_docs}
    docs = list(merged.values())

    if not docs:
        return "未检索到相关教材内容，请尝试换一种问法。"

    # 3. 引用编号分配
    ref_numbers_chunk = {doc["chunk_id"]: i + 1 for i, doc in enumerate(docs)}

    # 构建用于 LLM 的 context
    context = "\n\n".join(
        f"[{ref_numbers_chunk[d['chunk_id']]}] 《{d['metadata'].get('title', '未知书名')}》 标题信息: {d['path_titles']}\n正文信息:{d['content']}"
        for d in docs
    )

    # 文献去重（根据书名）
    unique_titles = []
    for d in docs:
        title = d["metadata"].get("title", "未知书名")
        if title not in unique_titles:
            unique_titles.append(title)

    # 为每本书选择最低层级章节标题
    title2chapter = {}
    for d in docs:
        title = d["metadata"].get("title", "未知书名")
        if title in title2chapter:
            continue

        path_titles = d.get("path_titles") or []
        if isinstance(path_titles, list) and len(path_titles) > 0:
            chapter_title = path_titles[-1]
        else:
            chapter_title = "附加章节"

        title2chapter[title] = chapter_title

    # 为每本书编号
    ref_numbers_title = {title: i + 1 for i, title in enumerate(unique_titles)}

    # 构造最终参考文献（书名 - 最低级标题）
    references = "\n".join(
        f"[{ref_numbers_title[title]}] 《{title}》 - {title2chapter[title]}"
        for title in unique_titles
    )

    # 4. Prompt：指导模型输出格式
    system_prompt = """
你是一名严格遵循引用规则的教材问答助手。

你必须按照以下步骤生成答案：

步骤 A：从参考内容中挑选出所有与回答相关的“原始编号”，
并按正文中出现顺序重新编码为 [1]、[2]、[3]、[4] ……。

步骤 B：输出一个 JSON 对象，包含：
{
  "used_original_ids": [...],
  "mapping": { "3": 1, "5": 2 },
  "body": "...正文内容，引用必须使用新编号如 [1][2]...",
  "references": {
      "1": "书名 - 最低级标题",
      "2": "书名 - 最低级标题"
  }
}

步骤 C：输出最终格式化答案：

【正文】
(body)

【参考文献】
[1] 书名 - 最低级标题
[2] 书名 - 最低级标题

要求：
- 正文必须使用所有“新编号”引用
- 不得编造参考文献
- 不得遗漏正文中的编号
"""

    user_prompt = f"""
问题：{query}

参考内容（原始编号 → 内容）：
{context}

参考文献（原始编号 → 书名 - 最低级标题）：
{references}

请严格按照 system_prompt 的步骤 A → B → C 来生成。
"""

    # 5. LLM 调用
    agent = ChatAgent(system_prompt, model=llm)
    response = agent.step(user_prompt)

    ans = response.msg.content if response.msg else ""
    body, refs = extract_body_and_refs(ans)

    # 将正文和参考文献拼成最终输出
    final_output = body + "\n\n【参考文献】\n" + "\n".join(refs)

    return final_output


# ======================================================
# 5. 命令行交互入口
# ======================================================

if __name__ == "__main__":

    print("\n===== 系统已加载完毕，可以开始查询 =====\n")

    while True:
        query = input("\n请输入你的问题（或 exit 退出）：\n> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        print("\n--- FAISS 搜索结果（Top 10）---")
        fdocs = faiss_search(query, top_k=10)
        for d in fdocs:
            print(f"[{d['chunk_id']}] {d['title']}  |  {d['content'][:60]}...")

        print("\n--- BM25 搜索结果（Top 10）---")
        bdocs = bm25_search(query, top_k=10)
        for d in bdocs:
            print(f"[{d['chunk_id']}] {d['title']}  |  {d['content'][:60]}...")

        print("\n--- 前置知识点 搜索结果（Top 10）---")
        results = pre_knowledge_search(query)
        for concept, chunk in results:
            print(f"- 前置知识点: {concept}")
            print(f"  chunk_id: {chunk.get('chunk_id')}")
            content_preview = chunk.get("content", "")[:120].replace("\n", " ")
            print(f"  content preview: {content_preview}...")
            print()

        print("\n--- 最终 RAG 回答 ---\n")
        final_output = rag_answer(query, top_k=10)
        print(final_output)
