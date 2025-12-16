import os
import re
import sys
import json
import yaml
import requests
import numpy as np
import faiss
import copy
from camel.agents import ChatAgent
from camel.models import OpenAIModel



# ================================
# 0) chunk 
# ================================
def load_llm_client(config_path):
    """
    根据你的 YAML 加载对应模型的 (api_key, api_base, model)
    并构造 openai SDK 的客户端。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_use = cfg["llm"]["use"]
    provider_cfg = cfg["llm"][llm_use]

    # 支持 key 池：如果是 list，则取第一个 key（可改成轮询）
    if isinstance(provider_cfg, list):
        provider_cfg = provider_cfg[0]

    model_name = provider_cfg["model"]
    api_key = provider_cfg["api_key"]
    api_base = provider_cfg["api_base"]  # 你 YAML 的字段名

    # Camel-AI 要求 url=xxx
    url = api_base.rstrip("/")  # 去掉结尾斜杠避免重复

    # 构建 model 配置字典（过滤 None）
    model_config = {
        "temperature": provider_cfg.get("temperature"),
        "top_p": provider_cfg.get("top_p"),
        "max_tokens": provider_cfg.get("max_tokens")
    }
    model_config = {k: v for k, v in model_config.items() if v is not None}

    model = OpenAIModel(
        model_type=model_name,       
        model_config_dict=model_config,
        api_key=api_key,
        url=url                     
    )

    return model

def extract_metadata_llm(text, config_path=None):
    # 如果未指定config_path，使用默认的相对路径
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "base2.yaml")
    model = load_llm_client(config_path)

    system_prompt = "你是一个专业的中文图书元信息抽取助手，严格输出 JSON。"
    agent = ChatAgent(system_prompt, model=model)

    user_prompt = f"""
        请从下面内容中提取书籍的元信息，务必输出合法 JSON。

        文本内容（前 2000 字）：
        {text}

        输出格式如下：
        {{
        "title": "",
        "author": "",
        "year": "",
        "extra": ""
        }}
    """

    response = agent.step(user_prompt).msg.content.strip()

    # 去掉可能的 ```json 包裹
    response = response.replace("```json", "").replace("```", "")

    try:
        return json.loads(response)
    except:
        print("⚠️ CAMEL 输出的 JSON 不合法，返回空对象")
        return {"title": "", "author": "", "year": "", "extra": ""}
    
def chunk_document(file_path):
    """
    Markdown 文档切分 —— 面向《算法设计与分析》教材的规则定制版

    规则摘要：
    1. “第N章 ...” → 章节根节点，level = 1
    2. 开头数字编号（如 1.2, 3.4.5） → 一般标题，level = 点数 + 1
    3. 含“题”字的标题：
       3.1 无前置数字编号 → 视作当前路径的子标题（level = parent + 1）
       3.2 有前置数字编号：
           - 若路径中存在“习题X” → 作为该“习题X”的子标题，同级分组（如 1. 算法分析题 / 2.算法实现题）
           - 否则 → 退化为普通数字标题
    4. 其他无编号标题 → 视作当前路径的子标题（level = parent + 1）
    5. 标题前面的多余标点（. 、，：等）会自动清理
    """

    # === 1. 读取文件 ===
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"文件读取失败: {e}", file=sys.stderr)
        return None

    # === 2. 元数据（可选） ===
    try:
        metadata = extract_metadata_llm("".join(lines[:2000]))
    except Exception:
        metadata = {"title": "", "author": "", "year": "", "extra": ""}

    # === 3. 标题正则 ===
    # group(1): 开头的数字编号（如 "1", "1.2", "3.4.5"）
    # group(3): 其余文本
    heading_pattern = re.compile(r'^#{1,6}\s*(\d+(\.\d+)*)?\s*(.*)')

    ignore_titles = {
        "计算机", "算法设计与分析", "(第3版)",
        "普通高等教育“十一五”国家级规划教材",
        "高等学校规划教材", "目录"
    }

    # === 4. 收集所有标题位置 ===
    split_points = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        m = heading_pattern.match(line)
        if not m:
            continue

        numbering = m.group(1)            # 可能为 None
        raw_title_body = m.group(3).strip()

        if raw_title_body in ignore_titles:
            continue

        # 检测“第N章”
        chapter_match = re.search(r'第(\d+)章', raw_title_body)

        # 清理标题前端多余符号（把 ". 算法分析题" 变成 "算法分析题"）
        cleaned_title = re.sub(r'^[\s\.\u3002\uFF0C\uFF1A:，、\-—·]+', '', raw_title_body)

        split_points.append({
            "line_index": idx,
            "numbering": numbering,
            "raw_title": raw_title_body,
            "title": cleaned_title,
            "chapter_match": chapter_match,
        })

    # === 5. 构建 chunks ===
    chunks = []
    logical_path = []   # 栈：[{title, numbering, logical_level}, ...]

    chunk_id = 1

    # 文档开头非标题部分
    if split_points:
        first_idx = split_points[0]["line_index"]
        intro = "".join(lines[:first_idx]).strip()
        if intro:
            chunks.append({
                "chunk_id": chunk_id,
                "title": "文档起始/前言/版权信息",
                "is_numbered": False,
                "logical_level": 0,
                "content": intro,
                "metadata": copy.deepcopy(metadata),
                "path_titles": ["文档起始/前言/版权信息"],
                "path_numbering": ["0"],
            })
            chunk_id += 1

    # === 6. 遍历标题，生成 chunk ===
    for i, sp in enumerate(split_points):
        start = sp["line_index"]
        end = split_points[i + 1]["line_index"] if i + 1 < len(split_points) else len(lines)
        content = "".join(lines[start:end]).strip()

        raw_title = sp["raw_title"]
        title = sp["title"]
        numbering = sp["numbering"]
        chapter_match = sp["chapter_match"]

        has_ti = ("题" in raw_title)
        has_numbering = bool(numbering)

        # ---------- 计算 current_level ----------
        if chapter_match:
            # 章节：第N章
            numbering = chapter_match.group(1)
            current_level = 1

        else:
            # 查找最近的“习题X”所在层级
            exercise_level = None
            for node in reversed(logical_path):
                if "习题" in node["title"]:
                    exercise_level = node["logical_level"]
                    break

            # 1）题字标题 + 无数字编号 → 子标题（往下钻）
            if has_ti and not has_numbering:
                if logical_path:
                    current_level = logical_path[-1]["logical_level"] + 1
                else:
                    current_level = 1

            # 2）题字标题 + 有数字编号 → 习题块中的分组标题
            elif has_ti and has_numbering:
                if exercise_level is not None:
                    # 作为“习题X”的直接子标题（同级分组）
                    current_level = exercise_level + 1
                else:
                    # 不在习题块里，就按普通数字标题处理
                    current_level = numbering.count('.') + 1

            # 3）普通数字标题
            elif has_numbering:
                current_level = numbering.count('.') + 1

            # 4）普通无编号标题 → 子标题
            else:
                if logical_path:
                    current_level = logical_path[-1]["logical_level"] + 1
                else:
                    current_level = 1

        # ---------- 更新 logical_path ----------
        # 正确维护路径，只保留真正的父节点链
        while logical_path and logical_path[-1]["logical_level"] >= current_level:
            logical_path.pop()

        logical_path.append({
            "title": title,
            "numbering": numbering,
            "logical_level": current_level,
        })


        # ---------- 路径字段 ----------
        path_titles = [n["title"] for n in logical_path]
        path_numbering = [
            (n["numbering"] if n["numbering"] else n["title"])
            for n in logical_path
        ]

        # ---------- 构建 chunk ----------
        chunk = {
            "chunk_id": chunk_id,
            "title": title,
            "is_numbered": numbering is not None,
            "logical_level": current_level,
            "content": content,
            "metadata": copy.deepcopy(metadata),
            "path_titles": path_titles,
            "path_numbering": path_numbering,
        }
        if numbering:
            chunk["metadata"]["chapter_numbering"] = numbering

        chunks.append(chunk)
        chunk_id += 1

    return chunks

# =========================================================
# 1) 读取配置文件（用于 SiliconFlow embedding）
# =========================================================

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "base2.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

emb_conf = cfg["embedding"]["siliconflow"]
print("Using SiliconFlow embedding:", emb_conf)


# =========================================================
# 2) SiliconFlow embedding API
# =========================================================

def embed_texts_siliconflow(texts, emb_conf):
    url = emb_conf["api_base"].rstrip("/") + "/embeddings"
    api_key = emb_conf["api_key"]
    model = emb_conf["model"]

    payload = {"model": model, "input": texts}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    data = r.json()
    return [item["embedding"] for item in data["data"]]


# =========================================================
# 3) refine 切分，提高召回率
# =========================================================

def split_content(content, max_length=300):
    """把长文本按句子切分，每块不超过 max_length 字符"""
    sentences = re.split(r'(?<=[。！!？?])\s*', content)

    output = []
    cur = ""

    for s in sentences:
        if len(cur) + len(s) > max_length:
            if cur.strip():
                output.append(cur.strip())
            cur = s
        else:
            cur += s

    if cur.strip():
        output.append(cur.strip())

    return output


def refine_chunks_and_metadata(chunks, max_length=300, starting_id=1):
    """
    传入 chunk_document 的原始 chunks，根据 max_length 二次切分。
    ID 从 starting_id 开始递增，保持全局唯一。
    """
    refined_chunks = []
    refined_meta = []
    cur_id = starting_id

    for c in chunks:
        content = c["content"]

        if len(content) <= max_length:
            # 原样保留
            new_chunk = copy.deepcopy(c)
            new_chunk["chunk_id"] = cur_id

            refined_chunks.append(new_chunk)
            refined_meta.append({
                "chunk_id": cur_id,
                "title": new_chunk["title"],
                "content": new_chunk["content"],
                "metadata": copy.deepcopy(new_chunk["metadata"]),
                "path_titles": new_chunk["path_titles"],
                "path_numbering": new_chunk["path_numbering"],
            })
            cur_id += 1

        else:
            # 拆分
            parts = split_content(content, max_length)
            for idx, sub_txt in enumerate(parts):
                new_chunk = {
                    "chunk_id": cur_id,
                    "title": f"{c['title']} (part {idx+1})",
                    "is_numbered": c["is_numbered"],
                    "logical_level": c["logical_level"],
                    "content": sub_txt,
                    "metadata": copy.deepcopy(c["metadata"]),
                    "path_titles": c["path_titles"],
                    "path_numbering": c["path_numbering"],
                }

                refined_chunks.append(new_chunk)
                refined_meta.append({
                    "chunk_id": cur_id,
                    "title": new_chunk["title"],
                    "content": sub_txt,
                    "metadata": copy.deepcopy(c["metadata"]),
                    "path_titles": c["path_titles"],
                    "path_numbering": c["path_numbering"],
                })

                cur_id += 1

    return refined_chunks, refined_meta, cur_id


# =========================================================
# 4) 遍历目录下所有 .md 文件并 chunk
# =========================================================

root_dir = "/home/guoziyang/output/textbook"  # 你的 textbook 根目录

all_chunks = []          # 存放所有文件的 raw chunks
global_id = 1            # 全局 ID 起点

for fname in sorted(os.listdir(root_dir)):
    if not fname.endswith(".md"):
        continue  # 跳过非 md 文件

    fpath = os.path.join(root_dir, fname)
    print(f"Processing file: {fpath}")

    # 每个文件先 chunk_document
    chunks = chunk_document(fpath)

    # 再 refine（让 id 跨文件递增）
    refined_chunks, refined_meta, next_id = refine_chunks_and_metadata(
        chunks,
        max_length=300,
        starting_id=global_id
    )

    global_id = next_id  # 更新全局 ID
    all_chunks.extend(refined_meta)  # 加入总列表

print("Total refined chunks:", len(all_chunks))

# =========================================================
# 5) 保存 refined metadata（docstore）
# =========================================================

refined_meta_path = os.path.join(PROJECT_ROOT, "data", "faiss", "refined_document_chunks3.json")
with open(refined_meta_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("Metadata saved to:", refined_meta_path)


# =========================================================
# 6) 批量 embedding
# =========================================================

def batch_embed(texts, bsz=16):
    all_vecs = []
    for i in range(0, len(texts), bsz):
        batch = texts[i: i+bsz]
        embedding = embed_texts_siliconflow(batch, emb_conf)
        all_vecs.extend(embedding)
    return np.array(all_vecs, dtype="float32")


all_texts = [c["content"] for c in all_chunks]
print("Embedding texts:", len(all_texts))

vectors = batch_embed(all_texts, bsz=16)
dim = vectors.shape[1]
print("Embedding dim:", dim)


# =========================================================
# 7) 构建 FAISS
# =========================================================

index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss_path = os.path.join(PROJECT_ROOT, "data", "faiss", "faiss3.index")
faiss.write_index(index, faiss_path)

print("FAISS index saved to:", faiss_path)
