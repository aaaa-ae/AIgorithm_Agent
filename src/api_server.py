from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retrieval.retrieval import (
    faiss_search,
    bm25_search,
    pre_knowledge_search,
)

from agent import agent_framework, format_response

from retrieval.qa_retrieval.qa_retrieval_advanced import (
    AdvancedQARetriever,
    AdvancedRetrievalConfig,
)

from controller import (
    recommend_controller,
    format_prerequisite_results,
    format_qa_results,
)


qa_retriever = AdvancedQARetriever(
    AdvancedRetrievalConfig(
        search_mode="hybrid"
    )
)

app = FastAPI()

# 允许跨域（前端可直接访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求体 ----------
class Query(BaseModel):
    query: str
    top_k: int = 10


# ---------- Root ----------
@app.get("/")
def root():
    return {"status": "RAG API running"}


# ---------- 三个接口 ----------
@app.post("/faiss_search")
def api_faiss(req: Query):
    return {"results": faiss_search(req.query, req.top_k)}


@app.post("/bm25_search")
def api_bm25(req: Query):
    return {"results": bm25_search(req.query, req.top_k)}


@app.post("/rag_answer")
def api_rag(req: Query):
    response = agent_framework(req.query)
    answer, citations = format_response(response)
    
    return {
            "answer": answer,
            "citations": citations
    }
    

@app.post("/pre_knowledge_search")
def api_pre_knowledge(req: Query):
    raw_results = pre_knowledge_search(req.query)

    prerequisites = []
    seen = set()

    for concept, chunk in raw_results:
        if concept in seen:
            continue
        seen.add(concept)

        prerequisites.append({
            "concept": concept,
            "content": chunk.get("content", "")
        })

    return {
        "query": req.query,
        "prerequisites": prerequisites
    }


@app.post("/qa_search")
def api_qa_search(req: Query):
    """
    高级题库检索接口
    """
    results = qa_retriever.search(
        query=req.query,
        top_k=req.top_k
    )

    return {
        "results": [
            {
                "score": score,
                "question": item.get("question"),
                "chapter": item.get("chapter"),
                "answer": item.get("answer", "")
            }
            for item, score in results
        ]
    }


@app.post("/smart_answer")
def api_smart_answer(req: Query):
    """
    智能推荐接口 - 根据 query 自动决定返回内容

    返回内容可能包括：
    - answer: 基础答案（始终返回）
    - citations: 参考文献（始终返回）
    - decision: 推荐决策信息
    - prerequisites: 前置知识点（如果 recommend_prerequisite=True）
    - related_questions: 相关题库（如果 recommend_qa_bank=True）
    """
    # 1. 推荐判断
    decision = recommend_controller(req.query)

    # 2. 基础答案 (始终返回)
    response = agent_framework(req.query)
    answer, citations = format_response(response)

    result = {
        "answer": answer,
        "citations": citations,
        "decision": {
            "recommend_prerequisite": decision["recommend_prerequisite"],
            "recommend_qa_bank": decision["recommend_qa_bank"],
            "detected_concepts": decision["detected_concepts"],
            "reason": decision["reason"]
        }
    }

    # 3. 根据决策添加额外内容
    if decision["recommend_prerequisite"]:
        prereq_raw = pre_knowledge_search(req.query)
        result["prerequisites"] = format_prerequisite_results(prereq_raw)

    if decision["recommend_qa_bank"]:
        qa_raw = qa_retriever.search(req.query, top_k=5)
        result["related_questions"] = format_qa_results(qa_raw)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
