import json
import time
import os
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional


# =========================
# 配置区
# =========================

API_BASE = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-32B"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
MAX_TOKENS = 700

RETRY_TIMES = 3
REQUEST_TIMEOUT = 180

# 请求间隔（秒），用于软限速
REQUEST_INTERVAL = 2.0


# =========================
# 核心类
# =========================

class AnswerAdder:
    def __init__(
        self,
        api_key: Optional[str],
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ):
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.processed_count = 0
        self.failed_count = 0

    # ---------- IO ----------

    def load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_json(self, data: Dict[str, Any], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- Prompt ----------

    def build_prompt(self, question: str, chapter: str) -> str:
        return f"""
你是一名算法教师，请针对《算法导论》中的题目给出【一种合理的解题思路与分析】，
重点解释算法思想、推导过程与复杂度分析。

注意：
- 不要声称这是“标准答案”或“唯一答案”
- 目标是帮助理解算法思想

章节：{chapter}

题目：
{question}

请按以下结构回答：
1. 问题理解
2. 核心思想 / 算法思路
3. 推导或证明（如需要）
4. 复杂度分析
5. 小结
""".strip()

    # ---------- API ----------

    def call_llm(self, question: str, chapter: str) -> Optional[str]:
        if not self.api_key:
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "你是一名资深算法教师。"},
                {"role": "user", "content": self.build_prompt(question, chapter)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": MAX_TOKENS,
        }

        for attempt in range(1, RETRY_TIMES + 1):
            try:
                resp = requests.post(
                    API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

                else:
                    print(f"  API失败({resp.status_code})，重试 {attempt}/{RETRY_TIMES}")

            except Exception as e:
                print(f"  请求异常: {e}，重试 {attempt}/{RETRY_TIMES}")

            time.sleep(2 ** attempt)

        return None

    # ---------- 处理逻辑 ----------

    # def process_items(self, items: List[Dict[str, Any]], item_type: str):
    #     """处理 exercises 或 problems（仅未回答的）"""

    #     pending = [x for x in items if 'answer' not in x]
    #     print(f"待处理 {item_type}: {len(pending)}")

    #     for idx, item in enumerate(pending, 1):
    #         chapter = item.get('chapter', '未知章节')

    #         if item_type == 'exercise':
    #             question = item.get('question', '')
    #         else:
    #             title = item.get('title', '')
    #             content = item.get('content', '')
    #             question = f"{title}\n{content}" if content else title

    #         print(f"[{item_type} {idx}] {chapter}")

    #         answer = self.call_llm(question, chapter)

    #         if answer:
    #             item['answer'] = answer
    #             item['answer_meta'] = {
    #                 "model": MODEL_NAME,
    #                 "temperature": self.temperature,
    #                 "top_p": self.top_p,
    #                 "generated_at": datetime.now().isoformat(),
    #             }
    #             self.processed_count += 1
    #         else:
    #             item['answer'] = "答案生成失败"
    #             item['answer_failed_at'] = datetime.now().isoformat()
    #             self.failed_count += 1

    #         time.sleep(REQUEST_INTERVAL)

    def process_items(self, items: List[Dict[str, Any]]):
        pending = [x for x in items if 'answer' not in x]
        print(f"待处理题目数: {len(pending)}")

        for idx, item in enumerate(pending, 1):
            chapter = item.get('chapter', '未知章节')
            question = item.get('question', '')

            print(f"[{idx}] {chapter}")

            answer = self.call_llm(question, chapter)

            if answer:
                item['answer'] = answer
                item['answer_meta'] = {
                    "model": MODEL_NAME,
                    "generated_at": datetime.now().isoformat()
                }
                self.processed_count += 1
            else:
                item['answer'] = "答案生成失败"
                self.failed_count += 1

            time.sleep(REQUEST_INTERVAL)



# =========================
# main
# =========================

def main():
    FILE_PATH = "/home/chenyifan/get_QA/算法导论mini.json"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("未检测到 OPENAI_API_KEY，程序终止")
        return

    adder = AnswerAdder(api_key)

    data = adder.load_json(FILE_PATH)

    # exercises = data.get('exercises', [])
    # problems = data.get('problems', [])

    if not isinstance(data, list):
        raise ValueError("期望 JSON 顶层是 list")

    items = data

    print("=== CLRS 解题思路生成器 ===")


    start_time = time.time()

    # if exercises:
    #     adder.process_items(exercises, 'exercise')
    #     adder.save_json(data, FILE_PATH)

    # if problems:
    #     adder.process_items(problems, 'problem')
    #     adder.save_json(data, FILE_PATH)

    adder.process_items(items)
    adder.save_json(data, FILE_PATH)

    end_time = time.time()

    print("\n=== 完成 ===")
    print(f"成功: {adder.processed_count}")
    print(f"失败: {adder.failed_count}")
    print(f"耗时: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main()
