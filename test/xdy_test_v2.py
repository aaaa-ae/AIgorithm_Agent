#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run PEV Agent Framework on CSV (Title + QuestionBody) and write answers to a NEW CSV.

Input : 算法问题_QA_前100条.csv
Output: 算法问题_QA_前100条_with_agent_TP.csv  (TP = Title+QuestionBody)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import time

# ========= 1) 项目路径设置 =========
TEST_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import agent_framework  # noqa: E402


# ========= 2) 自动识别编码读取 CSV =========
def read_csv_auto(path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ("utf-8-sig", "utf-8", "ISO-8859-1", "cp1252")
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[OK] CSV loaded with encoding: {enc}")
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"Failed to decode {path}. Last error: {last_err}")


# ========= 3) 构造 query：Title + QuestionBody =========
def build_query(title: str, qbody: str, max_body_chars: int = 2000) -> str:
    """
    把 Title + QuestionBody 拼起来喂给 agent。
    - max_body_chars: 限制 body 长度，避免极端长文本拖慢/触发模型不稳定。
    """
    title = (title or "").strip()
    qbody = (qbody or "").strip()

    if not qbody:
        return title

    if len(qbody) > max_body_chars:
        qbody = qbody[:max_body_chars] + "\n\n[...truncated...]"

    return f"{title}\n\n{qbody}"


def main():
    input_file = TEST_DIR / "算法问题_QA_前100条.csv"
    output_file = TEST_DIR / "算法问题_QA_前100条_with_agent_TP.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_file}")

    df, used_enc = read_csv_auto(input_file)

    # 必要列检查
    if "Title" not in df.columns:
        raise ValueError(f"输入文件缺少 Title 列。现有列：{list(df.columns)}")

    # QuestionBody 没有也能跑（会退化成只用 Title）
    if "QuestionBody" not in df.columns:
        print("[WARN] 输入文件没有 QuestionBody 列，将退化为只用 Title。")

    # 输出列（新文件，建议不复用旧列，避免混淆）
    # 你也可以保留旧列，这里给你更清晰的“第二轮实验”字段名
    for col in ["AgentAnswer_TP", "AgentIterations_TP", "AgentVerification_TP", "AgentStatus_TP", "AgentQuery_TP"]:
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    print("=" * 60)
    print("Run PEV Agent on Title + QuestionBody")
    print("=" * 60)
    print(f"Input : {input_file} ({used_enc})")
    print(f"Output: {output_file} (utf-8-sig)")
    print(f"Rows  : {total}")
    print("=" * 60)

    for i, row in df.iterrows():
        title = str(row.get("Title", "") or "").strip()
        qbody = str(row.get("QuestionBody", "") or "").strip() if "QuestionBody" in df.columns else ""

        if not title and not qbody:
            continue

        # 断点续跑：已有答案就跳过
        if isinstance(row.get("AgentAnswer_TP", ""), str) and row["AgentAnswer_TP"].strip():
            continue

        query = build_query(title, qbody, max_body_chars=2000)
        df.at[i, "AgentQuery_TP"] = query  # 记录实际喂给 agent 的 query，方便排查

        print(f"\n[{i+1}/{total}] Query(Title+Body): {title}")

        try:
            result = agent_framework(query)
            time.sleep(3)
            # result2 = xdy_test(query)
            # sleep(3)
            # suggestion = llm_judge(result1,result2) # 让大模型来比较两个回答，从相关性角度、覆盖面角度
            # sleep(3)
            
            final_answer = result.get("final_answer", "")
            verification = result.get("verification", {})
            iterations = result.get("iterations", "")

            df.at[i, "AgentAnswer_TP"] = final_answer
            df.at[i, "AgentIterations_TP"] = iterations
            df.at[i, "AgentVerification_TP"] = json.dumps(verification, ensure_ascii=False)

            status = "ok"
            if isinstance(final_answer, str) and ("抱歉" in final_answer and "错误" in final_answer):
                status = "agent_internal_error"
            df.at[i, "AgentStatus_TP"] = status

        except KeyboardInterrupt:
            print("\n[STOP] 用户中断，保存已完成部分...")
            break

        except Exception as e:
            df.at[i, "AgentAnswer_TP"] = f"[SCRIPT_ERROR] {repr(e)}"
            df.at[i, "AgentIterations_TP"] = ""
            df.at[i, "AgentVerification_TP"] = "{}"
            df.at[i, "AgentStatus_TP"] = "script_error"

        # 每 5 条落盘一次，避免跑一半丢
        if (i + 1) % 5 == 0:
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"[SAVE] progress saved to {output_file}")

    # 最终保存：utf-8-sig 方便 Excel 直接打开不乱码
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("\n[DONE] 已写入：", output_file)


if __name__ == "__main__":
    main()
