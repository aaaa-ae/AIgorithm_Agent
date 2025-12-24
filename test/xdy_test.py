#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run PEV Agent Framework on CSV Titles and write answers back to a new CSV.
Input : 算法问题_QA_前100条.csv
Output: 算法问题_QA_前100条_with_agent.csv   (UTF-8-SIG for Excel)
"""

import sys
import json
from pathlib import Path
import pandas as pd

TEST_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import agent_framework  # noqa: E402


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


def main():
    input_file = TEST_DIR / "算法问题_QA_前100条.csv"
    output_file = TEST_DIR / "算法问题_QA_前100条_with_agent.csv"

    df, used_enc = read_csv_auto(input_file)

    if "Title" not in df.columns:
        raise ValueError(f"Missing Title column. Columns: {list(df.columns)}")

    # columns
    for col in ["AgentAnswer", "AgentIterations", "AgentVerification", "AgentStatus"]:
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    print("=" * 60)
    print("Run PEV Agent on Titles")
    print("=" * 60)
    print(f"Input : {input_file} ({used_enc})")
    print(f"Output: {output_file} (utf-8-sig)")
    print(f"Rows  : {total}")
    print("=" * 60)

    for i, row in df.iterrows():
        title = str(row.get("Title", "")).strip()
        if not title:
            continue

        # resume
        if isinstance(row.get("AgentAnswer", ""), str) and row["AgentAnswer"].strip():
            continue

        print(f"\n[{i+1}/{total}] Query: {title}")

        try:
            result = agent_framework(title)

            final_answer = result.get("final_answer", "")
            verification = result.get("verification", {})
            iterations = result.get("iterations", "")

            df.at[i, "AgentAnswer"] = final_answer
            df.at[i, "AgentIterations"] = iterations
            df.at[i, "AgentVerification"] = json.dumps(verification, ensure_ascii=False)

            # status tag for analysis
            status = "ok"
            if isinstance(final_answer, str) and ("抱歉" in final_answer and "错误" in final_answer):
                status = "agent_internal_error"
            if isinstance(verification, dict) and verification.get("has_hallucination") is True:
                status = "hallucination_flagged"
            df.at[i, "AgentStatus"] = status

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted. Saving progress...")
            break
        except Exception as e:
            df.at[i, "AgentAnswer"] = f"[SCRIPT_ERROR] {repr(e)}"
            df.at[i, "AgentIterations"] = ""
            df.at[i, "AgentVerification"] = "{}"
            df.at[i, "AgentStatus"] = "script_error"

        # periodic save
        if (i + 1) % 5 == 0:
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"[SAVE] progress saved to {output_file}")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("\n[DONE] Saved:", output_file)


if __name__ == "__main__":
    main()
