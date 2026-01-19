import asyncio
import json
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from crawl4ai import AsyncWebCrawler


# =========================C
# 固定 URL 列表（v0 demo）
# =========================
CANDIDATE_URLS = [
    "https://en.wikipedia.org/wiki/Dynamic_programming",
    "https://cp-algorithms.com/dynamic_programming/intro-to-dp.html",
    "https://oi-wiki.org/dp/",
    "https://www.geeksforgeeks.org/dynamic-programming/",
    "https://www.topcoder.com/thrive/articles/dynamic-programming-from-novice-to-advanced",
    "https://en.wikipedia.org/wiki/Knapsack_problem",
    "https://cp-algorithms.com/dynamic_programming/knapsack.html",
    "https://oi-wiki.org/dp/knapsack/",
    "https://en.wikipedia.org/wiki/Longest_common_subsequence_problem",
    "https://en.wikipedia.org/wiki/Longest_increasing_subsequence",
    "https://cp-algorithms.com/sequences/longest_increasing_subsequence.html",
    "https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/",
    "https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/",
    "https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm",
    "https://cp-algorithms.com/graph/bellman_ford.html",
    "https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm",
    "https://cp-algorithms.com/graph/all-pair-shortest-path-floyd-warshall.html",
    "https://en.wikipedia.org/wiki/Edit_distance",
    "https://www.geeksforgeeks.org/edit-distance-dp-5/",
    "https://cp-algorithms.com/string/edit-distance.html",
]

# =========================
# 全局配置
# =========================
USER_AGENT = "v0-demo-bot"
PROBE_TIMEOUT = 12
PROBE_CONCURRENCY = 10
CRAWL_CONCURRENCY = 3
KEEP_MAX = 12

ALIVE_URLS_FILE = "alive_urls.json"
CRAWL_RESULTS_FILE = "crawl_results.jsonl"


# =========================
# robots.txt 处理（带缓存）
# =========================
_robot_cache = {}


def can_fetch(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if robots_url not in _robot_cache:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            _robot_cache[robots_url] = rp
        except Exception:
            # v0 阶段：robots.txt 读取失败 → 保守拒绝
            _robot_cache[robots_url] = None
            return False

    rp = _robot_cache[robots_url]
    if rp is None:
        return False

    return rp.can_fetch(user_agent, url)


# =========================
# URL 探活
# =========================
@dataclass
class ProbeResult:
    source_url: str
    final_url: Optional[str]
    status: str
    http_status: Optional[int]
    content_type: Optional[str]
    error: Optional[str]


async def probe_one(session: aiohttp.ClientSession, url: str) -> ProbeResult:
    headers = {"User-Agent": USER_AGENT}

    async def _req(method: str):
        try:
            resp = await session.request(
                method,
                url,
                headers=headers,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT),
            )
            return resp
        except Exception:
            return None

    resp = await _req("HEAD")
    if resp is None or resp.status in (403, 405):
        resp = await _req("GET")

    if resp is None:
        return ProbeResult(url, None, "timeout", None, None, "request_failed")

    http_status = resp.status
    final_url = str(resp.url)
    content_type = resp.headers.get("Content-Type")

    if http_status in (401, 403, 429):
        return ProbeResult(url, final_url, "blocked", http_status, content_type, None)
    if http_status in (404, 410):
        return ProbeResult(url, final_url, "dead", http_status, content_type, None)
    if content_type and "text/html" not in content_type.lower():
        return ProbeResult(url, final_url, "non_html", http_status, content_type, None)
    if 200 <= http_status < 400:
        return ProbeResult(url, final_url, "alive", http_status, content_type, None)

    return ProbeResult(url, final_url, "error", http_status, content_type, "unexpected_status")


async def probe_urls(urls: List[str]) -> List[ProbeResult]:
    sem = asyncio.Semaphore(PROBE_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        async def _run(u: str):
            async with sem:
                return await probe_one(session, u)

        return await asyncio.gather(*[_run(u) for u in urls])


# =========================
# Crawl4AI 抓正文
# =========================
async def crawl_urls(urls: List[str]):
    sem = asyncio.Semaphore(CRAWL_CONCURRENCY)

    # 清空旧结果
    open(CRAWL_RESULTS_FILE, "w", encoding="utf-8").close()

    async with AsyncWebCrawler(verbose=False) as crawler:

        async def _crawl(u: str):
            async with sem:
                record = {
                    "url": u,
                    "status": None,
                    "title": None,
                    "markdown": None,
                    "text": None,
                    "error": None,
                    "elapsed_sec": None,
                }
                start = time.time()
                try:
                    result = await crawler.arun(url=u)
                    record["title"] = getattr(result, "title", None)
                    record["markdown"] = getattr(result, "markdown", None)
                    record["text"] = getattr(result, "text", None)

                    if (record["markdown"] or "").strip() or (record["text"] or "").strip():
                        record["status"] = "success"
                    else:
                        record["status"] = "empty_content"
                        record["error"] = "no_text_or_markdown"

                except Exception as e:
                    record["status"] = "error"
                    record["error"] = str(e)
                finally:
                    record["elapsed_sec"] = round(time.time() - start, 3)
                    with open(CRAWL_RESULTS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        await asyncio.gather(*[_crawl(u) for u in urls])


# =========================
# 主流程
# =========================
async def main():
    print(f"[1] Candidate URLs: {len(CANDIDATE_URLS)}")

    probe_results = await probe_urls(CANDIDATE_URLS)

    alive = [r.final_url for r in probe_results if r.status == "alive"]
    print(f"[2] Probe alive: {len(alive)}")

    # robots.txt 过滤
    robots_allowed = []
    robots_blocked = []

    for u in alive:
        if can_fetch(u, USER_AGENT):
            robots_allowed.append(u)
        else:
            robots_blocked.append(u)

    robots_allowed = robots_allowed[:KEEP_MAX]

    print(f"[3] robots allowed: {len(robots_allowed)}")
    print(f"[3] robots blocked: {len(robots_blocked)}")

    with open(ALIVE_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(robots_allowed, f, ensure_ascii=False, indent=2)

    print(f"[Output] {ALIVE_URLS_FILE}")

    print("[4] Crawl4AI crawling...")
    await crawl_urls(robots_allowed)

    print(f"[Output] {CRAWL_RESULTS_FILE}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
