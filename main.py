import json
import os
import time
from datetime import datetime

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1024 * 10
TIMEOUT_S = 60.0

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://wjzdjvyefjtivtayayfc.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqemRqdnllZmp0aXZ0YXlheWZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM1MTE0OTYsImV4cCI6MjA5OTA4NzQ5Nn0.MxIpIu7kCJn__MF_ciyLpCbSQ0dIeMf8sgfuVhSYfl0")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


LANGUAGE_NAMES = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] ➡️ Request: {request.method} {request.url.path}", flush=True)
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ⬅️ Response: {request.method} {request.url.path} - Status: {response.status_code} ({process_time:.2f}ms)", flush=True)
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ❌ Response Error: {request.method} {request.url.path} - Error: {str(e)} ({process_time:.2f}ms)", flush=True)
        raise e



class SummaryOptions(BaseModel):
    emoji: bool = False
    kidFriendly: bool = False
    language: str = "ko"


class ProcessRequest(BaseModel):
    url: str
    options: SummaryOptions
    categories: list[str]


class ProcessResponse(BaseModel):
    success: bool
    id: int | None = None
    error: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    articleText: str
    messages: list[ChatMessage]


def get_supabase_headers():
    key = SUPABASE_SERVICE_ROLE_KEY if SUPABASE_SERVICE_ROLE_KEY else SUPABASE_ANON_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }


async def insert_supabase_content(url: str) -> int:
    payload = {
        "url": url,
        "tag": "Article",
        "status": "pending",
        "data": {
            "processing": True,
            "stage": "Fetching article..."
        },
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/contents",
            headers=get_supabase_headers(),
            json=payload
        )
        if res.status_code != 201:
            raise RuntimeError(f"Supabase insert failed: {res.status_code} - {res.text}")
        data = res.json()
        return data[0]["id"]


async def update_supabase_content(content_id: int, patch: dict):
    async with httpx.AsyncClient() as client:
        res = await client.patch(
            f"{SUPABASE_URL}/rest/v1/contents?id=eq.{content_id}",
            headers=get_supabase_headers(),
            json=patch
        )
        if res.status_code not in (200, 204):
            raise RuntimeError(f"Supabase update failed: {res.status_code} - {res.text}")


def crawl_url(url: str) -> dict:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {"success": False}
    extracted = trafilatura.extract(downloaded, url=url, output_format="json", with_metadata=True)
    if not extracted:
        return {"success": False}
    result = json.loads(extracted)
    text = result.get("text")
    if not text:
        return {"success": False}
    return {"success": True, "text": text, "image": result.get("image"), "title": result.get("title")}


def build_summary_system_prompt(options: SummaryOptions, categories: list[str]) -> str:
    style_lines = []
    if options.kidFriendly:
        style_lines.append("Summarize using simple words and short sentences a child could understand.")
    if options.emoji:
        style_lines.append("Sprinkle in emojis that fit the content throughout the summary.")
    return f"""You are an AI that classifies and summarizes news articles.
Classify the given article into exactly one of these categories: {', '.join(categories)}. The category value must exactly match one from this list.
Then summarize the article in {LANGUAGE_NAMES[options.language]} in exactly 3 lines, each at most 100 characters. Format exactly as "1. ...\\n2. ...\\n3. ..." — a \\n only ever separates one numbered item from the next. Never insert a \\n inside a single numbered item (e.g. "1. xx\\nxxx\\n2. xxx" is forbidden); each item must stay on one line.
{chr(10).join(style_lines)}
Output only the JSON format below. No markdown, no explanations.
{{ "category": "{categories[0]}", "summary": "..." }}"""


def build_chat_system_prompt(article_text: str) -> str:
    return f"""You are a helpful assistant answering questions about the article below. Use it as your primary source of truth. If the question cannot be answered from the article, say so clearly.

Article:
{article_text}"""


def extract_json(raw: str, categories: list[str]) -> dict:
    import re

    stripped = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", stripped)
    fenced = re.sub(r"\s*```\s*$", "", fenced).strip()
    parsed = json.loads(fenced)
    if parsed.get("category") not in categories:
        raise ValueError(f"Unknown category: {parsed.get('category')}")
    if not isinstance(parsed.get("summary"), str) or not parsed["summary"]:
        raise ValueError("Missing summary")
    return parsed


def compute_option_key(options: SummaryOptions) -> str:
    parts = []
    if options.emoji:
        parts.append("emoji")
    if options.kidFriendly:
        parts.append("child")
    return "_".join(parts) if parts else "default"


async def call_claude(system_prompt: str, article: str) -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        res = await client.post(
            f"{CLAUDE_BASE}/messages",
            headers={"Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": CLAUDE_MODEL, "max_tokens": MAX_TOKENS, "system": system_prompt, "messages": [{"role": "user", "content": article}]},
        )
        if res.status_code != 200:
            raise RuntimeError(f"Claude {res.status_code}: {res.text}")
        data = res.json()
        return data["content"][0]["text"]


async def run_process_pipeline(content_id: int, url: str, options: SummaryOptions, categories: list[str]):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Starting pipeline for URL: {url}", flush=True)
    # 1. 크롤링
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Crawling URL...", flush=True)
    crawled = crawl_url(url)
    if not crawled["success"]:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] ❌ Crawling failed.", flush=True)
        await update_supabase_content(content_id, {
            "tag": "Not Article",
            "data": {
                "processing": False,
                "error": "Failed to crawl or extract text from URL."
            }
        })
        return

    text = crawled["text"]
    title = crawled.get("title")
    image = crawled.get("image")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Successfully crawled. Title: '{title}', Text length: {len(text)} chars", flush=True)

    # DB stage update
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Updating status to 'Summarizing' in Supabase", flush=True)
    await update_supabase_content(content_id, {
        "data": {
            "processing": True,
            "stage": "Summarizing article...",
            "title": title,
            "thumbnail": image,
            "original": text
        }
    })

    # 2. Claude 요약
    try:
        system_prompt = build_summary_system_prompt(options, categories)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Requesting summary from Claude (model: {CLAUDE_MODEL})", flush=True)
        raw = await call_claude(system_prompt, text)
        parsed = extract_json(raw, categories)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Claude response parsed successfully. Category: '{parsed['category']}'", flush=True)
        
        option_key = compute_option_key(options)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Saving final summary under key '{option_key}' to Supabase", flush=True)
        await update_supabase_content(content_id, {
            "tag": "Article",
            "data": {
                "original": text,
                "title": title,
                "thumbnail": image,
                "category": parsed["category"],
                "summaries": {
                    option_key: parsed["summary"]
                },
                "processing": False
            }
        })
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] Pipeline completed successfully", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Pipeline:{content_id}] ❌ Pipeline failed: {str(e)}", flush=True)
        await update_supabase_content(content_id, {
            "tag": "Article",
            "data": {
                "original": text,
                "title": title,
                "thumbnail": image,
                "error": str(e),
                "processing": False
            }
        })


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest, background_tasks: BackgroundTasks) -> ProcessResponse:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] Received URL: {req.url}", flush=True)
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] Inserting initial record into Supabase...", flush=True)
        content_id = await insert_supabase_content(req.url)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] Inserted successfully. content_id: {content_id}", flush=True)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] Scheduling background pipeline task", flush=True)
        background_tasks.add_task(
            run_process_pipeline,
            content_id,
            req.url,
            req.options,
            req.categories
        )
        response = ProcessResponse(success=True, id=content_id)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] Returning response: {response.model_dump()}", flush=True)
        return response
    except Exception as e:
        response = ProcessResponse(success=False, error=str(e))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /process] ❌ Error occurred: {str(e)}. Returning: {response.model_dump()}", flush=True)
        return response


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /chat] Received request. Messages count: {len(req.messages)}, Article length: {len(req.articleText)} chars", flush=True)
    system_prompt = build_chat_system_prompt(req.articleText)

    async def event_stream():
        chunk_count = 0
        total_bytes = 0
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /chat] Starting Claude stream connection...", flush=True)
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            async with client.stream(
                "POST",
                f"{CLAUDE_BASE}/messages",
                headers={"Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01"},
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": system_prompt,
                    "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                    "stream": True,
                },
            ) as response:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /chat] Stream connected. Returning response generator.", flush=True)
                async for chunk in response.aiter_bytes():
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [POST /chat] Stream finished. Yielded {chunk_count} chunks, total {total_bytes} bytes.", flush=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [GET /health] Health check requested", flush=True)
    res = {"status": "ok"}
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [GET /health] Returning: {res}", flush=True)
    return res
