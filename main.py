import json
import os

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1024 * 10
TIMEOUT_S = 60.0

LANGUAGE_NAMES = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    text: str | None = None
    title: str | None = None
    image: str | None = None
    category: str | None = None
    summary: str | None = None
    error: str | None = None


class SummarizeRequest(BaseModel):
    text: str
    options: SummaryOptions
    categories: list[str]


class SummarizeResponse(BaseModel):
    success: bool
    category: str | None = None
    summary: str | None = None
    error: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    articleText: str
    messages: list[ChatMessage]


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


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    crawled = crawl_url(req.url)
    if not crawled["success"]:
        return ProcessResponse(success=False)

    text, title, image = crawled["text"], crawled.get("title"), crawled.get("image")
    try:
        system_prompt = build_summary_system_prompt(req.options, req.categories)
        raw = await call_claude(system_prompt, text)
        parsed = extract_json(raw, req.categories)
    except Exception as e:
        return ProcessResponse(success=True, text=text, title=title, image=image, error=str(e))

    return ProcessResponse(success=True, text=text, title=title, image=image, category=parsed["category"], summary=parsed["summary"])


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest) -> SummarizeResponse:
    try:
        system_prompt = build_summary_system_prompt(req.options, req.categories)
        raw = await call_claude(system_prompt, req.text)
        parsed = extract_json(raw, req.categories)
        return SummarizeResponse(success=True, category=parsed["category"], summary=parsed["summary"])
    except Exception as e:
        return SummarizeResponse(success=False, error=str(e))


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    system_prompt = build_chat_system_prompt(req.articleText)

    async def event_stream():
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
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
