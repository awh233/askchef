"""
AskChef — AI Cooking Assistant powered by Claude/ChatGPT + PromptBid Ads

A demo app showing the full PromptBid SDK integration:
  1. User sends a message
  2. Backend calls Claude or ChatGPT for the AI response
  3. Backend fires a bid request to PromptBid with conversation context
  4. If an ad wins the auction, it's returned alongside the AI response
  5. Frontend renders the ad as a native card in the conversation

Usage:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export PROMPTBID_API_KEY=pb_live_...
    export PROMPTBID_BASE_URL=http://localhost:8080  # or https://promptbiddev.onrender.com
    python server.py
"""

import os
import json
import uuid
import time
import asyncio
import re
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ──
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROMPTBID_API_KEY = os.getenv("PROMPTBID_API_KEY", "")
PROMPTBID_BASE_URL = os.getenv("PROMPTBID_BASE_URL", "http://localhost:8080")

APP_NAME = "AskChef"
APP_BUNDLE = "ai.askchef.app"
APP_CATEGORIES = ["IAB8", "IAB8-5", "IAB8-9"]  # Food & Drink, Cooking, Nutrition
DEFAULT_KEYWORDS = ["cooking", "recipes", "food", "kitchen", "meal", "dinner", "healthy"]

SYSTEM_PROMPT = """You are AskChef, a friendly and knowledgeable AI cooking assistant. You help people with:
- Recipe suggestions and step-by-step instructions
- Meal planning and prep advice
- Ingredient substitutions
- Cooking techniques and tips
- Dietary guidance (vegetarian, vegan, gluten-free, etc.)
- Kitchen equipment recommendations

Keep responses conversational, practical, and concise (2-4 paragraphs max).
Use bold for recipe names and key terms. Include specific measurements and times.
If someone asks about something non-food related, gently steer back to cooking."""

# ── App ──
app = FastAPI(title="AskChef", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session state ──
sessions: dict = {}  # session_id -> { messages: [], turn_count: int, last_ad_turn: int }


# ── Models ──
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: str = "claude"  # "claude" or "chatgpt"


class ClickRequest(BaseModel):
    impression_id: str


# ── Keyword extraction ──
FOOD_KEYWORDS = {
    "recipe", "cook", "cooking", "bake", "baking", "grill", "grilling", "fry",
    "roast", "sauté", "boil", "steam", "meal", "dinner", "lunch", "breakfast",
    "snack", "dessert", "appetizer", "salad", "soup", "pasta", "rice", "chicken",
    "beef", "pork", "fish", "salmon", "shrimp", "vegetable", "vegan", "vegetarian",
    "gluten-free", "keto", "healthy", "protein", "carb", "calorie", "nutrition",
    "ingredient", "spice", "herb", "sauce", "marinade", "seasoning", "oven",
    "stovetop", "instant pot", "air fryer", "slow cooker", "prep", "chop",
    "dice", "mince", "blend", "mix", "stir", "whisk", "knead", "marinate",
    "grocery", "organic", "fresh", "frozen", "pantry", "kitchen", "knife",
    "cutting board", "pan", "pot", "skillet", "wok", "baking sheet",
    "meal prep", "budget", "quick", "easy", "simple", "30-minute",
    "chocolate", "cake", "bread", "pizza", "burger", "taco", "sushi",
    "curry", "stew", "smoothie", "juice", "coffee", "tea",
}


def extract_keywords(text: str) -> list[str]:
    """Extract food/cooking keywords from user message."""
    words = set(re.findall(r'\b[a-z]+(?:-[a-z]+)?\b', text.lower()))
    found = words & FOOD_KEYWORDS
    # Always include base keywords
    result = list(found | {"cooking", "food", "recipes"})
    return result[:15]  # Cap at 15 keywords for the bid request


# ── AI Chat ──
async def call_claude(messages: list[dict]) -> str:
    """Call Anthropic Claude API."""
    if not ANTHROPIC_API_KEY:
        return "_Claude API key not configured._ Set `ANTHROPIC_API_KEY` env var to enable Claude responses."

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            },
        )
        if resp.status_code != 200:
            return f"_Claude API error ({resp.status_code})._ Check your API key."
        data = resp.json()
        return data["content"][0]["text"]


async def call_chatgpt(messages: list[dict]) -> str:
    """Call OpenAI ChatGPT API."""
    if not OPENAI_API_KEY:
        return "_OpenAI API key not configured._ Set `OPENAI_API_KEY` env var to enable ChatGPT responses."

    # Convert from Anthropic format to OpenAI format
    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 1024,
                "messages": oai_messages,
            },
        )
        if resp.status_code != 200:
            return f"_ChatGPT API error ({resp.status_code})._ Check your API key."
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── PromptBid Ad Request ──
async def request_ad(session_id: str, keywords: list[str]) -> Optional[dict]:
    """Send a bid request to PromptBid and return the winning ad if any."""
    if not PROMPTBID_API_KEY:
        return None

    bid_request = {
        "id": str(uuid.uuid4()),
        "imp": [
            {
                "id": "1",
                "native": {
                    "request": json.dumps({
                        "ver": "1.2",
                        "assets": [
                            {"id": 0, "required": 1, "title": {"len": 90}},
                            {"id": 1, "required": 1, "data": {"type": 1, "len": 200}},
                            {"id": 2, "required": 0, "data": {"type": 12, "len": 20}},
                            {"id": 3, "required": 0, "img": {"type": 3, "w": 320, "h": 200}},
                        ],
                    }),
                    "ver": "1.2",
                },
                "bidfloor": 0.5,
                "bidfloorcur": "USD",
            }
        ],
        "app": {
            "name": APP_NAME,
            "bundle": APP_BUNDLE,
            "cat": APP_CATEGORIES,
            "keywords": ",".join(keywords),
            "publisher": {"id": "askchef-demo"},
        },
        "device": {
            "ua": "AskChef/1.0 (Demo App)",
            "ip": "127.0.0.1",
            "os": "web",
            "js": 1,
            "connectiontype": 2,
        },
        "user": {
            "id": session_id,
        },
        "at": 2,  # Second-price auction
        "cur": ["USD"],
    }

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{PROMPTBID_BASE_URL}/api/v1/bid/request",
                headers={
                    "X-API-Key": PROMPTBID_API_KEY,
                    "Content-Type": "application/json",
                },
                json=bid_request,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            seatbids = data.get("seatbid", [])
            if not seatbids or not seatbids[0].get("bid"):
                return None

            bid = seatbids[0]["bid"][0]
            adm_raw = bid.get("adm", "{}")

            # Parse native markup
            try:
                adm = json.loads(adm_raw) if isinstance(adm_raw, str) else adm_raw
            except json.JSONDecodeError:
                adm = {}

            # Extract creative fields from native assets
            headline = ""
            description = ""
            cta_text = "Learn More"
            cta_url = ""
            image_url = ""
            impression_id = bid.get("id", "")

            assets = adm.get("assets", [])
            for asset in assets:
                if asset.get("title"):
                    headline = asset["title"].get("text", "")
                elif asset.get("data"):
                    dtype = asset["data"].get("type", 0)
                    if dtype == 1:
                        description = asset["data"].get("value", "")
                    elif dtype == 12:
                        cta_text = asset["data"].get("value", "Learn More")
                elif asset.get("img"):
                    image_url = asset["img"].get("url", "")

            link = adm.get("link", {})
            cta_url = link.get("url", "")

            # Get impression tracker
            imp_trackers = adm.get("imptrackers", [])
            imp_url = imp_trackers[0] if imp_trackers else f"{PROMPTBID_BASE_URL}/api/v1/bid/imp?impid={impression_id}"

            if not headline:
                return None

            return {
                "impression_id": impression_id,
                "headline": headline,
                "description": description,
                "cta_text": cta_text,
                "cta_url": cta_url,
                "image_url": image_url,
                "imp_tracker": imp_url,
                "price": bid.get("price", 0),
            }

    except Exception:
        return None


# ── Routes ──
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "index.html"),
        media_type="text/html",
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Initialize session
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "turn_count": 0,
            "last_ad_turn": -3,  # Allow ad on first eligible turn
        }

    session = sessions[session_id]
    session["turn_count"] += 1

    # Add user message to history
    session["messages"].append({"role": "user", "content": req.message})

    # Keep last 20 messages for context
    context_messages = session["messages"][-20:]

    # Call AI
    if req.model == "chatgpt":
        ai_response = await call_chatgpt(context_messages)
    else:
        ai_response = await call_claude(context_messages)

    # Add assistant response to history
    session["messages"].append({"role": "assistant", "content": ai_response})

    # Decide whether to show an ad (every ~3 turns, not on first message)
    ad = None
    turns_since_ad = session["turn_count"] - session["last_ad_turn"]
    if session["turn_count"] >= 2 and turns_since_ad >= 3:
        keywords = extract_keywords(req.message + " " + ai_response)
        ad = await request_ad(session_id, keywords)
        if ad:
            session["last_ad_turn"] = session["turn_count"]

    return {
        "session_id": session_id,
        "response": ai_response,
        "model": req.model,
        "ad": ad,
        "turn": session["turn_count"],
    }


@app.post("/click")
async def track_click(req: ClickRequest):
    """Proxy click tracking to PromptBid."""
    if not PROMPTBID_API_KEY:
        return {"success": False, "error": "Not configured"}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{PROMPTBID_BASE_URL}/api/v1/bid/click",
                headers={
                    "X-API-Key": PROMPTBID_API_KEY,
                    "Content-Type": "application/json",
                },
                json={"impression_id": req.impression_id},
            )
            return resp.json()
    except Exception:
        return {"success": False}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "chatgpt_configured": bool(OPENAI_API_KEY),
        "promptbid_configured": bool(PROMPTBID_API_KEY),
        "promptbid_url": PROMPTBID_BASE_URL,
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
    )
