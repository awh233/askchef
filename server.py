"""
PromptBid Demo — Multi-Vertical AI Assistant with Native Ads

Demonstrates PromptBid SDK integration across 8 different AI verticals,
each with its own personality, ad targeting categories, and keywords.

Streaming responses via SSE + OpenRTB 2.6 bid requests per conversation.

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    export PROMPTBID_API_KEY=pb_live_...
    export PROMPTBID_BASE_URL=http://localhost:8080
    python server.py
"""

import os
import json
import uuid
import re
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROMPTBID_API_KEY = os.getenv("PROMPTBID_API_KEY", "")
PROMPTBID_BASE_URL = os.getenv("PROMPTBID_BASE_URL", "http://localhost:8080")

# ── Verticals ──
VERTICALS = {
    "chef": {
        "name": "AskChef",
        "tagline": "AI Cooking Assistant",
        "icon": "🍳",
        "color": "#f97316",
        "bundle": "ai.askchef.app",
        "categories": ["IAB8", "IAB8-5", "IAB8-9"],  # Food & Drink
        "keywords": ["cooking", "recipes", "food", "kitchen", "meal", "dinner", "healthy", "ingredients"],
        "system": """You are AskChef, a friendly AI cooking assistant. Help with recipes, meal planning, ingredient swaps, cooking techniques, and dietary guidance. Use **bold** for recipe names and key terms. Include measurements, temperatures, and times. Keep responses between 50 and 300 words — be concise and practical. Add a tip or variation at the end when relevant.""",
        "suggestions": [
            {"icon": "🍝", "text": "Quick 20-minute weeknight pasta"},
            {"icon": "🥗", "text": "High-protein meal prep ideas"},
            {"icon": "🍕", "text": "Best homemade pizza dough recipe"},
            {"icon": "🍰", "text": "Easy impressive dinner party dessert"},
        ],
    },
    "fitness": {
        "name": "FitCoach",
        "tagline": "AI Fitness Trainer",
        "icon": "💪",
        "color": "#22c55e",
        "bundle": "ai.fitcoach.app",
        "categories": ["IAB17", "IAB17-12", "IAB17-18"],  # Sports, Exercise, Nutrition
        "keywords": ["fitness", "workout", "exercise", "gym", "strength", "cardio", "protein", "muscle"],
        "system": """You are FitCoach, an encouraging AI fitness trainer. Help with workout routines, form tips, nutrition for fitness goals, recovery, and training plans. Use **bold** for exercise names and key metrics. Include sets, reps, rest times, and alternatives. Keep responses between 50 and 300 words. Be motivating but realistic about expectations.""",
        "suggestions": [
            {"icon": "🏋️", "text": "Full body workout with no equipment"},
            {"icon": "🏃", "text": "Couch to 5K running plan"},
            {"icon": "💪", "text": "Best exercises for building muscle"},
            {"icon": "🧘", "text": "Morning stretch routine for flexibility"},
        ],
    },
    "travel": {
        "name": "TravelGuru",
        "tagline": "AI Travel Planner",
        "icon": "✈️",
        "color": "#3b82f6",
        "bundle": "ai.travelguru.app",
        "categories": ["IAB20", "IAB20-3", "IAB20-18"],  # Travel
        "keywords": ["travel", "flights", "hotels", "vacation", "trip", "destination", "budget", "itinerary"],
        "system": """You are TravelGuru, a worldly AI travel planner. Help with destination recommendations, itinerary planning, budget tips, packing lists, and local insider knowledge. Use **bold** for place names, prices, and key details. Keep responses between 50 and 300 words. Include practical logistics like timing and transport. Be enthusiastic about discovery.""",
        "suggestions": [
            {"icon": "🏖️", "text": "Best budget-friendly beach destinations"},
            {"icon": "🗼", "text": "3-day Paris itinerary for first-timers"},
            {"icon": "🎒", "text": "Essential packing list for backpacking"},
            {"icon": "🍜", "text": "Best cities for street food lovers"},
        ],
    },
    "finance": {
        "name": "MoneyMentor",
        "tagline": "AI Financial Guide",
        "icon": "💰",
        "color": "#eab308",
        "bundle": "ai.moneymentor.app",
        "categories": ["IAB13", "IAB13-4", "IAB13-7"],  # Personal Finance
        "keywords": ["finance", "budget", "investing", "savings", "money", "stocks", "retirement", "debt"],
        "system": """You are MoneyMentor, a practical AI financial guide. Help with budgeting, saving strategies, debt management, and financial literacy. Use **bold** for key terms and numbers. Keep responses between 50 and 300 words. Give actionable advice with specific steps. Always note you're not a licensed financial advisor and recommend professional advice for major decisions.""",
        "suggestions": [
            {"icon": "📊", "text": "How to create a monthly budget that works"},
            {"icon": "🏦", "text": "Best strategies to pay off student loans"},
            {"icon": "📈", "text": "Beginner's guide to index fund investing"},
            {"icon": "💳", "text": "How to improve my credit score fast"},
        ],
    },
    "code": {
        "name": "CodePilot",
        "tagline": "AI Coding Assistant",
        "icon": "⚡",
        "color": "#a855f7",
        "bundle": "ai.codepilot.app",
        "categories": ["IAB19", "IAB19-6", "IAB19-18"],  # Technology
        "keywords": ["programming", "code", "developer", "software", "api", "javascript", "python", "web"],
        "system": """You are CodePilot, a sharp AI coding assistant. Help with code examples, debugging, architecture decisions, and developer tools. Use code blocks with language tags. Keep responses between 50 and 300 words. Explain concepts clearly with practical examples. Mention time/space complexity when relevant.""",
        "suggestions": [
            {"icon": "🐍", "text": "Python async/await explained simply"},
            {"icon": "⚛️", "text": "React vs Next.js — when to use which"},
            {"icon": "🗄️", "text": "Design a REST API for a todo app"},
            {"icon": "🐛", "text": "Common JavaScript debugging techniques"},
        ],
    },
    "home": {
        "name": "HomeHelper",
        "tagline": "AI Home & DIY Guide",
        "icon": "🏠",
        "color": "#f59e0b",
        "bundle": "ai.homehelper.app",
        "categories": ["IAB10", "IAB10-1", "IAB10-9"],  # Home & Garden
        "keywords": ["home", "diy", "renovation", "furniture", "decor", "repair", "garden", "tools"],
        "system": """You are HomeHelper, a handy AI home improvement guide. Help with DIY projects, home repairs, interior design ideas, gardening, and tool recommendations. Use **bold** for tool names, materials, and measurements. Keep responses between 50 and 300 words. Include safety warnings where relevant. Give step-by-step instructions for projects.""",
        "suggestions": [
            {"icon": "🔨", "text": "How to patch and paint drywall"},
            {"icon": "🪴", "text": "Low-maintenance indoor plants for beginners"},
            {"icon": "🛋️", "text": "Small apartment organization hacks"},
            {"icon": "💡", "text": "How to install a dimmer switch safely"},
        ],
    },
    "study": {
        "name": "StudyBuddy",
        "tagline": "AI Learning Assistant",
        "icon": "📚",
        "color": "#06b6d4",
        "bundle": "ai.studybuddy.app",
        "categories": ["IAB5", "IAB5-2", "IAB5-10"],  # Education
        "keywords": ["education", "learning", "study", "school", "college", "exam", "tutor", "homework"],
        "system": """You are StudyBuddy, a patient AI tutor and learning assistant. Help with explaining concepts, study techniques, exam prep, and academic writing. Use **bold** for key terms and definitions. Keep responses between 50 and 300 words. Break complex topics into digestible pieces. Use analogies and examples to make concepts click.""",
        "suggestions": [
            {"icon": "🧠", "text": "Best study techniques backed by science"},
            {"icon": "📝", "text": "How to write a strong thesis statement"},
            {"icon": "🔬", "text": "Explain quantum physics simply"},
            {"icon": "📐", "text": "Help me understand calculus derivatives"},
        ],
    },
    "pet": {
        "name": "PetPal",
        "tagline": "AI Pet Care Assistant",
        "icon": "🐾",
        "color": "#ec4899",
        "bundle": "ai.petpal.app",
        "categories": ["IAB16", "IAB16-3", "IAB16-5"],  # Pets
        "keywords": ["pets", "dog", "cat", "veterinary", "training", "grooming", "puppy", "kitten"],
        "system": """You are PetPal, a caring AI pet care assistant. Help with pet health questions, training tips, nutrition, grooming, and behavior issues. Use **bold** for breed names, product names, and key terms. Keep responses between 50 and 300 words. Always recommend consulting a vet for medical concerns. Be warm and supportive — pet parents worry!""",
        "suggestions": [
            {"icon": "🐕", "text": "How to stop my puppy from biting"},
            {"icon": "🐈", "text": "Best diet for an indoor cat"},
            {"icon": "🦮", "text": "Leash training a stubborn dog"},
            {"icon": "🐾", "text": "Signs my pet needs to see a vet"},
        ],
    },
}

# ── App ──
app = FastAPI(title="PromptBid Demo", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Session state ──
sessions: dict = {}


# ── Models ──
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    vertical: str = "chef"

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Help me make pasta",
                "session_id": None,
                "vertical": "chef"
            }
        }

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.message) > 2000:
            raise ValueError("Message length cannot exceed 2000 characters")


class ClickRequest(BaseModel):
    impression_id: str


# ── Streaming AI ──
async def stream_chatgpt(messages: list[dict], system: str):
    if not OPENAI_API_KEY:
        yield "_OpenAI API key not configured._ Set `OPENAI_API_KEY` to enable."
        return

    oai_messages = [{"role": "system", "content": system}]
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST", "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 512,
                "stream": True,
                "messages": oai_messages,
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                err = body.decode()[:200]
                print(f"ChatGPT API error {resp.status_code}: {err}")
                yield f"_ChatGPT API error ({resp.status_code})._ {err}"
                return

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        pass


# ── PromptBid Ad Request ──
async def request_ad(session_id: str, vertical_key: str, extra_keywords: list[str]) -> Optional[dict]:
    if not PROMPTBID_API_KEY:
        return None

    v = VERTICALS.get(vertical_key, VERTICALS["chef"])
    keywords = list(set(v["keywords"] + extra_keywords))[:20]

    bid_request = {
        "id": str(uuid.uuid4()),
        "imp": [{
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
        }],
        "app": {
            "name": v["name"],
            "bundle": v["bundle"],
            "cat": v["categories"],
            "keywords": ",".join(keywords),
            "publisher": {"id": f"promptbid-demo-{vertical_key}"},
        },
        "device": {"ua": f"{v['name']}/3.0", "ip": "127.0.0.1", "os": "web", "js": 1},
        "user": {"id": session_id},
        "at": 2,
        "cur": ["USD"],
    }

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{PROMPTBID_BASE_URL}/api/v1/bid/request",
                headers={"X-API-Key": PROMPTBID_API_KEY, "Content-Type": "application/json"},
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
            try:
                adm = json.loads(adm_raw) if isinstance(adm_raw, str) else adm_raw
            except json.JSONDecodeError:
                adm = {}

            headline = description = cta_url = image_url = ""
            cta_text = "Learn More"
            impression_id = bid.get("id", "")

            for asset in adm.get("assets", []):
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

            cta_url = adm.get("link", {}).get("url", "")
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


def extract_keywords(text: str) -> list[str]:
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    # Filter common stop words
    stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out",
            "has", "have", "been", "some", "them", "than", "its", "this", "that", "with", "from", "what", "which",
            "would", "there", "their", "will", "each", "make", "like", "just", "about", "also", "very", "when"}
    return list(words - stop)[:15]


# ── Routes ──
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"), media_type="text/html")


@app.get("/api/verticals")
async def get_verticals():
    """Return all vertical configs for the frontend."""
    return {k: {key: v[key] for key in ["name", "tagline", "icon", "color", "suggestions"]} for k, v in VERTICALS.items()}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    vertical_key = req.vertical if req.vertical in VERTICALS else "chef"
    v = VERTICALS[vertical_key]

    # Session cleanup: if sessions dict exceeds 1000 entries, remove oldest half
    if len(sessions) > 1000:
        to_remove = sorted(sessions.keys())[:500]
        for sid in to_remove:
            del sessions[sid]

    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "turn_count": 0, "last_ad_turn": -3, "vertical": vertical_key}

    session = sessions[session_id]
    # If vertical changed, reset conversation
    if session.get("vertical") != vertical_key:
        session["messages"] = []
        session["turn_count"] = 0
        session["last_ad_turn"] = -3
        session["vertical"] = vertical_key

    session["turn_count"] += 1
    session["messages"].append({"role": "user", "content": req.message})
    context_messages = session["messages"][-20:]

    async def event_stream():
        full_response = []
        gen = stream_chatgpt(context_messages, v["system"])
        async for chunk in gen:
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

        ai_text = "".join(full_response)
        session["messages"].append({"role": "assistant", "content": ai_text})

        ad = None
        turns_since_ad = session["turn_count"] - session["last_ad_turn"]
        if session["turn_count"] >= 2 and turns_since_ad >= 3:
            extra_kw = extract_keywords(req.message + " " + ai_text)
            ad = await request_ad(session_id, vertical_key, extra_kw)
            if ad:
                session["last_ad_turn"] = session["turn_count"]

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'turn': session['turn_count'], 'ad': ad})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/click")
async def track_click(req: ClickRequest):
    if not PROMPTBID_API_KEY:
        return {"success": False}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{PROMPTBID_BASE_URL}/api/v1/bid/click",
                headers={"X-API-Key": PROMPTBID_API_KEY, "Content-Type": "application/json"},
                json={"impression_id": req.impression_id},
            )
            return resp.json()
    except Exception:
        return {"success": False}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "verticals": list(VERTICALS.keys()),
        "chatgpt_configured": bool(OPENAI_API_KEY),
        "promptbid_configured": bool(PROMPTBID_API_KEY),
        "promptbid_url": PROMPTBID_BASE_URL,
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("RENDER") is None)
