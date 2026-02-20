# PromptBid Demo — AI Assistants with Native Ads

A multi-vertical demo app showing the complete PromptBid SDK integration. Features 8 AI assistants (AskChef, FitCoach, TravelGuru, MoneyMentor, CodePilot, HomeHelper, StudyBuddy, PetPal) powered by ChatGPT that serve native ads mid-conversation via PromptBid's real-time auction.

## How It Works

1. User sends a question to their chosen vertical
2. Backend calls ChatGPT for the AI response with vertical-specific system prompt
3. Backend fires a bid request to PromptBid with conversation context (categories, keywords)
4. If an ad wins the auction, it's returned alongside the AI response
5. Frontend renders the ad as a native card in the conversation

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY=sk-...              # ChatGPT API key from OpenAI
export PROMPTBID_API_KEY=pb_live_...      # Builder API key from PromptBid
export PROMPTBID_BASE_URL=http://localhost:8080  # Or https://promptbiddev.onrender.com
```

The app requires `OPENAI_API_KEY`. The app gracefully handles missing PromptBid key (ads won't display).

### 3. Run the server

```bash
python server.py
```

Open http://localhost:3001 in your browser.

### 4. Set up ads (for end-to-end testing)

You can use the setup script to automatically create test advertiser/builder accounts and a demo campaign:

```bash
python setup.py
```

Or manually:

1. Start the PromptBid API (`python run.py` from the project root, port 8080)
2. Register a builder account at `/register-builder` → get `pb_live_` API key
3. Register an advertiser account at `/register-advertiser` → get `ad_live_` API key
4. Create a campaign targeting food/cooking categories via the advertiser dashboard
5. Fund the advertiser account
6. Set `PROMPTBID_API_KEY` to your builder key and restart AskChef

## Architecture

```
User → AI Assistant UI (browser) → FastAPI backend → ChatGPT API
                                           ↓
                                     PromptBid API
                                    (bid request)
                                           ↓
                                   Native Ad → UI
```

## Verticals

Eight AI assistants, each with category-specific ad targeting:

1. **AskChef** (🍳) — Food & Drink (IAB8)
2. **FitCoach** (💪) — Sports, Exercise, Nutrition (IAB17)
3. **TravelGuru** (✈️) — Travel & Tourism (IAB20)
4. **MoneyMentor** (💰) — Personal Finance (IAB13)
5. **CodePilot** (⚡) — Technology (IAB19)
6. **HomeHelper** (🏠) — Home & Garden (IAB10)
7. **StudyBuddy** (📚) — Education (IAB5)
8. **PetPal** (🐾) — Pets (IAB16)

## Ad Integration Details

- **Format**: OpenRTB 2.6 native ads
- **Targeting**: Category + keywords extracted from conversation
- **Frequency**: Ads shown roughly every 3rd exchange to keep the experience natural
- **Tracking**: Impression pixels fire on ad visibility, clicks tracked via PromptBid API
