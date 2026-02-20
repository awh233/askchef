# AskChef — AI Cooking Assistant with PromptBid Ads

A demo app showing the complete PromptBid SDK integration. AskChef is an AI cooking assistant powered by Claude or ChatGPT that serves native ads mid-conversation via PromptBid's real-time auction.

## How It Works

1. User sends a cooking question
2. Backend calls Claude or ChatGPT for the AI response
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
export ANTHROPIC_API_KEY=sk-ant-...       # For Claude responses
export OPENAI_API_KEY=sk-...              # For ChatGPT responses
export PROMPTBID_API_KEY=pb_live_...      # Builder API key from PromptBid
export PROMPTBID_BASE_URL=http://localhost:8080  # Or https://promptbiddev.onrender.com
```

You need at least one AI key (Claude or ChatGPT). The app gracefully handles missing keys.

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
User → AskChef UI (browser) → FastAPI backend → Claude/ChatGPT APIs
                                    ↓
                              PromptBid API
                             (bid request)
                                    ↓
                          Native Ad → UI
```

## Ad Integration Details

- **Format**: OpenRTB 2.6 native ads
- **Targeting**: Food & Drink categories (IAB8), keywords extracted from conversation
- **Frequency**: Ads shown roughly every 3rd exchange to keep the experience natural
- **Tracking**: Impression pixels fire on ad visibility, clicks tracked via PromptBid API
