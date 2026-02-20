"""
AskChef Demo Setup Script

Registers a test builder + advertiser on PromptBid, funds the advertiser,
creates a food-targeted campaign, and prints the env vars you need.

Usage:
    export PROMPTBID_BASE_URL=http://localhost:8080  # default
    python setup.py
"""

import asyncio
import json
import sys
import os
import secrets
import string

import httpx

BASE_URL = os.getenv("PROMPTBID_BASE_URL", "http://localhost:8080")

# Test account credentials
BUILDER_EMAIL = "askchef-builder@demo.test"
BUILDER_PASSWORD = "DemoBuilder123!"
BUILDER_APP_NAME = "AskChef"

ADVERTISER_EMAIL = "askchef-advertiser@demo.test"
ADVERTISER_PASSWORD = "DemoAdvertiser123!"
ADVERTISER_COMPANY = "HelloFresh Demo"

FUND_AMOUNT = 100.00  # $100 demo budget


def random_suffix():
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))


async def main():
    print("=" * 60)
    print("  AskChef Demo Setup")
    print(f"  PromptBid API: {BASE_URL}")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15) as client:

        # ── 1. Health check ──
        print("[1/6] Checking PromptBid API...")
        try:
            resp = await client.get("/health")
            if resp.status_code != 200:
                print(f"  ✗ API returned {resp.status_code}. Is PromptBid running on {BASE_URL}?")
                sys.exit(1)
            print("  ✓ API is healthy")
        except httpx.ConnectError:
            print(f"  ✗ Cannot connect to {BASE_URL}. Start PromptBid first: python run.py")
            sys.exit(1)

        # ── 2. Register builder ──
        print("\n[2/6] Registering builder account...")
        suffix = random_suffix()
        builder_email = f"askchef-builder-{suffix}@demo.test"
        resp = await client.post("/api/v1/builders/register", json={
            "email": builder_email,
            "password": BUILDER_PASSWORD,
            "app_name": BUILDER_APP_NAME,
            "app_url": "http://localhost:3001",
            "app_category": "Food & Drink",
        })
        if resp.status_code in (200, 201):
            builder_data = resp.json()
            builder_api_key = builder_data.get("api_key", "")
            builder_id = builder_data.get("id", builder_data.get("builder_id", ""))
            print(f"  ✓ Builder registered: {builder_email}")
            print(f"    API Key: {builder_api_key}")
        else:
            print(f"  ✗ Registration failed ({resp.status_code}): {resp.text[:200]}")
            print("  Trying to login instead...")
            resp = await client.post("/api/v1/builders/login", json={
                "email": BUILDER_EMAIL,
                "password": BUILDER_PASSWORD,
            })
            if resp.status_code == 200:
                builder_data = resp.json()
                builder_api_key = builder_data.get("api_key", "")
                builder_id = builder_data.get("id", builder_data.get("builder_id", ""))
                print(f"  ✓ Logged in as existing builder")
            else:
                print(f"  ✗ Login also failed. Check the API.")
                builder_api_key = ""
                builder_id = ""

        # ── 3. Register advertiser ──
        print("\n[3/6] Registering advertiser account...")
        advertiser_email = f"askchef-advertiser-{suffix}@demo.test"
        resp = await client.post("/api/v1/advertisers/register", json={
            "email": advertiser_email,
            "password": ADVERTISER_PASSWORD,
            "company_name": ADVERTISER_COMPANY,
            "website": "https://www.hellofresh.com",
            "industry": "Food & Beverage",
        })
        if resp.status_code in (200, 201):
            adv_data = resp.json()
            adv_api_key = adv_data.get("api_key", "")
            adv_id = adv_data.get("id", adv_data.get("advertiser_id", ""))
            print(f"  ✓ Advertiser registered: {advertiser_email}")
            print(f"    API Key: {adv_api_key}")
        else:
            print(f"  ✗ Registration failed ({resp.status_code}): {resp.text[:200]}")
            print("  Trying to login instead...")
            resp = await client.post("/api/v1/advertisers/login", json={
                "email": ADVERTISER_EMAIL,
                "password": ADVERTISER_PASSWORD,
            })
            if resp.status_code == 200:
                adv_data = resp.json()
                adv_api_key = adv_data.get("api_key", "")
                adv_id = adv_data.get("id", adv_data.get("advertiser_id", ""))
                print(f"  ✓ Logged in as existing advertiser")
            else:
                print(f"  ✗ Login also failed. Check the API.")
                adv_api_key = ""
                adv_id = ""

        # ── 4. Fund advertiser ──
        if adv_api_key:
            print(f"\n[4/6] Funding advertiser account (${FUND_AMOUNT:.2f})...")
            resp = await client.post(
                "/api/v1/payments/deposit",
                headers={"X-API-Key": adv_api_key},
                json={"amount": FUND_AMOUNT},
            )
            if resp.status_code in (200, 201):
                print(f"  ✓ Deposited ${FUND_AMOUNT:.2f}")
            else:
                print(f"  ⚠ Deposit returned {resp.status_code}: {resp.text[:200]}")
                print("    (The advertiser may already have funds)")
        else:
            print("\n[4/6] Skipping funding — no advertiser API key")

        # ── 5. Create campaign ──
        if adv_api_key:
            print("\n[5/6] Creating food/cooking demo campaign...")
            campaign_payload = {
                "name": "HelloFresh — AskChef Demo",
                "daily_budget": 50.0,
                "total_budget": 500.0,
                "bid_amount": 2.0,
                "status": "active",
                "targeting": {
                    "categories": ["IAB8", "IAB8-5", "IAB8-9"],
                    "keywords": ["cooking", "recipes", "food", "kitchen", "meal", "dinner", "healthy", "ingredients"],
                },
                "creative": {
                    "headline": "Fresh Ingredients, Delivered Weekly",
                    "description": "Get chef-curated recipes and pre-portioned ingredients delivered to your door. First box 60% off!",
                    "cta_text": "Get 60% Off",
                    "cta_url": "https://www.hellofresh.com/demo",
                    "image_url": "",
                },
            }
            resp = await client.post(
                "/api/v1/campaigns/",
                headers={"X-API-Key": adv_api_key},
                json=campaign_payload,
            )
            if resp.status_code in (200, 201):
                campaign = resp.json()
                campaign_id = campaign.get("id", campaign.get("campaign_id", ""))
                print(f"  ✓ Campaign created (ID: {campaign_id})")
                print(f"    Headline: {campaign_payload['creative']['headline']}")
                print(f"    Targeting: {', '.join(campaign_payload['targeting']['categories'])}")
            else:
                print(f"  ⚠ Campaign creation returned {resp.status_code}: {resp.text[:300]}")
        else:
            print("\n[5/6] Skipping campaign — no advertiser API key")

        # ── 6. Print summary ──
        print("\n" + "=" * 60)
        print("  Setup Complete!")
        print("=" * 60)
        print()
        print("Add these to your environment and start AskChef:")
        print()
        if builder_api_key:
            print(f"  export PROMPTBID_API_KEY={builder_api_key}")
        print(f"  export PROMPTBID_BASE_URL={BASE_URL}")
        print(f"  export ANTHROPIC_API_KEY=sk-ant-...  # your Claude key")
        print(f"  export OPENAI_API_KEY=sk-...          # your OpenAI key (optional)")
        print()
        print("  python server.py")
        print()
        print("Then open http://localhost:3001 and start chatting!")
        print()


if __name__ == "__main__":
    asyncio.run(main())
