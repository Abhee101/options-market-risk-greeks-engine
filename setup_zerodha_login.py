"""
Zerodha Kite Connect Daily Authentication & Session Generator.

Automates the OAuth login flow to generate the daily `access_token`:
1. Reads API Key & API Secret from `.env`.
2. Generates the Zerodha authorization URL.
3. Accepts either the redirected URL or the raw `request_token`.
4. Calls KiteConnect to generate and validate the active session.
5. Saves the new `KITE_ACCESS_TOKEN` directly into `.env`.
6. Tests the broker connection by fetching the trader's profile.
"""

import os
import re
import sys
from kiteconnect import KiteConnect


def load_env(filepath=".env"):
    env_vars = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars


def save_env(env_vars, filepath=".env"):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Zerodha Kite Connect Credentials (NEVER COMMIT TO GIT)\n")
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")


def main():
    print("======================================================================")
    print("   🔑 ZERODHA KITE CONNECT - DAILY SESSION GENERATOR")
    print("======================================================================")

    env_vars = load_env()
    api_key = env_vars.get("KITE_API_KEY", "")
    api_secret = env_vars.get("KITE_API_SECRET", "")

    if not api_key or not api_secret:
        print("❌ Error: KITE_API_KEY or KITE_API_SECRET missing in .env file.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"\n1. Open this URL in your web browser:\n")
    print(f"👉 \033[94m{login_url}\033[0m\n")
    print("2. Log in with your Zerodha User ID, Password, and TOTP/PIN.")
    print("3. After login, your browser will redirect to your Redirect URL.")
    print("   (e.g., http://127.0.0.1:5000/?action=login&status=success&request_token=XXXXXX)")
    print("======================================================================")

    try:
        user_input = input("\nPaste the FULL Redirected URL or the Request Token here:\n> ").strip()
    except EOFError:
        print("\nNo input received. Exiting.")
        sys.exit(0)

    if not user_input:
        print("❌ No input provided. Exiting.")
        sys.exit(1)

    # Extract request_token
    token_match = re.search(r"request_token=([a-zA-Z0-9]+)", user_input)
    if token_match:
        request_token = token_match.group(1)
    else:
        # User pasted token directly
        request_token = user_input.split("&")[0].split("=")[-1].strip()

    print(f"\n✓ Extracted Request Token: {request_token[:6]}...{request_token[-4:]}")
    print("Authenticating with Zerodha Kite Connect...")

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        user_name = data.get("user_name", "Trader")
        user_id = data.get("user_id", "")

        print(f"\n🎉 Authentication Successful!")
        print(f"   Trader: {user_name} ({user_id})")
        print(f"   Access Token: {access_token[:6]}...{access_token[-4:]}")

        # Save to .env
        env_vars["KITE_ACCESS_TOKEN"] = access_token
        save_env(env_vars)
        print("✓ Updated .env with daily KITE_ACCESS_TOKEN.")

        # Test profile fetch
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"✓ Broker Link Verified: Connected to account {profile.get('user_id')}.")

        print("\n🚀 You can now start the real-time engine:")
        print("   python live_greeks_to_excel.py")

    except Exception as e:
        print(f"\n❌ Authentication Failed: {e}")
        print("Tips: Ensure the request_token is fresh (valid for only a few minutes) and matches the API Key.")


if __name__ == "__main__":
    main()
