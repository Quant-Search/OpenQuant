import os
import argparse
from openquant.utils.alerts import send_alert

def main():
    p = argparse.ArgumentParser(description="Test WhatsApp Notification via Webhook")
    p.add_argument("--url", type=str, help="Webhook URL (e.g., CallMeBot or similar)", required=True)
    p.add_argument("--message", type=str, default="Hello from OpenQuant Robot!", help="Message to send")
    args = p.parse_args()
    
    print(f"Sending alert to {args.url}...")
    try:
        send_alert(
            subject="OpenQuant Test",
            body=args.message,
            severity="INFO",
            webhook_url=args.url
        )
        print("Alert sent (check your WhatsApp).")
    except Exception as e:
        print(f"Failed to send alert: {e}")

if __name__ == "__main__":
    main()
