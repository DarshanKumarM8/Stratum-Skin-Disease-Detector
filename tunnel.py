from pyngrok import ngrok, conf
import time

# Configure ngrok to use the latest version
conf.get_default().region = "us"

# Open a ngrok tunnel to the Flask server
try:
    public_url = ngrok.connect(5000, bind_tls=True)
    print(f"\n{'='*60}")
    print(f"🚀 Public URL: {public_url}")
    print(f"{'='*60}\n")
    print(f"Your application is now accessible at: {public_url}")
    print("Press Ctrl+C to stop the tunnel...")
    
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nShutting down tunnel...")
    ngrok.kill()
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTo fix ngrok authentication error:")
    print("1. Sign up at https://dashboard.ngrok.com/signup")
    print("2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken")
    print("3. Run: ngrok config add-authtoken YOUR_TOKEN")
    print("\nAlternatively, use localtunnel: npm install -g localtunnel && lt --port 5000")
