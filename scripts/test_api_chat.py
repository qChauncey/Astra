"""Quick test of the running Astra API endpoint."""
import httpx, json, sys, time

url = "http://localhost:8080/v1/chat/completions"
payload = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10,
    "stream": False,
}

for attempt in range(1, 4):
    try:
        r = httpx.post(url, json=payload, timeout=120)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        print(f"Attempt {attempt}: {e}")
        time.sleep(5)

print("FAILED: could not reach API after 3 attempts")
sys.exit(1)