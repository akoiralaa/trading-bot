import requests

API_KEY = "your_api_key_here"  # Replace with your key

params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": "QQQ",
    "apikey": API_KEY,
    "outputsize": "full"
}

response = requests.get("https://www.alphavantage.co/query", params=params)
data = response.json()

print("Response status:", response.status_code)
print("Keys in response:", list(data.keys()))
print("\nFull response (first 500 chars):")
print(str(data)[:500])
