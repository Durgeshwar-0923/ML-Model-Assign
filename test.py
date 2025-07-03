import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Age": 35,
    "Experience": 10,
    "Income": 70,
    "Family": 2,
    "CCAvg": 1.5,
    "Education": 2,
    "Mortgage": 0,
    "Online": 1
}

res = requests.post(url, json=data)

print("Status:", res.status_code)
try:
    print("Output:", res.json())
except Exception as e:
    print("❌ Failed to decode JSON:", e)
    print("Raw response:", res.text)
