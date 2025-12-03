import os
import requests
import json

url = "https://j5hdd8bc9cd5cjhjjk5ebob85hgppdpc.openapi-qb-ai.sii.edu.cn/v1/chat/completions"

# 你的 API key，等价于 curl 中的 $INF_API_KEY
api_key = "c7VO4wrfPhh2G22xyArMRq6jh8aAyIQj1wR/pTD2nDg="

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

# 如果需要发送 JSON body，在这里填
payload = {
    "model": "/inspire/hdd/project/embodied-multimodality/public/pywang/share/models/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
}

print("Start posting...")

response = requests.post(url, headers=headers, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)