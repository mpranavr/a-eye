import requests


API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
headers = {"Authorization": "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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

audio_bytes = query({
	"inputs": "input from caption",
})

from IPython.display import Audio
Audio(audio_bytes, autoplay=True)

