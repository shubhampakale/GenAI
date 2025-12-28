from google import genai

api_key = "AIzaSyDMBCJtnR5D8s6i-DQSgF-6Wg8AWZWFhNc"

# Pass the API key to the client
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents= [
        {
            "role": "user",
            "parts": [{"text": "Hi I am Shubh" }]
        },
        {
            "role": "model",
            "parts": [{"text": "Hi Shubh! It's nice to meet you. How can I help you today?" }]
        },
        {
            "role": "user",
            "parts": [{"text": "What is my Name?" }]
        }
    ]
)
print(response.text)