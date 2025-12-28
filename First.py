from google import genai

# Store the API key in a variable
api_key = "AIzaSyDMBCJtnR5D8s6i-DQSgF-6Wg8AWZWFhNc"  # Replace with your actual API key

# Pass the API key to the client
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how machine learning works in a few words"
)
print(response.text)