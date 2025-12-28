from google import genai

api_key = "AIzaSyDMBCJtnR5D8s6i-DQSgF-6Wg8AWZWFhNc"

client = genai.Client(api_key=api_key)

# Initialize conversation history
conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    conversation.append({"role": "user", "parts": [{"text": user_input}]})
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation
    )
    model_output = response.text
    print("Model:", model_output)
    conversation.append({"role": "model", "parts": [{"text": model_output}]})