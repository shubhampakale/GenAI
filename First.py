from google import genai

api_key = "AIzaSyDMBCJtnR5D8s6i-DQSgF-6Wg8AWZWFhNc"  # Replace with your actual API key
client = genai.Client(api_key=api_key)
chat = client.chats.create(model="gemini-2.5-flash")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    response = chat.send_message(user_input)
    print("Model:", response.text)

print("\nConversation history:")
for message in chat.get_history():
    print(f'role - {message.role}: {message.parts[0].text}')