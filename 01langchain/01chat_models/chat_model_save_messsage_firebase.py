from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

from langchain_openai import ChatOpenAI

"""
Steps to integrate firestore to store message history
1. Create a Firebase Account
2. Create a new firebase project
- Copy the project ID
3. Create a firestore database in the firebase project
4. Install the google cloud CLI on your computer
- https://cloud.google.com/sdk/docs/install
- Authenticate the Google Cloud CLI with your Google account
 - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
 - Set your default project to the new Firebase project you created
5. Enable the firestore API in the google cloud console
 - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
  
"""

load_dotenv()

#Setup firebse firestore
PROJECT_ID = "firebase-projectID"
SESSION_ID = "user_session" #This could be a username or a unique ID
COLLECTION_NAME= "chat_history"

#Intialize FireStore Client
print("Initializing FireStore Client...")
client = firestore.Client(project=PROJECT_ID)

#Intiliaze Firestore chat message history
print("Initializing Firestore chat message history...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)
print("Chat History Initialzied")
print("Current chat history: ", chat_history.messages)

#Intialize chat Model
model = ChatOpenAI()

print("Start chatting with the AI. Type exit to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")