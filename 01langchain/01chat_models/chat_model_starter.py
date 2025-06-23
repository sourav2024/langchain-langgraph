from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is square root of 49")

print(result.content)


