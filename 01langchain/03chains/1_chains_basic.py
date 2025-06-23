from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#Create a chatopen at model
model = ChatOpenAI(model="gpt-4o")

#Define prompt templates ( no need for separate Runnable chains)
prompt_templates = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts export who knows facts about {animal}. "),
        ("human", "Tell me {fact_count} facts."),
    ]
)

#Create the combined chain using langchain expresssion Language (LCEL)

chain = prompt_templates | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "fact_count" : 2})

print(result)