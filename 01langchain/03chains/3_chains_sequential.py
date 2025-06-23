from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

#model
model = ChatOpenAI(model="gpt-4")

#prompt
animal_facts_template = ChatPromptTemplate.from_messages([
    ("system", "You like telling facts and you tell facts about {animal}."),
    ("human", "Tell me {count} facts.")
])

#Define a prompt template for translation to french
translation_template = ChatPromptTemplate.format_messages([
    ("system", "You are a translator and convert the provided text into {language}."),
    ("human",  "Translate the following text to {language} : {text}"),
])

#Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n {x}")

prepare_for_translation = RunnableLambda(lambda output : {"text": output, "language": "french"})

#Create the combined chain using LangChain expression language (LCEL)
chain = animal_facts_template| model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "count": 2})

print(result)