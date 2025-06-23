from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

#Create a modal
model = ChatOpenAI(model="gpt-4")

#Define a prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You love facts and you tell facts about {animal}."),
    ("human", "Tell me {count} facts")
])

#create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x:model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

#Create the RunnableSequence (equivalnet to the LCEL chain)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response =  chain.invoke({"animal": "cat", "count": 2})