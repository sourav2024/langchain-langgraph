from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

# model
model = ChatOpenAI(model="gpt-4")

#Define promplt template for different feedback
positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}")
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this negative feedback: {feedback}")
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human","Generate a request for more details in this neutral feedback: {feedback}")
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human","Generate a message to escalate this feedback to a human agent: {feedback}")
])

#Define the feedback classification template

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}")

])

#branching
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "escalate" in x,
        escalate_feedback_template | model | StrOutputParser()
    ),
)

classfication_chain = classification_template | model | StrOutputParser()

chain = classfication_chain| branches

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

print(result)
