from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

#model
model = ChatOpenAI(model="gpt-4")

#Define prompt template for movie summary 
summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic"),
    ("human", "Provide a brief summary of the movie {movie_name}")
])

#Define plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("sytem", "You are a movie critic"),
        ("human", "Analyze the plot: {plot}. What are its strength and weakness")
    ])

    return plot_template.format_prompt(plot=plot)

#Define character anaylsis step
def analyze_character(characters):
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic"),
        ("human", "Analyze the characters: {characters}. What are their strenghts and weakness?" )

    ])
    return character_template.format_prompt(characters=characters)

#Combine analyses into a final verdict
def combine_verdicts(plot_analysis, chracter_analysis):
    return f"Plot Analysis: \n{plot_analysis}\n\nCharacter Analysis: \n {chracter_analysis}"

#simplfiy branches with LCEL
plot_branch_chain = {
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
}

character_branch_chain = {
    RunnableLambda(lambda x: analyze_character(x)) | model | StrOutputParser()
}

#create the combined chain using langchain exporesssion langauge
chain = {
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
}

result = chain.invoke({"movie_name": "Inception"})

print(result)