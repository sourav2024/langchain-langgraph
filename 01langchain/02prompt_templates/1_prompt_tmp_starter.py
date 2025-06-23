from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatOpenAI(model="gpt-4")

# Step 1 create a string  template (We can understand)
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

#Step 2 create a string template langchain can start
prompt_template = ChatPromptTemplate.from_template(template)

# assign values to variables
prompt = prompt_template.invoke({
  "tone": "energetic",
  "company": "samsung",
  "position": "AI Engineer",
  "skill": "AI"
})

result = llm.invoke(prompt)
print(result.content)

