from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on following text. /n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2

result = chain.invoke({'topic':'black hole'})

print(result)