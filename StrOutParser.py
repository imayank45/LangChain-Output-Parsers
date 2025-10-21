from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

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

prompt1 = template1.invoke({'topic':'LLM Poisoning'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result2)