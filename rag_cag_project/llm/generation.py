from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()  

prompt_string="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_string)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

def generate_answer(query, context):
    llm=get_llm()
    response = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "question": query,
        "context": context
    })
    return response
