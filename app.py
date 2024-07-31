import os
import json
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
import json
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


#Carrega variáveis de ambiente do arquivo .env
load_dotenv()

#Variáveis de ambiente - chaves de acesso
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

if not GROQ_API_KEY or not OPEN_AI_API_KEY:
    raise ValueError("As chaves de API não foram definidas corretamente")

#Carrega e divide o documento PDF
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=10, add_start_index=True)
    return text_splitter.split_documents(docs)

#Inicializar embeddings e banco de dados Chromas
def initialize_chroma(chunks, api_key, persist_directory):
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma.from_documents(chunks, embedding_function, persist_directory=persist_directory)

# Principal fluxo de execução
def main():
    # Caminho do arquivo PDF e do banco de dados Chroma
    pdf_path = './db/normas.pdf'
    chroma_path = './db/chroma'

    # Carregar e dividir documentos
    chunks = load_and_split_documents(pdf_path)

    # Inicializar Chroma
    db_chroma = initialize_chroma(chunks, OPEN_AI_API_KEY, chroma_path)

    # Configurar o recuperador de documentos
    retriever = db_chroma.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    #Extrair metadatos
    schema = {
        "properties": {
            "Capítulo": {"type": "string"},
            "Página": {"type": "integer"},
            "Palavra-chave": {"type": "string"},
            "Curso": {"type": "string"},
            "Assunto": {"type": "string"},
        },
        "required": ["Palavra-chave", "Curso", "Assunto"]
    }

    # Carregar o prompt e o modelo
    system_prompt = (
    "Você é um assistente para perguntas e respostas de um vestibular de uma universidade."
    "Utilize os seguintes chunks {context} para responder as perguntas sobre o processo seletivo."
    "Responda a pergunta baseada no contexto acima. "
    "Responda com o máximo de detalhes possíveis. "
    "Não justifique as respostas."
    "Não dê informações que não sejam baseadas pelo contexto."
    "Contexto: {context}"
    "\n\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", question),
        ]
    )  
    
    llm = ChatGroq(model="llama3-70b-8192")
    #llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPEN_AI_API_KEY)
    question = input("Digite a pergunta: ")
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    print(response["answer"])
    
    
    #app config
    st.set_page_config(page_title="Chatbot de Vestibular", page_icon="🎓")
    st.title("Chatbot de Vestibular")
    #User input
    user_query = st.chat_input("Digite sua pergunta: ")
    retriever = retriever.invoke(user_query)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")


