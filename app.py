import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage


#Carrega variáveis de ambiente do arquivo .env
load_dotenv()

#Variáveis de ambiente - chaves de acesso
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not GROQ_API_KEY or not OPEN_AI_API_KEY:
    raise ValueError("As chaves de API não foram definidas corretamente")

# Caminho do arquivo PDF e do banco de dados Chroma
pdf_path = './db/normas.pdf'
chroma_path = './db/chroma'


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


#app config
st.set_page_config(page_title="Chatbot de Vestibular", page_icon="🎓")
st.title("Chatbot de Vestibular")

        



