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

#Carrega e divide o documento PDF, inicializa embeddings, e banco de dados Chromas
def get_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=10, add_start_index=True)
    docs_chunks = text_splitter.split_documents(docs)

    chroma_db = Chroma.from_documents(docs_chunks, OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY), persist_directory=chroma_path)
    return chroma_db

def get_context_retriever_chain(chroma_db, system_prompt, question):
    llm = ChatGroq(model="llama3-70b-8192")

    # Configurar o recuperador de documentos
    retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history")]
        [
            ("system", system_prompt),
            ("human", question),
        ]
    )  
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def get_conversational_rag_chain(retriever_chain, system_prompt, user_input):
    llm  = ChatGroq(model="llama3-70b-8192")

    prompt = ChatPromptTemplate.from_messages([
        {"system": system_prompt},
        MessagesPlaceholder(variable_name="chat_history"),
        {"user": user_input},
    ])

    stuff_docments_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_docments_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.chroma_db)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "user": user_input,
    })

    return response["answer"]

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
#User input
question = st.chat_input("Digite a pergunta:")




