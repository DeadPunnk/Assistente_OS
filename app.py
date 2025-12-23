import streamlit as st
import os
from dotenv import load_dotenv

# Imports do LangChain (usando as versões estáveis para evitar erros de ModuleNotFound)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração da página Streamlit
st.set_page_config(page_title="🤖 AI Chat RAG", layout="centered")
st.title("🤖 Assistente de ordens de serviço (Ollama)")



# 1. FUNÇÃO PARA CARREGAR E PROCESSAR (Teu novo trecho)
def preparar_documentos(pasta):
    documentos = []
    if not os.path.exists(pasta):
        st.error(f"Pasta não encontrada: {pasta}")
        return []
        
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.endswith(".txt"):
            documentos.extend(TextLoader(caminho).load())
        elif arquivo.endswith(".pdf"):
            documentos.extend(PyPDFLoader(caminho).load())
        elif arquivo.endswith(".docx"):
            documentos.extend(UnstructuredWordDocumentLoader(caminho).load())
    
    # Teu trecho de Splitter melhorado
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documentos)




# --- INICIALIZAÇÃO DO BACKEND ---
@st.cache_resource # Garante que o banco e o modelo carreguem apenas uma vez
def inicializar_rag():
    load_dotenv()

    pasta_data = "documentos/"
    
    # 1. Embeddings (Locais)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Carregar o banco Chroma existente
    #if os.path.exists("db_chroma"):
    #    vectordb = Chroma(persist_directory="db_chroma", embedding_function=embeddings)
    #else:
    #    st.error("Erro: Banco de dados 'db_chroma' não encontrado. Execute o script de criação primeiro.")
    #    return None

    # Verifica se o banco já existe, senão cria com os teus documentos
    if not os.path.exists("db_chroma_ai"):
        with st.spinner("A processar novos documentos..."):
            chunks = preparar_documentos(pasta_data)
            vectordb = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory="db_chroma_ai"
            )
    else:
        vectordb = Chroma(persist_directory="db_chroma", embedding_function=embeddings)


    # 3. Configurar o LLM (Ollama)
    llm = ChatOllama(model="qwen3:4b", temperature=0.4) # Ajustado para versão estável do Qwen
    
    # 4. Setup da Chain
    system_prompt = (
        "Você é um Assitente administrativo de projetos especialista em análise de documentos técnicos e atas de reunião. "
         "Use estritamente os trechos de contexto fornecidos abaixo para responder à pergunta do usuário. "
         "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação nos documentos. "
         "Ao mencionar datas ou nomes de documentos, formate-os em negrito."
         "\n\n" 
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Inicializa a chain
rag_chain = inicializar_rag()

# --- INTERFACE DE CHAT ---

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if user_input := st.chat_input("Pergunte algo sobre os documentos..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gera resposta usando a RAG Chain
    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            if rag_chain:
                response = rag_chain.invoke({"input": user_input})
                full_response = response["answer"]
                
                # Exibe resposta
                st.markdown(full_response)
                
                # Opcional: Mostrar fontes (metadados)
                with st.expander("Ver fontes consultadas"):
                    for doc in response["context"]:
                        st.write(f"- {doc.metadata.get('source', 'Desconhecido')}")
            else:
                full_response = "Erro ao carregar o sistema RAG."
                st.error(full_response)

    # Salva no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})