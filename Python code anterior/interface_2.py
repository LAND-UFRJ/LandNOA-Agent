import streamlit as st
import requests
import os
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from jv import get_chroma_client

# Carrega as variáveis de ambiente do ficheiro .env
load_dotenv()

# --- CONFIGURAÇÃO A PARTIR DE VARIÁVEIS DE AMBIENTE ---
# A URL do host agora é lida do ambiente. O .env para execução local
# deve ter HOST_AGENT_URL=http://localhost:8000
HOST_AGENT_URL = os.getenv("HOST_AGENT_URL", "http://localhost:8000")
HOST_ENDPOINT = '/query'
UPLOAD_FOLDER = "uploaded_pdfs"

# Garante que a pasta de uploads exista
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

# --- FUNÇÕES DE LÓGICA ---
def ask_host(query: str):
    """
    Envia uma pergunta para o host_agent e retorna a resposta.
    """
    try:
        url = f'{HOST_AGENT_URL}{HOST_ENDPOINT}'
        payload = {
            "query": query,
            "uuid": st.session_state.session_uuid
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Retorna diretamente o conteúdo da chave 'answer'
        return response.json().get('answer', 'O agente não forneceu uma resposta no formato esperado.')
    except requests.exceptions.RequestException as e:
        error_msg = f"**Erro ao comunicar com o Host em `{HOST_AGENT_URL}`**\n\n*Detalhes:*\n{e}"
        return {"error": error_msg}

def save_pdf(file):
    """
    Salva um ficheiro PDF enviado na pasta de uploads local.
    """
    try:
        save_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        return {"status": "success", "message": f"PDF salvo como {file.name}", "path": save_path}
    except Exception as e:
        return {"error": f"Falha ao salvar PDF: {str(e)}"}

def update_metadata(file_path, area):
    """
    Cria ou atualiza um CSV com os metadados dos documentos enviados.
    """
    csv_path = os.path.join(UPLOAD_FOLDER, "document_metadata.csv")
    new_entry = {"path": file_path, "area": area, "indexed": False}
    try:
        try:
            df = pd.read_csv(csv_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame(columns=["path", "area", "indexed"])

        if not df[df['path'] == file_path].empty:
            df.loc[df['path'] == file_path, ['area', 'indexed']] = [area, False]
            st.warning("Metadados para este arquivo atualizados.")
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao atualizar metadados: {str(e)}")
        return False

# --- CONEXÃO COM O CHROMADB ---
# Garanta que a sua função em 'jv.py' também usa variáveis de ambiente.
# O seu .env local deve apontar para o IP correto do ChromaDB.
client = get_chroma_client()

# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["💬 Chat com Agentes", "📄 Upload de Documentos para RAG"])

# --- LÓGICA DO CHAT ---
with tab1:
    st.title("🤖 Sistema Multiagentes COPPE/UFRJ")

    # Exibe as mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do utilizador
    if prompt := st.chat_input("Digite a sua pergunta..."):
        # Adiciona e exibe a mensagem do utilizador
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtém e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Os agentes estão a trabalhar na sua resposta..."):
                response_content = ask_host(prompt)

                if isinstance(response_content, dict) and "error" in response_content:
                    st.error(response_content["error"])
                    st.session_state.messages.append({"role": "assistant", "content": response_content["error"]})
                else:
                    # Se a API retornou payload estendido, pode ser dict
                    if isinstance(response_content, dict):
                        text = response_content.get('answer') or ''
                        st.markdown(text)
                        # fontes usadas pelo RAG
                        sources = response_content.get('sources')
                        if sources:
                            st.info("Fontes utilizadas no RAG:\n- " + "\n- ".join(sources))
                        # debug/telemetria
                        meta = []
                        if response_content.get('chosen_temperature') is not None:
                            meta.append(f"Temperatura escolhida: {response_content['chosen_temperature']}")
                        if response_content.get('similarity') is not None:
                            meta.append(f"Similaridade: {response_content['similarity']:.4f}")
                        if meta:
                            st.caption(" | ".join(meta))
                        st.session_state.messages.append({"role": "assistant", "content": text})
                    else:
                        st.markdown(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- LÓGICA DE UPLOAD DE PDF ---
with tab2:
    st.title("📄 Upload e Gestão de Documentos para RAG")

    # --- BARRA LATERAL PARA GESTÃO DE COLEÇÕES ---
    st.sidebar.header("🗂️ Gerenciamento de Coleções")
    collections_names = []
    if client is not None:
        try:
            collections_names = [c.name for c in client.list_collections()]
        except Exception as e:
            st.sidebar.error(f"Erro ao listar coleções: {e}")
    else:
        st.sidebar.error("Cliente do ChromaDB não inicializado. Verifique as variáveis de ambiente e o serviço do ChromaDB.")

    action = st.sidebar.radio("Ação:", ("Usar coleção existente", "Criar nova coleção"))

    collection_name = None
    if action == "Usar coleção existente":
        if collections_names:
            collection_name = st.sidebar.selectbox("Escolha a coleção:", collections_names)
        else:
            st.sidebar.warning("Nenhuma coleção encontrada. Crie uma nova.")

    elif action == "Criar nova coleção":
        new_collection_name = st.sidebar.text_input("Nome da nova coleção:")
        if st.sidebar.button("Criar Coleção"):
            if new_collection_name:
                try:
                    if client is None:
                        st.sidebar.error("Cliente do ChromaDB não inicializado.")
                    else:
                        client.create_collection(name=new_collection_name)
                        st.sidebar.success(f"Coleção '{new_collection_name}' criada!")
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Erro ao criar coleção: {e}")
            else:
                st.sidebar.warning("Por favor, insira um nome.")

    if collection_name:
        st.info(f"Coleção selecionada para upload: **{collection_name}**")

    # --- FUNCIONALIDADE DE UPLOAD ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Enviar um novo documento")
        areas = ["Biologia", "LAND", "Guia de IA"]
        selected_area = st.selectbox("Selecione a área do documento:", areas)
        uploaded_file = st.file_uploader("Escolha um arquivo PDF:", type="pdf")

        if uploaded_file and collection_name:
            if st.button("Salvar e Registrar Documento"):
                with st.spinner("A salvar o ficheiro..."):
                    result = save_pdf(uploaded_file)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"PDF salvo em: `{result['path']}`")
                    if update_metadata(result["path"], selected_area):
                        st.success("Metadados do documento registrados com sucesso!")
        elif uploaded_file and not collection_name:
            st.warning("Por favor, selecione ou crie uma coleção na barra lateral antes de fazer o upload.")

    with col2:
        st.subheader("2. Documentos Registrados")
        try:
            pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
            if pdf_files:
                st.dataframe(pdf_files, use_container_width=True)
            else:
                st.info("Nenhum PDF armazenado ainda.")
        except Exception as e:
            st.error(f"Erro ao listar arquivos: {str(e)}")
