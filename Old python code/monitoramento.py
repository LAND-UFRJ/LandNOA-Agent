import psycopg2
# Este módulo centraliza integração com LLM, métricas (Postgres) e histórico de conversa (MongoDB).
from datetime import datetime
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import uuid
from pymongo import MongoClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re

MONGO_HOST = "10.246.47.194"  # Host do MongoDB usado para histórico
MONGO_PORT = 27017  # Porta padrão do MongoDB
MONGO_DB = "chat_history"  # Base que armazena sessões e mensagens

# Connection without authentication
client = MongoClient(f"mongodb://{MONGO_HOST}:{MONGO_PORT}/")


# Access a specific database
db = client[MONGO_DB]
collection_name = "chat_history"
collection = db[collection_name]


# Configuração do LLM (usado por diversos agentes)
LLM_BASE_URL = 'http://10.246.47.184:10000/v1'
LLM_API_KEY = 'asd'
LLM_MODEL = 'qwen2.5:14b'


def make_llm(temperature: float | None = None, seed: int | None = None) -> ChatOpenAI:
    """Fábrica de clientes LLM.
    - temperature/seed controlam variação da saída
    - top_p definido explicitamente (evita warnings)
    """
    temp_value = float(temperature) if temperature is not None else None
    # Passe top_p explicitamente em vez de model_kwargs para evitar warnings
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=SecretStr(LLM_API_KEY),
        model=LLM_MODEL,
        temperature=temp_value,
        seed=int(seed) if seed is not None else None,
        top_p=0.95,
    )


llm = make_llm()  # cliente padrão (sem temperatura/seed) para utilidades gerais


def extract_metrics(call):
    """Extrai métricas do retorno do LLM (modelo e uso de tokens)."""
    try:
        model = call.response_metadata['model_name']
        input_tokens = call.response_metadata['token_usage']['prompt_tokens']
        output_tokens = call.response_metadata['token_usage']['completion_tokens']
        return model,input_tokens,output_tokens
    except Exception as e:
        print('data not exported due to:')
        print(e)

def metrics(call):
    """Armazena métricas básicas em Postgres (se disponíveis)."""
    call = extract_metrics(call)
    if not call:
        # Nada para registrar
        return
    conn = psycopg2.connect(
        dbname="ollama_stats",
        user="land",
        password="landufrj123",
        host="10.246.47.171",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO metrics_table (timestamp, model, input_tokens, output_tokens)
        VALUES (to_timestamp(%s), %s, %s, %s)
    """, (int(datetime.now().timestamp()), *call))
    conn.commit()
    cur.close()


def append_message(call,sender_agent_id,agent_id,uuid,ip='teste'):
    """Append a message, creating document if it doesn't exist"""
    try:
        content = call.content
    except AttributeError:
        try:
            content = call['content']
        except (KeyError, TypeError):
            content = call

    new_message = {
        'sender': sender_agent_id,
        'message': content,
        'destination': agent_id,
        'timestamp':  datetime.now().timestamp()
    }
    update_result = collection.update_one(
        {'uuid': uuid},  # filter to find the document
        {
            '$setOnInsert': {'ip': ip},  # only set IP if creating new doc
            '$push': {'messages': new_message}  # append operation
        },
        upsert=True  # create document if it doesn't exist
    )
    
    if update_result.upserted_id:
        print(f"Created new document with uuid: {uuid}")
    else:
        print(f"Appended to existing document with uuid: {uuid}")


def get_context(uuid):
    """Recupera histórico da sessão; se grande, faz sumário via LLM."""
    doc = collection.find_one({'uuid':uuid})
    if not doc:
        return []
    resultado = doc.get('messages', [])
    if isinstance(resultado, list) and len(resultado) > 5:
        sumary = llm.invoke(str([item.get('message') for item in resultado if isinstance(item, dict)]))
        return sumary.content if hasattr(sumary, 'content') else str(sumary)
    return resultado

def remove_between_rag_tags(input_string):
    # Use regular expression to remove everything between <RAG> tags (including the tags themselves)
    output_string = re.sub(r'<RAG>.*?<\/RAG>', '', input_string, flags=re.DOTALL)
    return output_string



def invoke_model(prompt):
    """Invoca o modelo padrão e retorna a resposta com métricas."""
    answer = llm.invoke(prompt)
    metrics(answer)
    return answer


def invoke_model_with_temperature(prompt, temperature: float, seed: int | None = None):
    """Invoca o modelo com temperatura customizada (e seed opcional) e retorna a resposta com métricas."""
    local_llm = make_llm(temperature, seed)
    answer = local_llm.invoke(prompt)
    metrics(answer)
    return answer

def append_conversation(message, sender_id, receiver_id, session_uuid, ip='teste'):
    """Appends a message to conversation history"""
    # Handle different message types
    content = message.content if hasattr(message, 'content') else message
    
    new_message = {
        'sender': sender_id,
        'message': content,
        'destination': receiver_id,
        'timestamp': datetime.now().timestamp()
    }
    
    update_result = collection.update_one(
        {'uuid': session_uuid},
        {
            '$setOnInsert': {'ip': ip},
            '$push': {'messages': new_message}
        },
        upsert=True
    )
    
    if update_result.upserted_id:
        print(f"Created new session: {session_uuid}")
    else:
        print(f"Updated session: {session_uuid}")

