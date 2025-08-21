"""
AI Guide Agent
----------------
Agente especialista focado em responder perguntas sobre o uso ético e responsável de IA.

Fluxo de alto nível:
- Recebe consultas do Host via /execute (A2A) protegidas por Bearer token
- Recupera contexto via RAG (ChromaDB) a partir da coleção 'Boas_Praticas_IA'
- Gera múltiplas respostas com temperaturas diferentes e escolhe a mais ancorada no contexto
- Registra histórico de conversas no Mongo (via monitoramento), sem quebrar o fluxo em caso de falha

Principais variáveis de ambiente utilizadas:
- AI_GUIDE_AGENT_ID, AI_GUIDE_AGENT_SECRET_TOKEN, AI_GUIDE_AGENT_BASE_URL
- REGISTRY_BASE_URL (para registro/heartbeat do agente)
- AI_GUIDE_FLASK_RUN_PORT (porta HTTP deste agente)
"""

import os
import requests
import time
import atexit
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from monitoramento import invoke_model, invoke_model_with_temperature, append_conversation, get_context
from sentence_transformers import SentenceTransformer
from rag_edson import ChromaDBRetriever
import threading


load_dotenv()

# Identificadores e configuração do agente (providos via .env)
AGENT_ID = os.getenv("AI_GUIDE_AGENT_ID")                         # ID único do agente
AGENT_SECRET_TOKEN = os.getenv("AI_GUIDE_AGENT_SECRET_TOKEN")     # Token para autenticação A2A
AGENT_BASE_URL = os.getenv("AI_GUIDE_AGENT_BASE_URL")             # URL pública para o Registry
REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL")                # URL do Registry (descoberta/heartbeat)
FLASK_RUN_PORT = int(os.getenv("AI_GUIDE_FLASK_RUN_PORT", 8010))  # Porta HTTP

app = Flask(__name__)

# RAG retriever (ChromaDB) e modelo de embeddings para verificação de ancoragem
retriever = ChromaDBRetriever()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Esquema público de ferramentas que o Host usa para roteamento
TOOLS_SCHEMA = [
    {
        "name": "responder_como_guia_de_ia",
        "description": "Use esta ferramenta para responder perguntas sobre o uso de IA de maneiras éticas e adequadas à uma educação saudável e responsável.",
        "parameters": {"pergunta": "string"     }
    }
]


# Prompt do sistema com grounding no contexto RAG, permitindo inferência cautelosa e evitando negações desnecessárias
SYSTEM_PROMPT = """
Você é um assistente de IA que responde com base no contexto fornecido.
Siga estas diretrizes:
1) Analise a pergunta do usuário e o <RAG>Contexto</RAG> com atenção.
2) Responda usando APENAS informações do contexto (sem conhecimento externo).
3) Quando a resposta não estiver totalmente explícita, FAÇA inferências cautelosas e bem fundamentadas a partir do que estiver no contexto, citando o que os trechos indicam e sinalizando incertezas com clareza (ex.: "Com base no contexto, os documentos indicam que...").
4) Evite negar ou dizer que não há informação quando existirem pistas, menções parciais ou termos relacionados no contexto. Priorize sintetizar evidências e produzir a melhor resposta provável, explicando o raciocínio.
5) Somente use a frase de falta de material se o contexto estiver vazio/não relacionado. Caso contrário, sempre produza a melhor resposta possível, ainda que com ressalvas.
Sua única fonte de verdade é o texto dentro das tags <RAG></RAG>.
Responda de forma sucinta, direta e em português do Brasil.
"""

def verificar_proximidade(resposta_gerada: str, documentos_contexto: list, limiar=0.35):
    """Determina se a resposta gerada está suficientemente próxima do conteúdo do RAG.
    Retorna True se a similaridade máxima for >= limiar (default 0.35).
    """
    if not documentos_contexto:
        return False
    embedding_resposta = np.asarray(embedding_model.encode(resposta_gerada, convert_to_tensor=False), dtype=float)
    embeddings_contexto = np.asarray(embedding_model.encode(documentos_contexto, convert_to_tensor=False), dtype=float)
    denom_r = np.linalg.norm(embedding_resposta) or 1.0
    denom_c = np.linalg.norm(embeddings_contexto, axis=1, keepdims=True)
    denom_c[denom_c == 0] = 1.0
    embedding_resposta = embedding_resposta / denom_r
    embeddings_contexto = embeddings_contexto / denom_c
    similaridades = np.dot(embeddings_contexto, embedding_resposta)
    maior_similaridade = np.max(similaridades)
    print(f"AI_GUIDE_AGENT: Maior similaridade encontrada: {maior_similaridade:.4f}")
    return maior_similaridade >= limiar


def calcular_similaridade_maxima(texto: str, documentos: list) -> float:
    """Calcula a similaridade máxima entre um texto e uma lista de documentos usando embeddings."""
    if not documentos:
        return 0.0
    try:
        e_text = np.asarray(embedding_model.encode(texto, convert_to_tensor=False), dtype=float)
        e_docs = np.asarray(embedding_model.encode(documentos, convert_to_tensor=False), dtype=float)
        e_text = e_text / (np.linalg.norm(e_text) + 1e-9)
        e_docs = e_docs / (np.linalg.norm(e_docs, axis=1, keepdims=True) + 1e-9)
        return float(np.max(np.dot(e_docs, e_text)))
    except Exception:
        return 0.0


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para o Docker verificar se o AI Guide Agent está online."""
    return jsonify({"status": "ok"}), 200


@app.route('/execute', methods=['POST'])
def execute_task():
    """Rota A2A: valida o token, executa RAG, gera respostas e retorna a melhor ancorada.
    Espera payload.query (pergunta do usuário) e payload.uuid (identificador de sessão).
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {AGENT_SECRET_TOKEN}":
        return jsonify({"error": "Acesso não autorizado."}), 403
    
    a2a_message = request.get_json()
    query = a2a_message.get("payload", {}).get("query")
    user_uuid = a2a_message.get("payload", {}).get("uuid")
    
    if not query:
        return jsonify({"error": "A 'query' não foi recebida."}), 400
    
    try:
        # Recupera contexto RAG e fontes (3 valores retornados)
        rag_context, source_names, raw_documents = retriever.retrieve_and_format(
            collection_name='Boas_Praticas_IA', 
            query_text=query,
            n_results=6
        )
        
        try:
            append_conversation(message=query, sender_id=a2a_message.get('sender_agent_id'), receiver_id=AGENT_ID, session_uuid=user_uuid)
        except Exception as log_err:
            print(f"AI_GUIDE_AGENT: aviso - falha ao registrar conversa (entrada usuário): {log_err}")
        
        # Constrói mensagens para o LLM (inclui histórico e contexto RAG)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=f'The history of this conversation is :\n{get_context(user_uuid)}'),
            # Passa o contexto já formatado (evita tags <RAG> duplicadas)
            SystemMessage(content=f"<RAG>Context:\n{rag_context}</RAG>"),
            HumanMessage(content=query)
        ]

        # Gera 3 respostas com diferentes temperaturas e mede ancoragem por similaridade
        temps = [0.3, 0.6, 0.9]
        candidates = []  # (text, is_anchor, score, temp)
        for idx, t in enumerate(temps):
            seed = int(time.time()) % 100000 + idx
            try:
                resp = invoke_model_with_temperature(messages, t, seed=seed)
                text = resp.content if hasattr(resp, 'content') else str(resp)
                if not isinstance(text, str):
                    text = str(text)
                score = calcular_similaridade_maxima(text, raw_documents)
                is_anchor = score >= 0.35
                candidates.append((text, is_anchor, score, t))
                print(f"AI_GUIDE_AGENT: temp={t} seed={seed} similarity={score:.4f} anchored={is_anchor}")
            except Exception as e:
                print(f"AI_GUIDE_AGENT: falha na temperatura {t} (seed={seed}): {e}")

        # Seleciona a melhor resposta entre as ancoradas; senão, usa resposta padrão
        resposta_padrao = "Não encontrei material sobre este tópico específico nas minhas diretrizes. Recomendo pesquisar em fontes confiáveis sobre educação e tecnologia."
        anchored = [c for c in candidates if c[1]]
        if anchored:
            best = max(anchored, key=lambda x: x[2])
            resposta_final, _, best_score, best_temp = best
        else:
            resposta_final, best_score, best_temp = resposta_padrao, 0.0, None
        print(f"AI_GUIDE_AGENT: escolhido temp={best_temp} similarity={best_score:.4f}")

        # Acrescenta linha de fontes (nomes de arquivos amigáveis)
        display_sources = [os.path.basename(s) for s in source_names if isinstance(s, str) and s]
        fontes_line = f"Fontes: {' e '.join(display_sources)}" if display_sources else "Fontes: -"
        resposta_final = f"{resposta_final}\n\n{fontes_line}"
        
        try:
            append_conversation(message=resposta_final, sender_id=AGENT_ID, receiver_id=a2a_message.get('sender_agent_id'), session_uuid=user_uuid)
        except Exception as log_err:
            print(f"AI_GUIDE_AGENT: aviso - falha ao registrar conversa (resposta): {log_err}")
        
        # Retorno padronizado para a UI/Host (mantém 'result' por compatibilidade)
        return jsonify({
            "answer": resposta_final,  # para UI
            "result": resposta_final,  # compatibilidade
            "sources": source_names,
            "chosen_temperature": best_temp,
            "similarity": best_score
        })
            
    except Exception as e:
        return jsonify({"error": f"Erro interno do agente Guia de IA: {e}"}), 500

def register_with_registry():
    """Registra/renova o registro do agente no Registry (heartbeat)."""
    payload = {"agent_id": AGENT_ID, "base_url": AGENT_BASE_URL, "tools": TOOLS_SCHEMA, "secret_token": AGENT_SECRET_TOKEN}
    try:
        requests.post(f"{REGISTRY_BASE_URL}/register", json=payload).raise_for_status()
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Registro/Heartbeat enviado com sucesso para o Registry!")
    except requests.exceptions.RequestException as e:
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha no registro/heartbeat. Erro: {e}")

def deregister_from_registry():
    """Remove este agente do Registry quando o processo for encerrado."""
    try:
        requests.post(f"{REGISTRY_BASE_URL}/deregister", json={"agent_id": AGENT_ID}, timeout=2)
    except requests.exceptions.RequestException as e:
        print(f"AI_GUIDE_AGENT ({AGENT_ID}): Falha ao desregistrar. Erro: {e}")

def heartbeat_task():
    """
    Função que será executada em segundo plano para enviar o re-registro periódico.
    """
    while True:
        time.sleep(40)
        register_with_registry()


if __name__ == '__main__':
    atexit.register(deregister_from_registry)

    time.sleep(2)
    register_with_registry()

    heartbeat_thread = threading.Thread(target=heartbeat_task, daemon=True)
    heartbeat_thread.start()

    # Sobe o servidor Flask
    app.run(port=FLASK_RUN_PORT, host='0.0.0.0')
