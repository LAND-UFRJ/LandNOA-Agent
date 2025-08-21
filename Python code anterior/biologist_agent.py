import os
# Agente Biólogo
# - Recebe tarefas via /execute (A2A) do Host
# - Usa RAG (ChromaDB) para recuperar contexto (docs em EN); pergunta pode vir em PT
# - Tradução PT->EN da query para recuperar e pontuar melhor
# - Gera múltiplas respostas (temperaturas), escolhe a mais ancorada aos docs
# - Loga conversas no Mongo (via monitoramento), sem quebrar o fluxo em caso de falha
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

# Variáveis de ambiente principais
AGENT_ID = os.getenv("BIOLOGIST_AGENT_ID")                # ID único deste agente
AGENT_SECRET_TOKEN = os.getenv("BIOLOGIST_AGENT_SECRET_TOKEN")  # Token para autenticação A2A
AGENT_BASE_URL = os.getenv("BIOLOGIST_AGENT_BASE_URL")    # URL pública deste agente (para registry)
REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL")        # URL do Registry para registro/heartbeat
FLASK_RUN_PORT = int(os.getenv("BIOLOGIST_FLASK_RUN_PORT", 8006))  # Porta HTTP

app = Flask(__name__)

retriever = ChromaDBRetriever()
# Modelo multilíngue melhora similaridade entre PT↔EN
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

TOOLS_SCHEMA = [
    {
        "name": "responder_como_biologo",
        "description": "Use esta ferramenta para responder perguntas sobre biologia, ecossistemas, animais, plantas e ciências da vida com a precisão de um biólogo.",
        "parameters": {"pergunta": "string"}
    }
]


SYSTEM_PROMPT = """
Você é um biólogo de campo e pesquisador.
1. Sua principal diretriz é responder à pergunta do usuário baseando-se ESTRITAMENTE no contexto fornecido pela consulta RAG.
2. NÃO utilize nenhum conhecimento externo ou prévio.
3. Se a informação para responder à pergunta não estiver no contexto do RAG, você DEVE responder: "Com base nos documentos disponíveis, não tenho informações suficientes para responder a essa pergunta."
4. A sua resposta deve ser APENAS o texto da resposta, sem preâmbulos.
5. Responda em português. O contexto poderá estar em inglês; use-o literalmente como fonte.
"""

def verificar_proximidade(resposta_gerada: str, documentos_contexto: list, limiar=0.35):
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
    print(f"BIOLOGIST_AGENT: Maior similaridade encontrada: {maior_similaridade:.4f}")
    return maior_similaridade >= limiar


def calcular_similaridade_maxima(texto: str, documentos: list) -> float:
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


def translate_pt_to_en(text: str) -> str:
    """Traduz PT→EN via LLM; retorna original em caso de falha."""
    try:
        prompt = (
            "Traduza o seguinte texto do português para o inglês.\n"
            "Responda APENAS com o texto traduzido, sem comentários adicionais.\n\n"
            f"Texto: {text}"
        )
        translated = invoke_model(prompt)
        result = translated.content if hasattr(translated, 'content') else translated
        # Normaliza para string em qualquer cenário
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    # tenta campos comuns
                    parts.append(str(item.get('text') or item.get('content') or item))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return result if isinstance(result, str) else str(result)
    except Exception:
        return text


def rerank_documents_by_query(query_text: str, documents: list[str], top_k: int = 3) -> list[str]:
    """Reranqueia documentos por similaridade com a query (em inglês)."""
    if not documents:
        return []
    try:
        q = np.asarray(embedding_model.encode(query_text, convert_to_tensor=False), dtype=float)
        d = np.asarray(embedding_model.encode(documents, convert_to_tensor=False), dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        sims = np.dot(d, q)
        order = np.argsort(-sims)
        top_idx = order[: min(top_k, len(documents))]
        return [documents[i] for i in top_idx]
    except Exception:
        return documents[: top_k]

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificação de saúde (Docker/K8s/Monitoramento)."""
    return jsonify({"status": "ok"}), 200


@app.route('/execute', methods=['POST'])
def execute_task():
    """Rota A2A: executa a ferramenta 'responder_como_biologo'.
    Espera JSON com payload.query (pergunta do usuário) e payload.uuid (sessão).
    Retorna JSON com 'answer' (texto final), 'sources' e métricas auxiliares.
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
        # Consulta SOMENTE em EN (docs estão em inglês)
        query_en = translate_pt_to_en(query)
        _, src_en, docs_en = retriever.retrieve_and_format(
            collection_name='ecologista',
            query_text=query_en,
            n_results=6,
        )

        # Remove duplicatas e reranqueia pelos mais similares à query EN
        uniq_docs = []
        seen = set()
        for d in (docs_en or []):
            if isinstance(d, str) and d not in seen:
                seen.add(d)
                uniq_docs.append(d)

        raw_documents = rerank_documents_by_query(query_en, uniq_docs, top_k=3)
        source_names = src_en or []
        rag_context = retriever._format_context_for_llm(raw_documents)
        
        try:
            append_conversation(message=query, sender_id=a2a_message.get('sender_agent_id'), receiver_id=AGENT_ID, session_uuid=user_uuid)
        except Exception as log_err:
            print(f"BIOLOGIST_AGENT: aviso - falha ao registrar conversa (entrada usuário): {log_err}")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=f'The history of this conversation is :\n{get_context(user_uuid)}'),
            # Passa o contexto já formatado (evita dupla tag <RAG>)
            SystemMessage(content=rag_context),
            HumanMessage(content=query)
        ]
        
        # Gera 3 respostas com diferentes temperaturas e escolhe a mais ancorada
        temps = [0.3, 0.6, 0.9]
        candidates = []  # (text, is_anchor, score, temp)
        for idx, t in enumerate(temps):
            # usa seeds diferentes para incentivar diversidade
            seed = int(time.time()) % 100000 + idx
            try:
                resp = invoke_model_with_temperature(messages, t, seed=seed)
                text = resp.content if hasattr(resp, 'content') else str(resp)
                if not isinstance(text, str):
                    text = str(text)
                # Traduz resposta para EN para melhor alinhamento na similaridade
                text_en = translate_pt_to_en(text)
                score = calcular_similaridade_maxima(text_en, raw_documents)
                is_anchor = score >= 0.35
                candidates.append((text, is_anchor, score, t))
                print(f"BIOLOGIST_AGENT: temp={t} seed={seed} similarity={score:.4f} anchored={is_anchor}")
            except Exception as e:
                print(f"BIOLOGIST_AGENT: falha na temperatura {t} (seed={seed}): {e}")
        
        resposta_padrao = "Com base nos documentos disponíveis, não tenho informações suficientes para responder a essa pergunta."
        anchored = [c for c in candidates if c[1]]
        if anchored:
            best = max(anchored, key=lambda x: x[2])
            resposta_final, _, best_score, best_temp = best
        else:
            resposta_final, best_score, best_temp = resposta_padrao, 0.0, None
        # Log candidatos para diagnóstico
        for (txt, is_anchor, score, t) in candidates:
            preview = (txt[:120] + '…') if isinstance(txt, str) and len(txt) > 120 else txt
            print(f"BIOLOGIST_AGENT: candidate temp={t} score={score:.6f} anchored={is_anchor} preview={preview!r}")
        print(f"BIOLOGIST_AGENT: escolhido temp={best_temp} similarity={best_score:.6f}")

        # Adiciona linha fixa de fontes ao texto final
        display_sources = [os.path.basename(s) for s in source_names if isinstance(s, str) and s]
        fontes_line = f"Fontes: {' e '.join(display_sources)}" if display_sources else "Fontes: -"
        resposta_final = f"{resposta_final}\n\n{fontes_line}"

        try:
            append_conversation(message=resposta_final, sender_id=AGENT_ID, receiver_id=a2a_message.get('sender_agent_id'), session_uuid=user_uuid)
        except Exception as log_err:
            print(f"BIOLOGIST_AGENT: aviso - falha ao registrar conversa (resposta): {log_err}")

        return jsonify({
            "answer": resposta_final,  # para UI
            "result": resposta_final,  # compatibilidade
            "sources": source_names,
            "chosen_temperature": best_temp,
            "similarity": best_score
        })
            
    except Exception as e:
        return jsonify({"error": f"Erro interno do agente Biólogo: {e}"}), 500


def register_with_registry():
    """Envia registro/heartbeat deste agente ao Registry."""
    payload = {"agent_id": AGENT_ID, "base_url": AGENT_BASE_URL, "tools": TOOLS_SCHEMA, "secret_token": AGENT_SECRET_TOKEN}
    try:
        requests.post(f"{REGISTRY_BASE_URL}/register", json=payload).raise_for_status()
        print(f"BIOLOGIST_AGENT ({AGENT_ID}): Registro/Heartbeat enviado com sucesso para o Registry!")
    except requests.exceptions.RequestException as e:
        print(f"BIOLOGIST_AGENT ({AGENT_ID}): Falha no registro/heartbeat. Erro: {e}")

def deregister_from_registry():
    """Desregistra este agente do Registry (executado no encerramento)."""
    try:
        requests.post(f"{REGISTRY_BASE_URL}/deregister", json={"agent_id": AGENT_ID}, timeout=2)
    except requests.exceptions.RequestException as e:
        print(f"BIOLOGIST_AGENT ({AGENT_ID}): Falha ao desregistrar. Erro: {e}")

def heartbeat_task():
    """
    Thread em segundo plano para renovar o registro (heartbeat) periodicamente.
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

    app.run(port=FLASK_RUN_PORT, host='0.0.0.0')
