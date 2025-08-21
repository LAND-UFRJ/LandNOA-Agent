"""
Host Agent
-----------
Responsável por:
- Descobrir agentes especialistas no Registry e suas ferramentas
- Roteamento: decidir qual ferramenta/agente usar para cada pergunta via LLM
- Health check do agente alvo antes de delegar
- Encaminhar a requisição A2A com autenticação e retornar a resposta para a UI

Variáveis de ambiente relevantes:
- SPECIALIST_AGENTS_SECRETS_JSON: mapa {agent_id: secret_token}
- HOST_AGENT_ID: identificador lógico deste Host
- REGISTRY_BASE_URL: URL do serviço Registry
- LLM_*: configuração do LLM de roteamento
- FLASK_RUN_HOST/PORT: bind do servidor
"""

import os
import json
import requests
import time
import uuid
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

load_dotenv()

# Mapa de segredos por agente especialista para autenticação A2A
SPECIALIST_AGENTS_SECRETS = json.loads(os.getenv("SPECIALIST_AGENTS_SECRETS_JSON", "{}"))
HOST_AGENT_ID = os.getenv("HOST_AGENT_ID", "host-agent")
REGISTRY_BASE_URL = os.getenv("REGISTRY_BASE_URL", "http://localhost:8080")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:14b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.246.47.184:10000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "asd")
FLASK_RUN_HOST = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", "8000"))

# LLM usado exclusivamente para tomada de decisão de roteamento
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=SecretStr(LLM_API_KEY) if LLM_API_KEY else None,
)

app = Flask(__name__)

discovered_tools = []            # Lista "flatten" de ferramentas descobertas
discovered_agents_data = {}      # Mapa agent_id -> {base_url, tools, ...}

def fetch_tools_from_registry():
    """Consulta o Registry para obter agentes e suas ferramentas.
    Atualiza discovered_tools/discovered_agents_data para o roteamento.
    """
    global discovered_tools, discovered_agents_data
    list_agents_url = f"{REGISTRY_BASE_URL}/list_agents"
    print("HOST: Verificando o Registry para descobrir agentes...")
    try:
        response = requests.get(list_agents_url)
        response.raise_for_status()
        registered_agents = response.json()

        all_tools = []
        agent_data = {}
        for agent_id, info in registered_agents.items():
            agent_data[agent_id] = info
            tools = info.get("tools", [])
            for tool in tools:
                tool['owner_agent_id'] = agent_id
                tool['agent_execute_url'] = f"{info.get('base_url')}/execute"
            all_tools.extend(tools)
        
        if all_tools:
            discovered_tools = all_tools
            discovered_agents_data = agent_data
            print(f"HOST: Descoberta concluída. {len(discovered_tools)} ferramentas de {len(discovered_agents_data)} agentes estão ativas.")
        else:
            discovered_tools = []
            discovered_agents_data = {}
            print("HOST: Descoberta concluída, mas nenhum agente/ferramenta encontrado no Registry.")

    except requests.exceptions.RequestException as e:
        print(f"HOST: Erro ao contatar o Registry. Usando a última lista conhecida. Erro: {e}")


def heuristic_route(query: str):
    """Roteia por heurística quando a decisão do LLM falha.
    Procura palavras-chave óbvias e escolhe a ferramenta correspondente.
    Retorna um dict de decisão compatível ou None.
    """
    if not query or not discovered_tools:
        return None

    q = query.lower()
    # Palavras-chave para IA/educação/ética
    ai_kw = [
        "inteligência artificial", "inteligencia artificial", " ia ", " ai ",
        "machine learning", "ml ", "ética", "etica", "responsável", "responsavel",
        "algoritmo", "algoritmos", "modelo de ia", "modelos de ia"
    ]
    # Palavras-chave para biologia/ecologia
    bio_kw = [
        "biologia", "ecologia", "ecossistema", "animal", "animais", "planta", "plantas",
        "espécie", "especie", "habitat", "conservação", "conservacao", "biólogo", "biologo"
    ]

    def tool_matches(predicate):
        for t in discovered_tools:
            name = (t.get("name") or "").lower()
            desc = (t.get("description") or "").lower()
            if predicate(name, desc):
                return {
                    "tool_to_use": t.get("name"),
                    "owner_agent_id": t.get("owner_agent_id"),
                    "agent_execute_url": t.get("agent_execute_url")
                }
        return None

    if any(k in f" {q} " for k in ai_kw):
        # Prefer ferramentas que mencionem "ia" ou "guia"
        decision = tool_matches(lambda n, d: "guia" in n or "ia" in n or "ia" in d)
        if decision:
            return decision

    if any(k in q for k in bio_kw):
        decision = tool_matches(lambda n, d: "biolog" in n or "biolog" in d or "ecolog" in d)
        if decision:
            return decision

    return None

def get_system_prompt():
    """Monta o prompt do sistema que descreve as ferramentas descobertas ao LLM."""
    tools_json = json.dumps(discovered_tools, indent=2)
    return f"""
        Você é um agente roteador mestre. Sua tarefa é analisar a pergunta do usuário e determinar qual das ferramentas disponíveis é a mais adequada para respondê-la.

        Ferramentas Descobertas:
        {tools_json}

        Analise a pergunta e escolha a melhor ferramenta. Sua resposta DEVE ser APENAS um objeto JSON contendo a ferramenta escolhida. Se nenhuma ferramenta for claramente adequada para a pergunta, você deve responder com um JSON indicando que nenhuma ferramenta foi encontrada.

        Se a pergunta for uma saudação, um agradecimento ou algo que não se encaixa em nenhuma ferramenta, responda com: {{"tool_to_use": "none", "owner_agent_id": "none", "agent_execute_url": "none"}}

        O formato da resposta para uma ferramenta válida deve ser: {{"tool_to_use": "nome_da_ferramenta", "owner_agent_id": "id_do_agente_dono", "agent_execute_url": "url_para_execucao"}}.
        """

def is_capabilities_query(text: str) -> bool:
    """Heurística simples para detectar perguntas sobre capacidades/listagem."""
    if not text:
        return False
    t = text.strip().lower()
    keywords = [
        "o que você faz", "o que voce faz", "o que pode fazer", "o que podem fazer",
        "quais agentes", "quais ferramentas", "ferramentas disponíveis", "ferramentas disponiveis",
        "o que pode", "ajuda", "help", "capabilities", "capacidades", "comandos", "oi quais são suas ferramentas?","o que consegue fazer"
    ]
    return any(k in t for k in keywords)


def format_capabilities_response() -> dict:
    """Retorna uma visão legível e estruturada dos agentes/ferramentas disponíveis."""
    # Estrutura agrupada por agente
    agents = []
    for agent_id, info in discovered_agents_data.items():
        tools = info.get("tools", [])
        agents.append({
            "agent_id": agent_id,
            "base_url": info.get("base_url"),
            "tools": [{
                "name": t.get("name"),
                "description": t.get("description")
            } for t in tools]
        })

    # Resumo textual
    if not agents:
        text = "No momento, não há agentes registrados no sistema."
    else:
        lines = ["Posso coordenar os seguintes agentes e ferramentas:"]
        for a in agents:
            lines.append(f"- {a['agent_id']}")
            for t in a.get("tools", []):
                lines.append(f"    • {t.get('name')}: {t.get('description')}")
        text = "\n".join(lines)

    return {"answer": text, "agents": agents}

def delegate_to_llm(query: str):
    """Pede ao LLM que escolha a ferramenta mais adequada e retorna a decisão JSON.
    Caso não haja parsing possível, retorna None.
    """
    system_prompt = get_system_prompt()
    print(f"HOST ({HOST_AGENT_ID}): Enviando query '{query}' para o LLM de roteamento...")
    
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        response = llm.invoke(messages)
        llm_output = response.content
        
        print(f"HOST ({HOST_AGENT_ID}): Saída bruta do LLM: {llm_output}")

        # Garante string para json.loads
        if not isinstance(llm_output, str):
            llm_output = str(llm_output)
        # Tentativa direta
        try:
            decision = json.loads(llm_output)
        except json.JSONDecodeError:
            # Tenta extrair substring de JSON
            start = llm_output.find('{')
            end = llm_output.rfind('}')
            if start != -1 and end != -1 and end > start:
                decision = json.loads(llm_output[start:end+1])
            else:
                raise
        print(f"HOST ({HOST_AGENT_ID}): Decisão de roteamento do LLM: {decision}")
        
        if "tool_to_use" not in decision or decision.get("tool_to_use") == "none":
            print(f"HOST ({HOST_AGENT_ID}): LLM decidiu que nenhuma ferramenta é adequada.")
            return None

        return decision

    except (json.JSONDecodeError, Exception) as e:
        print(f"HOST ({HOST_AGENT_ID}): Erro ao processar saída do LLM: {e}")
        return None

def request_deregistration(agent_id: str):
    """
    Envia um pedido ao Registry para des-registrar um agente que falhou no health check.
    """
    deregister_url = f"{REGISTRY_BASE_URL}/deregister"
    payload = {"agent_id": agent_id}
    
    print(f"HOST ({HOST_AGENT_ID}): O agente '{agent_id}' falhou no health check. A solicitar o seu des-registro...")
    
    try:
        response = requests.post(deregister_url, json=payload, timeout=3)
        if response.status_code == 200:
            print(f"HOST ({HOST_AGENT_ID}): Agente '{agent_id}' des-registrado com sucesso do Registry.")
        else:
            print(f"HOST ({HOST_AGENT_ID}): O Registry respondeu com status {response.status_code} ao tentar des-registrar '{agent_id}'.")
    except requests.exceptions.RequestException as e:
        print(f"HOST ({HOST_AGENT_ID}): Falha ao comunicar com o Registry para des-registrar o agente '{agent_id}'. Erro: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para o Docker verificar se o host agent está online."""
    return jsonify({"status": "ok"}), 200


@app.route('/query', methods=['POST'])
def handle_query():
    """Endpoint consumido pela UI: roteia a pergunta para um agente especialista.
    Aplica fallback amigável quando não há ferramenta adequada.
    """
    payload = request.get_json(silent=True) or {}
    user_query = payload.get('query')
    user_uuid = payload.get('uuid')
    
    if not user_query:
        return jsonify({"error": "A 'query' não pode estar vazia."}), 400

    fetch_tools_from_registry()

    if not discovered_tools:
        return jsonify({"error": "Nenhum agente especialista disponível no momento."}), 503

    # Se for uma pergunta sobre capacidades, responde diretamente
    if is_capabilities_query(user_query):
        return jsonify(format_capabilities_response()), 200

    decision = delegate_to_llm(user_query)

    if not decision or "agent_execute_url" not in decision or decision.get("agent_execute_url") == "none":
        # Tenta roteamento por heurística antes do fallback amigável
        h = heuristic_route(user_query)
        if h:
            decision = h
        else:
        # Fallback amigável com explicação e lista de agentes
            cap = format_capabilities_response()
            explicacao = (
                "Olá! Sou o agente Host. Eu recebo sua pergunta, escolho o agente mais adequado e encaminho sua solicitação.\n"
                "Quando não encontro um agente ideal, posso te mostrar quem está disponível agora:\n\n"
            )
            return jsonify({"answer": f"{explicacao}{cap['answer']}", "agents": cap.get("agents", [])}), 200

    # Revalida a decisão antes de prosseguir
    if (not decision or
        not isinstance(decision.get("agent_execute_url"), str) or not decision.get("agent_execute_url") or
        not isinstance(decision.get("owner_agent_id"), str) or not decision.get("owner_agent_id")):
        cap = format_capabilities_response()
        explicacao = (
            "Olá! Sou o agente Host. Eu recebo sua pergunta, escolho o agente mais adequado e encaminho sua solicitação.\n"
            "Quando não encontro um agente ideal, posso te mostrar quem está disponível agora:\n\n"
        )
        return jsonify({"answer": f"{explicacao}{cap['answer']}", "agents": cap.get("agents", [])}), 200

    target_agent_id = decision.get("owner_agent_id")
    target_url = decision.get("agent_execute_url")
    secret_token = SPECIALIST_AGENTS_SECRETS.get(target_agent_id)

    if not secret_token:
        return jsonify({"error": f"Erro de segurança interno: token para o agente {target_agent_id} não encontrado."}), 500

    agent_info = discovered_agents_data.get(target_agent_id)
    if not agent_info:
        return jsonify({"error": f"Dados do agente '{target_agent_id}' não encontrados."}), 500
    
    agent_base_url = agent_info.get("base_url")
    health_check_url = f"{agent_base_url}/health"

    try:
        print(f"HOST ({HOST_AGENT_ID}): A verificar a saúde do agente '{target_agent_id}' em {health_check_url}...")
        health_response = requests.get(health_check_url, timeout=2)
        health_response.raise_for_status()
        print(f"HOST ({HOST_AGENT_ID}): Agente '{target_agent_id}' está saudável. A delegar tarefa...")
    except requests.exceptions.RequestException as e:
        error_message = f"O agente especialista '{target_agent_id}' está indisponível no momento. Por favor, tente novamente mais tarde."
        print(f"HOST ({HOST_AGENT_ID}): Health check para '{target_agent_id}' falhou. Erro: {e}")
        
        # Des-registra apenas se o ID for válido
        if isinstance(target_agent_id, str) and target_agent_id:
            request_deregistration(target_agent_id)
        
        return jsonify({"error": error_message}), 503

    a2a_message = {
        "message_id": f"msg_{uuid.uuid4()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender_agent_id": HOST_AGENT_ID,
        "receiver_agent_id": target_agent_id,
        "payload_type": "text_query",
        "payload": {
            "query": user_query,
            "uuid": user_uuid
        }
    }

    headers = {
        "Authorization": f"Bearer {secret_token}",
        "Content-Type": "application/json"
    }

    try:
        # Chamada ao agente especialista
        response = requests.post(str(target_url), json=a2a_message, headers=headers)
        response.raise_for_status()
        specialist_result = response.json()

        return jsonify({
            "answer": specialist_result.get("result"),
            "source_agent_id": target_agent_id,
            "source_tool": decision.get("tool_to_use"),
            "sources": specialist_result.get("sources"),
            "chosen_temperature": specialist_result.get("chosen_temperature"),
            "similarity": specialist_result.get("similarity"),
        })

    except requests.exceptions.RequestException as e:
        error_message = f"Erro de comunicação com o agente especialista: {e}"
        if e.response is not None:
             error_message = f"Erro ao comunicar com o agente especialista: {e.response.status_code} {e.response.text}"
        
        print(f"HOST ({HOST_AGENT_ID}): {error_message}")
        return jsonify({"error": error_message}), 502

if __name__ == '__main__':
    print("HOST: Iniciando a primeira descoberta de agentes...")
    
    while True:
        fetch_tools_from_registry()
        if discovered_tools:
            print("HOST: Descoberta inicial bem-sucedida!")
            break
        print("HOST: Nenhum agente encontrado ainda. A tentar novamente em 5 segundos...")
        time.sleep(5)

    print(f"Agente Host (IA com A2A) iniciado e dinâmico em http://{FLASK_RUN_HOST}:{FLASK_RUN_PORT} ... ID: {HOST_AGENT_ID}")
    app.run(host=FLASK_RUN_HOST, port=FLASK_RUN_PORT)
