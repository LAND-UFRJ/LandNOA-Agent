# registry.py
#
# Este serviço mantém um REGISTRO de agentes especialistas em Redis.
# - Cada agente se REGISTRA periodicamente (heartbeat) com: agent_id, base_url, tools e secret_token.
# - Os dados do agente são guardados em duas chaves (namespaces):
#     agent:<agent_id>  -> dados públicos (base_url, tools)
#     secret:<agent_id> -> segredo (secret_token) usado pelo Host para autenticar chamadas A2A
# - O TTL (HEARTBEAT_TTL) é renovado a cada registro; se expirar, o agente é considerado offline.
# - Endpoints:
#     POST /register   -> registra/renova um agente
#     POST /deregister -> remove um agente
#     GET  /list_agents-> lista dados públicos dos agentes (sem segredos!)
#     GET  /health     -> verificação de saúde do serviço Registry
import os
import json
import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


# Variáveis de ambiente principais:
# - REDIS_HOST/REDIS_PORT: conexão com Redis
# - REGISTRY_AGENT_ID: identificador deste serviço de registro (aparece nos logs)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
AGENT_ID = os.getenv("REGISTRY_AGENT_ID", "registry-agent-v1-main")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    print(f"REGISTRY ({AGENT_ID}): Conexão com Redis em {REDIS_HOST}:{REDIS_PORT} estabelecida com sucesso.")
except RedisConnectionError as e:
    print(f"REGISTRY ({AGENT_ID}): ERRO CRÍTICO - Não foi possível conectar ao Redis. Verifique se o servidor está online. Erro: {e}")
    redis_client = None

# Namespaces para chaves no Redis
AGENT_DATA_PREFIX = "agent:"
AGENT_SECRET_PREFIX = "secret:"
HEARTBEAT_TTL = 60  # segundos (tempo de vida/renovação de cada registro)

def _agent_data_key(agent_id: str) -> str:
    return f"{AGENT_DATA_PREFIX}{agent_id}"

def _agent_secret_key(agent_id: str) -> str:
    return f"{AGENT_SECRET_PREFIX}{agent_id}"

def _iter_agent_data_keys(pattern: str = "*"):
    """Iterador eficiente sobre as chaves de dados dos agentes (usa SCAN/scan_iter)."""
    if not redis_client:
        return
    scan_pat = f"{AGENT_DATA_PREFIX}{pattern}"
    try:
        for k in redis_client.scan_iter(match=scan_pat, count=100):
            yield k
    except Exception as e:
        print(f"REGISTRY ({AGENT_ID}): Erro ao iterar chaves: {e}")


@app.route('/register', methods=['POST'])
def register_agent():
    """Registra ou renova (heartbeat) um agente no Registry.
    Corpo esperado (JSON): {
        agent_id: str,
        base_url: str,
        tools: list[ { name, description, parameters? } ],
        secret_token: str
    }
    """
    if not redis_client:
        return jsonify({"error": "Serviço de registro indisponível (sem conexão com a base de dados)."}), 503

    data = request.get_json(silent=True) or {}
    agent_id = (data.get('agent_id') or '').strip()
    base_url = (data.get('base_url') or '').strip()
    tools = data.get('tools')
    secret_token = (data.get('secret_token') or '').strip()

    if not agent_id or not base_url or not secret_token or not isinstance(tools, list):
        return jsonify({"error": "Dados de registro incompletos ou inválidos (agent_id, base_url, tools:list, secret_token)."}), 400

    # Armazena dados públicos e segredo separadamente; NUNCA exponha o segredo em listagens
    public_payload = json.dumps({
        "base_url": base_url,
        "tools": tools,
    })

    redis_client.set(_agent_data_key(agent_id), public_payload)
    redis_client.set(_agent_secret_key(agent_id), secret_token)

    # TTL como heartbeat
    redis_client.expire(_agent_data_key(agent_id), HEARTBEAT_TTL)
    redis_client.expire(_agent_secret_key(agent_id), HEARTBEAT_TTL)

    total_agents = sum(1 for _ in _iter_agent_data_keys())
    print(f"REGISTRY ({AGENT_ID}): Agente '{agent_id}' REGISTRADO. Total: {total_agents}")
    return jsonify({"message": f"Agente {agent_id} registrado com sucesso."}), 200

@app.route('/deregister', methods=['POST'])
def deregister_agent():
    """Remove um agente do Registry (dados públicos e segredo)."""
    if not redis_client:
        return jsonify({"error": "Serviço de registro indisponível."}), 503

    data = request.get_json(silent=True) or {}
    agent_id = (data.get('agent_id') or '').strip()

    if not agent_id:
        return jsonify({"error": "O 'agent_id' para desregistro não foi fornecido."}), 400

    k_data = _agent_data_key(agent_id)
    k_secret = _agent_secret_key(agent_id)
    if redis_client.exists(k_data):
        redis_client.delete(k_data)
        redis_client.delete(k_secret)
        total_agents = sum(1 for _ in _iter_agent_data_keys())
        print(f"REGISTRY ({AGENT_ID}): Agente '{agent_id}' DESREGISTRADO. Total: {total_agents}")
        return jsonify({"message": f"Agente {agent_id} desregistrado com sucesso."}), 200
    else:
        print(f"REGISTRY ({AGENT_ID}): Tentativa de desregistrar agente inexistente '{agent_id}'.")
        return jsonify({"error": f"Agente {agent_id} não encontrado na lista de registros."}), 404

@app.route('/list_agents', methods=['GET'])
def list_agents():
    """Lista agentes registrados (somente dados públicos: base_url e tools)."""
    if not redis_client:
        return jsonify({"error": "Serviço de registro indisponível."}), 503

    registered_agents = {}
    for key in _iter_agent_data_keys():
        agent_id = key.split(AGENT_DATA_PREFIX, 1)[1]
        agent_data_json = redis_client.get(key)
        if not agent_data_json:
            continue
        try:
            data = json.loads(str(agent_data_json))
        except json.JSONDecodeError:
            continue
        # NUNCA inclua o secret_token nesta listagem
        registered_agents[agent_id] = {
            "base_url": data.get("base_url"),
            "tools": data.get("tools", []),
        }

    print(f"REGISTRY ({AGENT_ID}): Enviando lista de {len(registered_agents)} agentes registrados.")
    return jsonify(registered_agents)


@app.route('/health', methods=['GET'])
def health_check():
    """Verifica se o Registry está saudável (usado por orquestradores/monitoramento)."""
    return jsonify({"status": "ok"}), 200



if __name__ == '__main__':
    port = int(os.getenv("REGISTRY_FLASK_RUN_PORT", 8080))
    if redis_client:
        print(f"Agente de Registro (Registry) iniciado em http://localhost:{port} ... ID: {AGENT_ID}")
        app.run(host='0.0.0.0', port=port)
