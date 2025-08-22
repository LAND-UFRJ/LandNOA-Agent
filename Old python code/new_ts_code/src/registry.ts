import express from 'express';
import dotenv from 'dotenv';
import Redis from 'ioredis';

dotenv.config();

const REDIS_HOST = process.env.REDIS_HOST || 'localhost';
const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379', 10);
const HEARTBEAT_TTL = parseInt(process.env.HEARTBEAT_TTL || '60', 10);

let redisClient: Redis | null = null;
try {
  redisClient = new Redis({ host: REDIS_HOST, port: REDIS_PORT });
  redisClient.ping().then(() => console.log(`REGISTRY: conectado ao Redis ${REDIS_HOST}:${REDIS_PORT}`));
} catch (e) {
  console.warn('REGISTRY: falha ao conectar ao Redis; modo fallback (memoria) ativado', e);
  redisClient = null;
}

const AGENT_DATA_PREFIX = 'agent:';
const AGENT_SECRET_PREFIX = 'secret:';

const inMemoryAgents: Record<string, any> = {};

const app = express();
app.use(express.json());

function dataKey(agentId: string) { return `${AGENT_DATA_PREFIX}${agentId}`; }
function secretKey(agentId: string) { return `${AGENT_SECRET_PREFIX}${agentId}`; }

app.post('/register', async (req, res) => {
  const data = req.body || {};
  const agentId = (data.agent_id || '').trim();
  const baseUrl = (data.base_url || '').trim();
  const tools = data.tools;
  const secretToken = (data.secret_token || '').trim();
  if (!agentId || !baseUrl || !secretToken || !Array.isArray(tools)) return res.status(400).json({ error: 'Dados de registro incompletos.' });

  const publicPayload = JSON.stringify({ base_url: baseUrl, tools });
  if (redisClient) {
    await redisClient.set(dataKey(agentId), publicPayload);
    await redisClient.set(secretKey(agentId), secretToken);
    await redisClient.expire(dataKey(agentId), HEARTBEAT_TTL);
    await redisClient.expire(secretKey(agentId), HEARTBEAT_TTL);
  } else {
    inMemoryAgents[agentId] = { base_url: baseUrl, tools, secret_token: secretToken, ttl: Date.now() + HEARTBEAT_TTL * 1000 };
  }
  console.log(`REGISTRY: agente ${agentId} registrado`);
  return res.json({ message: `Agente ${agentId} registrado com sucesso.` });
});

app.post('/deregister', async (req, res) => {
  const agentId = (req.body?.agent_id || '').trim();
  if (!agentId) return res.status(400).json({ error: "O 'agent_id' nao foi fornecido." });
  if (redisClient) {
    const key = dataKey(agentId);
    const exists = await redisClient.exists(key);
    if (exists) {
      await redisClient.del(dataKey(agentId));
      await redisClient.del(secretKey(agentId));
      console.log(`REGISTRY: agente ${agentId} desregistrado`);
      return res.json({ message: `Agente ${agentId} desregistrado com sucesso.` });
    } else return res.status(404).json({ error: `Agente ${agentId} nao encontrado.` });
  } else {
    if (inMemoryAgents[agentId]) { delete inMemoryAgents[agentId]; return res.json({ message: 'desregistrado' }); }
    return res.status(404).json({ error: 'nao encontrado' });
  }
});

app.get('/list_agents', async (_req, res) => {
  const out: Record<string, any> = {};
  if (redisClient) {
    try {
      const keys = await redisClient.keys(dataKey('*'));
      for (const k of keys) {
        const agentId = k.replace(AGENT_DATA_PREFIX, '');
        const jsonStr = await redisClient.get(k);
        if (!jsonStr) continue;
        const data = JSON.parse(jsonStr);
        out[agentId] = { base_url: data.base_url, tools: data.tools };
      }
    } catch (e) { console.warn('REGISTRY: erro ao listar agentes', e); }
  } else {
    for (const [id, v] of Object.entries(inMemoryAgents)) {
      out[id] = { base_url: v.base_url, tools: v.tools };
    }
  }
  return res.json(out);
});

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

const port = parseInt(process.env.REGISTRY_FLASK_RUN_PORT || '8080', 10);
app.listen(port, () => console.log(`Registry rodando em http://0.0.0.0:${port}`));
