import express from 'express';
import axios from 'axios';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';

dotenv.config();

const SPECIALIST_AGENTS_SECRETS = JSON.parse(process.env.SPECIALIST_AGENTS_SECRETS_JSON || '{}');
const HOST_AGENT_ID = process.env.HOST_AGENT_ID || 'host-agent';
const REGISTRY_BASE_URL = process.env.REGISTRY_BASE_URL || 'http://localhost:8080';
const PORT = parseInt(process.env.FLASK_RUN_PORT || '8000', 10);

const app = express();
app.use(express.json());

type Decision = { tool_to_use: string; owner_agent_id: string; agent_execute_url: string } | null;

let discoveredTools: any[] = [];
let discoveredAgentsData: Record<string, any> = {};

async function fetchToolsFromRegistry() {
  try {
    const res = await axios.get(`${REGISTRY_BASE_URL}/list_agents`, { timeout: 3000 });
    const registered = res.data || {};
    const allTools: any[] = [];
    const agentData: Record<string, any> = {};
    for (const [agentId, info] of Object.entries(registered)) {
      agentData[agentId] = info;
      const tools = (info as any).tools || [];
      for (const tool of tools) {
        (tool as any).owner_agent_id = agentId;
        (tool as any).agent_execute_url = `${(info as any).base_url}/execute`;
        allTools.push(tool);
      }
    }
    discoveredTools = allTools;
    discoveredAgentsData = agentData;
    console.log(`HOST: descobriu ${discoveredTools.length} ferramentas de ${Object.keys(discoveredAgentsData).length} agentes`);
  } catch (e: any) {
    console.warn('HOST: falha ao consultar Registry, mantendo ultimo estado conhecido', e.message || e);
  }
}

function heuristicRoute(query: string): Decision {
  if (!query || discoveredTools.length === 0) return null;
  const q = query.toLowerCase();
  const aiKw = ['inteligencia artificial', 'ia', 'ai', 'ética', 'etica', 'modelo de ia'];
  const bioKw = ['biologia', 'ecologia', 'biologo'];

  if (aiKw.some(k => q.includes(k))) {
    const decision = discoveredTools.find(t => (t.name || '').toLowerCase().includes('guia') || (t.description || '').toLowerCase().includes('ia'));
    if (decision) return { tool_to_use: decision.name, owner_agent_id: decision.owner_agent_id, agent_execute_url: decision.agent_execute_url };
  }
  if (bioKw.some(k => q.includes(k))) {
    const decision = discoveredTools.find(t => (t.name || '').toLowerCase().includes('biolog') || (t.description || '').toLowerCase().includes('biolog'));
    if (decision) return { tool_to_use: decision.name, owner_agent_id: decision.owner_agent_id, agent_execute_url: decision.agent_execute_url };
  }
  return null;
}

function isCapabilitiesQuery(text: string) {
  if (!text) return false;
  const t = text.trim().toLowerCase();
  const keywords = ['o que voce faz', 'quais agentes', 'ferramentas disponiveis', 'capacidades', 'o que pode'];
  return keywords.some(k => t.includes(k));
}

function formatCapabilitiesResponse() {
  const agents: any[] = [];
  for (const [agentId, info] of Object.entries(discoveredAgentsData)) {
    agents.push({ agent_id: agentId, base_url: (info as any).base_url, tools: (info as any).tools?.map((t: any) => ({ name: t.name, description: t.description })) });
  }
  let text = 'Posso coordenar os seguintes agentes e ferramentas:\n';
  for (const a of agents) {
    text += `- ${a.agent_id}\n`;
    for (const t of a.tools) text += `   • ${t.name}: ${t.description}\n`;
  }
  return { answer: text, agents };
}

async function delegateToLLM(query: string): Promise<Decision> {
  // TODO: integrar com LLM real. Aqui fazemos heuristica simples.
  // Return null to indicate no decision from LLM
  return null;
}

async function requestDeregistration(agentId: string) {
  try {
    await axios.post(`${REGISTRY_BASE_URL}/deregister`, { agent_id: agentId }, { timeout: 3000 });
    console.log(`HOST: solicitou desregistro de ${agentId}`);
  } catch (e: any) {
    console.warn('HOST: falha ao solicitar desregistro', e.message || e);
  }
}

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

app.post('/query', async (req, res) => {
  const payload = req.body || {};
  const userQuery = payload.query;
  const userUuid = payload.uuid;
  if (!userQuery) return res.status(400).json({ error: "A 'query' nao pode estar vazia." });

  await fetchToolsFromRegistry();
  if (!discoveredTools || discoveredTools.length === 0) return res.status(503).json({ error: 'Nenhum agente especialista disponivel no momento.' });

  if (isCapabilitiesQuery(userQuery)) return res.json(formatCapabilitiesResponse());

  let decision = await delegateToLLM(userQuery);
  if (!decision) decision = heuristicRoute(userQuery);
  if (!decision) {
    const cap = formatCapabilitiesResponse();
    const explicacao = 'Ol\u00e1! Sou o agente Host. Quando nao encontro um agente ideal, posso mostrar quem esta disponivel agora:\n\n';
    return res.json({ answer: explicacao + cap.answer, agents: cap.agents });
  }

  const targetAgentId = decision.owner_agent_id;
  const targetUrl = decision.agent_execute_url;
  const secretToken = SPECIALIST_AGENTS_SECRETS[targetAgentId];
  if (!secretToken) return res.status(500).json({ error: `Erro de seguranca interno: token para o agente ${targetAgentId} nao encontrado.` });

  const agentInfo = discoveredAgentsData[targetAgentId];
  if (!agentInfo) return res.status(500).json({ error: `Dados do agente '${targetAgentId}' nao encontrados.` });

  try {
    const healthUrl = `${agentInfo.base_url}/health`;
    await axios.get(healthUrl, { timeout: 2000 });
  } catch (e: any) {
    if (targetAgentId) await requestDeregistration(targetAgentId);
    return res.status(503).json({ error: `O agente especialista '${targetAgentId}' esta indisponivel no momento.` });
  }

  const a2aMessage = { message_id: `msg_${uuidv4()}`, timestamp: new Date().toISOString(), sender_agent_id: HOST_AGENT_ID, receiver_agent_id: targetAgentId, payload_type: 'text_query', payload: { query: userQuery, uuid: userUuid } };
  try {
    const response = await axios.post(targetUrl, a2aMessage, { headers: { Authorization: `Bearer ${secretToken}` } });
    const specialistResult = response.data || {};
    return res.json({ answer: specialistResult.result || specialistResult.answer, source_agent_id: targetAgentId, source_tool: decision.tool_to_use, sources: specialistResult.sources, chosen_temperature: specialistResult.chosen_temperature, similarity: specialistResult.similarity });
  } catch (e: any) {
    const msg = e.response ? `${e.response.status} ${e.response.data}` : e.message;
    return res.status(502).json({ error: `Erro ao comunicar com o agente especialista: ${msg}` });
  }
});

// bootstrap discovery loop
(async () => {
  while (true) {
    await fetchToolsFromRegistry();
    if (discoveredTools && discoveredTools.length > 0) break;
    console.log('HOST: nenhum agente encontrado ainda. tentando novamente em 5s...');
    await new Promise(r => setTimeout(r, 5000));
  }
  console.log(`Host Agent iniciado em http://0.0.0.0:${PORT} ID: ${HOST_AGENT_ID}`);
  app.listen(PORT, () => {});
})();
