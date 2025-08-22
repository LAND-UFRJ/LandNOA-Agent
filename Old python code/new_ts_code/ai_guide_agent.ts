import express from 'express';
import dotenv from 'dotenv';
import ChromaDBRetriever from './rag_edson';
import monitor from './monitoramento';

dotenv.config();

const AGENT_ID = process.env.AI_GUIDE_AGENT_ID || 'ai-guide-agent';
const AGENT_SECRET_TOKEN = process.env.AI_GUIDE_AGENT_SECRET_TOKEN || 'secret';
const PORT = parseInt(process.env.AI_GUIDE_FLASK_RUN_PORT || '8010', 10);

const app = express();
app.use(express.json());

const retriever = new ChromaDBRetriever();

const SYSTEM_PROMPT = `Você é um assistente de IA que responde com base no contexto fornecido.`;

function cosineSimilarity(a: number[], b: number[]) {
  const dot = a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0);
  const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0)) || 1;
  const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0)) || 1;
  return dot / (na * nb);
}

function encodeToVector(text: string, dim = 128): number[] {
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    vec[i % dim] += text.charCodeAt(i) % 10;
  }
  return vec;
}

function calcularSimilaridadeMaxima(texto: string, documentos: string[]) {
  if (!documentos || documentos.length === 0) return 0;
  const vText = encodeToVector(texto);
  const docsVecs = documentos.map(encodeToVector);
  const sims = docsVecs.map(dv => cosineSimilarity(dv, vText));
  return sims.length ? Math.max(...sims) : 0;
}

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

app.post('/execute', async (req, res) => {
  const auth = req.headers['authorization'];
  if (!auth || auth !== `Bearer ${AGENT_SECRET_TOKEN}`) return res.status(403).json({ error: 'Acesso nao autorizado.' });

  const a2a = req.body || {};
  const query = a2a.payload?.query;
  const userUuid = a2a.payload?.uuid;
  if (!query) return res.status(400).json({ error: "A 'query' nao foi recebida." });

  try {
    const [ragContext, sourceNames, rawDocs] = await retriever.retrieveAndFormat('Boas_Praticas_IA', query, 6);

    try { await monitor.appendConversation(query, a2a.sender_agent_id, AGENT_ID, userUuid); } catch (e) { console.warn('ai_guide_agent: falha append input', e); }

    const temps = [0.3, 0.6, 0.9];
    const candidates: Array<{ text: string; isAnchor: boolean; score: number; temp: number | null }> = [];

    for (let i = 0; i < temps.length; i++) {
      const t = temps[i];
      try {
        const resp = await monitor.invokeModelWithTemperature([SYSTEM_PROMPT, ragContext, query], t, Date.now() % 100000 + i);
        const text = String(resp.content || resp);
        const score = calcularSimilaridadeMaxima(text, rawDocs || []);
        const isAnchor = score >= 0.35;
        candidates.push({ text, isAnchor, score, temp: t });
        console.log(`ai_guide_agent: temp=${t} similarity=${score.toFixed(4)} anchored=${isAnchor}`);
      } catch (e) {
        console.warn('ai_guide_agent: falha ao invocar modelo', e);
      }
    }

    const respostaPadrao = 'Nao encontrei material sobre este topico especifico nas minhas diretrizes.';
    const anchored = candidates.filter(c => c.isAnchor);
    let respostaFinal = respostaPadrao;
    let bestTemp: number | null = null;
    let bestScore = 0;
    if (anchored.length > 0) {
      const best = anchored.reduce((a, b) => (a.score > b.score ? a : b));
      respostaFinal = best.text;
      bestTemp = best.temp;
      bestScore = best.score;
    }

    const displaySources = (sourceNames || []).filter(Boolean).map((s: string) => s.split('/').pop());
    const fontesLine = displaySources.length ? `Fontes: ${displaySources.join(' e ')}` : 'Fontes: -';
    respostaFinal = `${respostaFinal}\n\n${fontesLine}`;

    try { await monitor.appendConversation(respostaFinal, AGENT_ID, a2a.sender_agent_id, userUuid); } catch (e) { console.warn('ai_guide_agent: falha append resposta', e); }

    return res.json({ answer: respostaFinal, result: respostaFinal, sources: sourceNames, chosen_temperature: bestTemp, similarity: bestScore });
  } catch (e: any) {
    return res.status(500).json({ error: `Erro interno do agente Guia de IA: ${e.message || e}` });
  }
});

app.listen(PORT, () => console.log(`AI Guide Agent rodando em http://0.0.0.0:${PORT}`));
