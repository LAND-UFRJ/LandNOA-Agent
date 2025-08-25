import express from 'express';
import dotenv from 'dotenv';
import { ChromaDBRetriever } from './chromaDBRetriever';
import * as crypto from 'node:crypto';
import type { AddDocumentsPayload, A2AMessage } from './types';
import { LLMConversation } from './llm';
import { prom_metrics } from './metrics';

dotenv.config();

const AGENT_SECRET_TOKEN = process.env.AGENT_SECRET_TOKEN || crypto.randomBytes(16).toString('hex');
  
console.log(AGENT_SECRET_TOKEN)
const API_PORT = parseInt(process.env.API_PORT || '8000', 10);
const CHROMA_PORT = parseInt(process.env.CHROMA_PORT || '8000', 10);
const CHROMA_URI = process.env.CHROMA_URI || 'localhost';

const app = express();
app.use(express.json({ limit: '2mb' }));

const chroma = new ChromaDBRetriever(CHROMA_URI, CHROMA_PORT);

// TODO: Make this an array to work with multiple uuids
const llm = new LLMConversation();

function authenticateBearer(req: express.Request, res: express.Response, next: express.NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Bearer token required' });
  }
  const token = authHeader.substring(7);
  if (token !== AGENT_SECRET_TOKEN) {
    return res.status(403).json({ error: 'Invalid token' });
  }
  next();
}

app.get('/health', authenticateBearer, async (_req: express.Request, res: express.Response) => {
  try {
    const hb = await chroma.heartbeat();
    res.json({ ok: true, chroma: hb });
  } catch (e: any) {
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

app.post('/collections/add', authenticateBearer, async (req: express.Request, res: express.Response) => {
  const body = req.body as AddDocumentsPayload;
  if (!Array.isArray(body.ids) || !Array.isArray(body.documents)) {
    return res.status(400).json({ error: 'Parâmetros inválidos: collection, ids[], documents[] são obrigatórios' });
  }
  if (body.ids.length !== body.documents.length) {
    return res.status(400).json({ error: 'ids e documents devem ter o mesmo tamanho' });
  }

  try {
    const result = await chroma.addEmbeddings(
      body.ids,
      body.documents,
    );

    res.json({ ok: true, result });
  } catch (e: any) {
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

app.post('/query', authenticateBearer, async (req: express.Request, res: express.Response) => {
  const body = req.body as A2AMessage;
  if (!body?.payload || !body?.payload.query) {
    return res.status(400).json({ error: 'Parâmetros inválidos: query obrigatória' });
  }

  const query = body.payload.query;
  const user_uuid = body.payload.uuid;
  const sender_id = body.sender_agent_id;

  console.log(`Received query from ${sender_id} (${user_uuid}): ${query}`);
  const queryResult = await chroma.retrieveFormatted(query);
  console.log(`Query result: ${JSON.stringify(queryResult)}`);
  // Invoke LLM with Query result and the user query
  const llmResponse = await llm.invokeModel(query, queryResult.context);
  console.log(`LLM response: ${llmResponse}`);

  res.json({
    ok: true,
    response: llmResponse,
    queryResult,
  });
});

// Prometheus metrics endpoint
app.get('/metrics', async (_req: express.Request, res: express.Response) => {
  res.set('Content-Type', prom_metrics.register.contentType);
  res.end(await prom_metrics.register.metrics());
});

app.listen(API_PORT, () => {
  console.log(`Backend up on http://0.0.0.0:${API_PORT}`);
});

