import dotenv from 'dotenv';
import { MongoClient } from 'mongodb';

dotenv.config();

const MONGO_DB = process.env.MONGO_DB || 'chat_history';
// If MONGO_URI isn't provided, try to build it from MONGO_HOST/MONGO_USER/MONGO_PASS/MONGO_DB
let MONGO_URI = process.env.MONGO_URI || '';
if (!MONGO_URI && process.env.MONGO_HOST) {
  const user = process.env.MONGO_USER || '';
  const pass = process.env.MONGO_PASS || '';
  const host = process.env.MONGO_HOST;
  const auth = user && pass ? `${encodeURIComponent(user)}:${encodeURIComponent(pass)}@` : '';
  MONGO_URI = `mongodb://${auth}${host}/${MONGO_DB}`;
}

let mongoClient: MongoClient | null = null;
let collection: any = null;

async function initMongo() {
  if (!MONGO_URI) return;
  try {
    mongoClient = new MongoClient(MONGO_URI);
    await mongoClient.connect();
    const db = mongoClient.db(MONGO_DB);
    collection = db.collection('chat_history');
    console.log('monitoramento: conectado ao MongoDB');
  } catch (e) {
    console.warn('monitoramento: falha ao conectar ao MongoDB (usando fallback em memória)', e);
    mongoClient = null;
    collection = null;
  }
}

// Fallback em memória quando Mongo não estiver disponível
const inMemoryStore: Record<string, any[]> = {};

export type LLMResponse = { content: string; response_metadata?: any };

export async function invokeModel(prompt: any): Promise<LLMResponse> {
  // TODO: integrar com LLM real. Aqui retornamos echo simples.
  const text = typeof prompt === 'string' ? prompt : JSON.stringify(prompt);
  return { content: `LLM_STUB_RESPONSE: ${text}` };
}

export async function invokeModelWithTemperature(prompt: any, temperature: number, seed?: number): Promise<LLMResponse> {
  // TODO: usar temperature/seed no cliente real
  const base = await invokeModel(prompt);
  return { content: `${base.content} [temp=${temperature} seed=${seed ?? 'none'}]` };
}

export async function appendConversation(message: any, senderId: string, receiverId: string, sessionUuid: string, ip = 'local') {
  const content = typeof message === 'string' ? message : message?.content ?? JSON.stringify(message);
  const newMessage = { sender: senderId, message: content, destination: receiverId, timestamp: Date.now() };

  if (collection) {
    await collection.updateOne({ uuid: sessionUuid }, { $setOnInsert: { ip }, $push: { messages: newMessage } }, { upsert: true });
  } else {
    inMemoryStore[sessionUuid] = inMemoryStore[sessionUuid] || [];
    inMemoryStore[sessionUuid].push(newMessage);
  }
}

export async function getContext(sessionUuid: string): Promise<string | any[]> {
  if (collection) {
    const doc = await collection.findOne({ uuid: sessionUuid });
    if (!doc) return [];
    const msgs = doc.messages || [];
    // Se houver muitas mensagens, aqui poderíamos sumarizar via LLM
    if (Array.isArray(msgs) && msgs.length > 5) {
      const summary = await invokeModel(msgs.map((m: any) => m.message).join('\n'));
      return summary.content;
    }
    return msgs;
  }
  return inMemoryStore[sessionUuid] || [];
}

export async function removeBetweenRagTags(input: string) {
  return input.replace(/<RAG>[\s\S]*?<\/RAG>/g, '');
}

// inicializa conexões (não bloqueante)
initMongo().catch(() => {});

export default { initMongo, invokeModel, invokeModelWithTemperature, appendConversation, getContext, removeBetweenRagTags };
