import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

// Support both CHROMA_* and CHROMADB_* env var names (the .env uses CHROMADB_*)
const CHROMA_HOST = process.env.CHROMA_HOST || process.env.CHROMADB_HOST || 'localhost';
const CHROMA_PORT = process.env.CHROMA_PORT || process.env.CHROMADB_PORT || '8000';

export class ChromaDBRetriever {
  baseUrl: string;
  defaultN: number;

  constructor() {
    this.baseUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}`;
    this.defaultN = parseInt(process.env.RAG_N_RESULTS || '6', 10);
  }

  async _queryCollection(collectionName: string, queryText: string, nResults?: number) {
    if (!nResults) nResults = this.defaultN;
    try {
      // Chroma HTTP API expects a POST to /collection/<name>/query or similar depending on server
      const res = await axios.post(`${this.baseUrl}/collection/${collectionName}/query`, {
        query_texts: [queryText],
        n_results: nResults,
        include: ['documents', 'metadatas', 'distances', 'ids']
      }, { timeout: 3000 });
      return res.data;
    } catch (e: any) {
      console.warn('ChromaDBRetriever: falha ao consultar ChromaDB, retornando vazio', e.message || e);
      return {};
    }
  }

  _formatContextForLLM(documents: string[]) {
    if (!documents || documents.length === 0) return '<RAG>Nenhum contexto encontrado.</RAG>';
    let ctx = '<RAG> Contexto recuperado:\n';
    for (const d of documents) {
      ctx += `\n---\n${d}\n---\n`;
    }
    ctx += '</RAG>';
    return ctx;
  }

  async retrieveAndFormat(collectionName: string, queryText: string, nResults?: number): Promise<[string, string[], string[]]> {
    const results = await this._queryCollection(collectionName, queryText, nResults);
    const rawDocuments = (results?.documents?.[0]) || [];
    const metadatas = (results?.metadatas?.[0]) || [];
    const ids = (results?.ids?.[0]) || [];

    const sourceNames: string[] = [];
    for (const meta of metadatas) {
      let label: string | null = null;
      if (meta && typeof meta === 'object') {
        for (const k of ['source', 'file_name', 'filename', 'path', 'name', 'title', 'url']) {
          if (meta[k]) {
            label = String(meta[k]).split('/').pop() || String(meta[k]);
            break;
          }
        }
      }
      sourceNames.push(label || 'Origem desconhecida');
    }

    const formattedContext = this._formatContextForLLM(rawDocuments);
    const uniqueSourceNames = Array.from(new Set(sourceNames));
    return [formattedContext, uniqueSourceNames, rawDocuments];
  }
}

export default ChromaDBRetriever;
