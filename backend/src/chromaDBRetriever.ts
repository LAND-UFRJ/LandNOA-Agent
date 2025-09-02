import { ChromaClient, IncludeEnum } from "chromadb";

const collectionName = 'landagents';

// Função de embedding simples que retorna vetores aleatórios
class SimpleEmbeddingFunction {
  async generate(texts: string[]): Promise<number[][]> {
    // Para teste, retorna vetores aleatórios de tamanho fixo
    return texts.map(() => Array.from({ length: 384 }, () => Math.random() - 0.5));
  }
}

export class ChromaDBRetriever {
  private client: ChromaClient;
  private embeddingFunction: SimpleEmbeddingFunction;

  constructor(URI: string, port: number) {
    this.client = new ChromaClient({
      host: URI,
      port: port,
    });
    this.embeddingFunction = new SimpleEmbeddingFunction();
  }

  async heartbeat() {
    return await this.client.heartbeat();
  }

  async addEmbeddings(
    ids: string[],
    documents?: string[],
  ) {
    const collection = await this.client.getOrCreateCollection({ 
      name: collectionName,
      embeddingFunction: this.embeddingFunction as any
    });
    return collection.add({
      ids: ids,
      documents: documents,
    });
  }

  async query(
    query: string[],
    n_results: number = 3,
    include: IncludeEnum[] = [IncludeEnum.distances, IncludeEnum.metadatas, IncludeEnum.documents]
  ) {
    const collection = await this.client.getOrCreateCollection({ 
      name: collectionName,
      embeddingFunction: this.embeddingFunction as any
    });
    return collection.query({
      queryTexts: query,
      nResults: n_results,
      include,
    });
  }

  async retrieveFormatted(query: string, nResults: number = 3) {
    console.log(`Retrieving documents for query: ${query}`);

    const query_results = await this.query(
      [query], nResults
    );

    const raw_documents = query_results.documents;
    const metadatas = query_results.metadatas;

    const source_names: string[] = [];
    if (metadatas && Array.isArray(metadatas)) {
      for (const meta of metadatas) {
        if (meta && typeof meta === 'object' && 'source' in meta) {
          source_names.push(meta['source'] as string);
        } else {
          source_names.push("Origem desconhecida");
        }
      }
    }

    const unique_source_names = Array.from(new Set(source_names));

    // Format result in LLM style
    let context_str = "<RAG> Contexto recuperado:\n";
    if (raw_documents && raw_documents.length > 0) {
      for (const docArr of raw_documents) {
        for (const doc of docArr) {
          context_str += `\n---\n${doc}\n---\n`;
        }
      }
    }
    context_str += "</RAG>";

    return {
      context: context_str,
      sources: unique_source_names,
    };
  }

}