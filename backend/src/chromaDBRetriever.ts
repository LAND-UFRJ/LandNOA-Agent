import { ChromaClient, IncludeEnum } from "chromadb";

const collectionName = 'landagents';

export class ChromaDBRetriever {
  private client: ChromaClient;

  constructor(URI: string, port: number) {
    this.client = new ChromaClient({
      host: URI,
      port: port,
    });
  }

  async heartbeat() {
    return await this.client.heartbeat();
  }

  async addEmbeddings(
    ids: string[],
    documents?: string[],
  ) {
    const collection = await this.client.getOrCreateCollection({ name: collectionName });
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
    const collection = await this.client.getOrCreateCollection({ name: collectionName });
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

