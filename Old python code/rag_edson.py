import os
import chromadb
from dotenv import load_dotenv
from typing import Any

load_dotenv()

class ChromaDBRetriever:
    """
    Uma classe padrão para interagir com o ChromaDB, usada por todos os agentes.
    Responsabilidades:
    - Conectar ao serviço ChromaDB (HTTP) com host/port do .env
    - Executar queries numa coleção (collection_name) com um texto (query_text)
    - Formatar os documentos recuperados no padrão <RAG> ... </RAG>
    - Retornar: (contexto_formatado, lista_de_fontes, documentos_brutos)
    """
    def __init__(self):
        """Inicializa o cliente do ChromaDB usando as configurações do ambiente.
        Variáveis:
        - CHROMADB_HOST: host do serviço ChromaDB (default: localhost)
        - CHROMADB_PORT: porta do serviço (default: 8000)
        - RAG_N_RESULTS: número padrão de documentos a recuperar (default: 6)
        """
        try:
            # Número padrão de documentos a retornar (configurável via .env)
            try:
                self.default_n_results = int(os.getenv("RAG_N_RESULTS", "6") or 6)
            except ValueError:
                self.default_n_results = 6

            # Valores seguros por padrão
            chromadb_host = (os.getenv("CHROMADB_HOST") or "localhost").strip() or "localhost"
            port_env = os.getenv("CHROMADB_PORT")
            try:
                chromadb_port = int(port_env) if port_env not in (None, "") else 8000
            except (TypeError, ValueError):
                print(f"Retriever: CHROMADB_PORT inválido ({port_env!r}). Usando porta padrão 8000.")
                chromadb_port = 8000

            self.client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
            print(f"Retriever: Conexão com ChromaDB em {chromadb_host}:{chromadb_port} estabelecida.")
        except Exception as e:
            print(f"Retriever: Falha crítica ao preparar cliente do ChromaDB. Operações de RAG serão desativadas. Erro: {e}")
            self.client = None

    def _query_collection(self, collection_name: str, query_text: str, n_results: int | None = None) -> Any:
        """Executa uma busca numa coleção e retorna o resultado bruto do ChromaDB.
        Parâmetros:
        - collection_name: nome da coleção já existente no Chroma
        - query_text: texto da consulta (string)
        - n_results: quantos documentos retornar
        Retorno: dict com chaves 'documents', 'metadatas', etc.
        """
        if not self.client:
            print("Retriever: Cliente ChromaDB não inicializado. Impossível realizar a busca.")
            return {}
        
        try:
            if n_results is None:
                n_results = self.default_n_results
            collection = self.client.get_collection(name=collection_name)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]  # Inclui metadados e distâncias (IDs podem vir separados)
            )
            return results
        except Exception as e:
            print(f"Retriever: Erro ao buscar na coleção '{collection_name}'. Erro: {e}")
            return {}

    def _format_context_for_llm(self, documents: list) -> str:
        """Formata uma lista de docs em um bloco <RAG> consumível pelo LLM."""
        if not documents:
            return "<RAG>Nenhum contexto encontrado.</RAG>"

        context_str = "<RAG> Contexto recuperado:\n"
        for doc in documents:
            context_str += f"\n---\n{doc}\n---\n"
        context_str += "</RAG>"
        
        return context_str

    def retrieve_and_format(self, collection_name: str, query_text: str, n_results: int | None = None) -> tuple[str, list, list]:
        """Fluxo completo: busca, formata contexto e extrai fontes.
        Retorno (tupla): (contexto_formatado, fontes_uniques, documentos_brutos)
        """
        print(f"Retriever: Buscando na coleção '{collection_name}' com a query: '{query_text}'")

        query_results = self._query_collection(collection_name, query_text, n_results)

        raw_documents = query_results.get('documents', [[]])[0]
        metadatas = query_results.get('metadatas', [[]])[0]
        ids = query_results.get('ids', [[]])[0] if isinstance(query_results.get('ids'), list) else []
        distances = query_results.get('distances', [[]])[0]

        # --- INÍCIO DA LÓGICA DE IMPRESSÃO DO CONTEÚDO BRUTO ---
        print("\n" + "="*50)
        print("||     Conteúdo Recuperado do RAG (Documentos Brutos)     ||")
        print("="*50)
        if raw_documents:
            for i, doc in enumerate(raw_documents):
                print(f"\n--- Documento {i+1} ---")
                # Metadata, ID e distância
                src = None
                meta_i = metadatas[i] if isinstance(metadatas, list) and i < len(metadatas) else None
                id_i = ids[i] if isinstance(ids, list) and i < len(ids) else None
                dist_i = distances[i] if isinstance(distances, list) and i < len(distances) else None

                if isinstance(meta_i, dict):
                    # Tenta várias chaves comuns para "fonte"
                    for k in ("source", "file_name", "filename", "path", "name", "title", "url"):
                        val = meta_i.get(k)
                        if val:
                            src = os.path.basename(val) if isinstance(val, str) else str(val)
                            break
                print(f"Fonte: {src if src else '(desconhecida)'}")
                if id_i is not None:
                    print(f"ID: {id_i}")
                if dist_i is not None:
                    print(f"Distância: {dist_i}")
                if src is None and isinstance(meta_i, dict):
                    print(f"Metadados (completos): {meta_i}")
                print("")
                print(doc)
            print("\n" + "="*50)
            print("||              Fim do Conteúdo Recuperado              ||")
            print("="*50 + "\n")
        else:
            print("Nenhum documento foi encontrado para esta consulta.")
            print("="*50 + "\n")
        # --- FIM DA LÓGICA DE IMPRESSÃO ---

        # Extrai os nomes dos ficheiros a partir dos metadados
        source_names = []
        if metadatas:
            for meta in metadatas:
                label = None
                if isinstance(meta, dict):
                    for k in ("source", "file_name", "filename", "path", "name", "title", "url"):
                        val = meta.get(k)
                        if val:
                            label = os.path.basename(val) if isinstance(val, str) else str(val)
                            break
                source_names.append(label if label else "Origem desconhecida")
        
        # Remove duplicados para uma lista limpa
        unique_source_names = list(dict.fromkeys(source_names))

        formatted_context = self._format_context_for_llm(raw_documents)
        
        # Retorna três valores agora
        return formatted_context, unique_source_names, raw_documents
