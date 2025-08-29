import { Metadata } from "chromadb";

export interface AddDocumentsPayload {
  ids: string[];                // IDs dos documentos
  documents: string[];          // textos
  metadatas?: Metadata[];       // metadados opcionais (um por doc)
}

export interface A2AMessage {
  sender_agent_id: string; 
  payload: {
    query: string;        
    uuid: string;         
  }
}

export interface MCP_Json {
  type:string;
  server_label:string;
  server_url:string;
  require_approval:string
}