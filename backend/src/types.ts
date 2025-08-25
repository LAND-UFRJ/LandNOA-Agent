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
