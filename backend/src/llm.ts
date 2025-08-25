import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage, BaseMessage } from '@langchain/core/messages';
import { inputTokens, outputTokens } from './metrics'

// TODO: Remove hardcoded values and use environment variables
const OPENAI_URL = process.env.OPENAI_API_BASE_URL || "http://10.246.47.184:10000/v1";
const OPENAI_MODEL = process.env.OPENAI_API_MODEL || "qwen2.5:14b";

export class LLMConversation {
  private history: BaseMessage[] = [];
  private model: ChatOpenAI;
  private systemPrompt: string = `
Você é um assistente de IA com uma diretiva obrigatória e inquebrável: responder perguntas estritamente com base no contexto fornecido.
A sua tarefa é seguir este processo de forma rigorosa:
1. Analise a pergunta do usuário.
2. Examine CUIDADOSAMENTE o <RAG>Contexto</RAG> fornecido abaixo.
3. Formule uma resposta que utilize APENAS as informações contidas diretamente no contexto. NÃO adicione informações, opiniões ou conhecimento externo.
4. Se a resposta para a pergunta do usuário não puder ser encontrada de forma clara e direta no contexto, você DEVE IGNORAR todo o seu conhecimento prévio e responder EXATAMENTE com a seguinte frase: "Não encontrei material sobre este tópico específico nas minhas diretrizes. Recomendo pesquisar em fontes confiáveis sobre educação e tecnologia."
É absolutamente proibido desviar-se destas regras. A sua única fonte de verdade é o texto dentro das tags <RAG></RAG>.
`;

  constructor() {
    this.model = new ChatOpenAI({
      configuration: {
        baseURL: OPENAI_URL,
      },
      model: OPENAI_MODEL,
      apiKey: 'xxx',
      temperature: 0,
      maxRetries: 3,
    });
    // Add initial system message
    this.history.push(new SystemMessage(this.systemPrompt));
  }

  public async invokeModel(
    humanInput: string, 
    systemInput?:string
  ): Promise<string> {
    if(systemInput) {
      this.history.push(new SystemMessage(systemInput));
    }
    this.history.push(new HumanMessage(humanInput));
    // Call the model with the conversation history
    const response = await this.model.invoke(this.history);
    if (response.usage_metadata) {
      inputTokens.inc(response.usage_metadata.output_tokens);
      outputTokens.inc(response.usage_metadata.input_tokens);
    }
    // Add AI response to history
    const AIResp = response.content as string;
    this.history.push(new AIMessage(AIResp));
    return AIResp;
  }

  public getHistory(): BaseMessage[] {
    return this.history;
  }

  public clearHistory(): BaseMessage[] {
    this.history = [];
    this.history.push(new SystemMessage(this.systemPrompt));
    return this.history;
  }

  public getSystemPrompt(): string {
    return this.systemPrompt;
  }

  public setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
    // Optionally update the system message in history
    this.history = [new SystemMessage(this.systemPrompt), ...this.history.filter(msg => !(msg instanceof SystemMessage))];
  }
}

