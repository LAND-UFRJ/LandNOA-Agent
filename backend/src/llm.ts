import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { inputTokens, outputTokens } from './metrics'

// TODO: Remove hardcoded values and use environment variables
const OPENAI_URL = process.env.OPENAI_API_BASE_URL || 
  "http://10.246.47.184:10000/v1";
const OPENAI_MODEL = process.env.OPENAI_API_MODEL || "qwen2.5:14b";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'xxx';

export class LLMConversation {
  private model: ChatOpenAI;
  private systemPromptTemplate: string = `Você é um agente de IA que possui 
apenas estes três conhecimentos em ordem de relevância de conhecimento,
sendo o primeiro o mais relevante para dar uma resposta ao usuário e os 
demais conhecimentos apenas complementam a resposta.
Com base apenas nestes três conhecimentos, apresente uma resposta para a 
pergunta do usuário.

Conhecimento 1:
{knowledge1}

Conhecimento 2:
{knowledge2}

Conhecimento 3:
{knowledge3}`;

  constructor() {
    this.model = new ChatOpenAI({
      configuration: {
        baseURL: OPENAI_URL,
      },
      model: OPENAI_MODEL,
      apiKey: OPENAI_API_KEY,
      temperature: 0,
      maxRetries: 3,
    });
  }

  public async invokeModel(
    humanInput: string, 
    knowledge1: string = "",
    knowledge2: string = "",
    knowledge3: string = ""
  ): Promise<string> {
    try {
      // Validate input
      if (!humanInput?.trim()) {
        throw new Error('Human input cannot be empty');
      }

      // Log knowledge availability for debugging
      const hasKnowledge = [knowledge1, knowledge2, knowledge3]
        .some(k => k?.trim());
      if (!hasKnowledge) {
        console.warn('No knowledge provided to LLM');
      }

      // Create system prompt with knowledge variables
      const systemPrompt = this.systemPromptTemplate
        .replace('{knowledge1}', knowledge1)
        .replace('{knowledge2}', knowledge2)
        .replace('{knowledge3}', knowledge3);
      
      // Create messages array with only system prompt and user question
      const messages = [
        new SystemMessage(systemPrompt),
        new HumanMessage(humanInput)
      ];
      
      // Call the model
      const response = await this.model.invoke(messages);
      
      // Update metrics (fixed token counting)
      if (response.usage_metadata) {
        inputTokens.inc(response.usage_metadata.input_tokens);
        outputTokens.inc(response.usage_metadata.output_tokens);
      }
      
      return response.content as string;
    } catch (error) {
      console.error('Error in LLM invocation:', error);
      throw new Error(`Failed to get response from LLM: ${error}`);
    }
  }

  public getSystemPromptTemplate(): string {
    return this.systemPromptTemplate;
  }

  public setSystemPromptTemplate(template: string): void {
    if (!template?.trim()) {
      throw new Error('Template cannot be empty');
    }
    this.systemPromptTemplate = template;
  }

  public generateSystemPrompt(
    knowledge1: string, 
    knowledge2: string, 
    knowledge3: string
  ): string {
    return this.systemPromptTemplate
      .replace('{knowledge1}', knowledge1)
      .replace('{knowledge2}', knowledge2)
      .replace('{knowledge3}', knowledge3);
  }
}