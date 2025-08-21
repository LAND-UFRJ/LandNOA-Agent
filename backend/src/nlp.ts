
import { pipeline } from '@huggingface/transformers';

type EmbeddedVector = { data: number[] };

async function embed(text: string): Promise<EmbeddedVector> {
  const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  const result = await model(text);
  // Assuming the model returns a 2D array, we take the first element
  return { data: result.data };
}

function cosineSimilarity(vec1: EmbeddedVector, vec2: EmbeddedVector): number {
  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;

  for (let i = 0; i < vec1.data.length; i++) {
    dotProduct += vec1.data[i] * vec2.data[i];
    magnitude1 += vec1.data[i] * vec1.data[i];
    magnitude2 += vec2.data[i] * vec2.data[i];
  }

  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);

  if (magnitude1 === 0 || magnitude2 === 0) {
    return 0; // Handle division by zero if a vector is zero
  }

  return dotProduct / (magnitude1 * magnitude2);
}

async function stringSimilarity(str1: string, str2: string): Promise<number> {
  const vec1 = await embed(str1);
  const vec2 = await embed(str2);
  return cosineSimilarity(vec1, vec2);
}

export { embed, stringSimilarity };
