import express from 'express';
import dotenv from 'dotenv';
import formidable from 'formidable';
import path from 'path';
import fs from 'fs';

dotenv.config();

const HOST_AGENT_URL = process.env.HOST_AGENT_URL || 'http://localhost:8000';
const UPLOAD_FOLDER = process.env.UPLOAD_FOLDER || 'uploaded_pdfs';

fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });

const app = express();
app.use(express.json());

app.post('/upload', (req, res) => {
  const form = formidable({ multiples: false, uploadDir: UPLOAD_FOLDER, keepExtensions: true });
  form.parse(req, (err, fields, files) => {
    if (err) return res.status(500).json({ error: String(err) });
    // TODO: atualizar metadados CSV, indexar no Chroma etc.
    return res.json({ status: 'uploaded', files });
  });
});

app.post('/chat', async (req, res) => {
  const { query, uuid } = req.body || {};
  if (!query) return res.status(400).json({ error: 'query required' });
  try {
    const resp = await fetch(`${HOST_AGENT_URL}/query`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ query, uuid }) } as any);
    const data = await resp.json();
    return res.json(data);
  } catch (e: any) {
    return res.status(502).json({ error: e.message || e });
  }
});

const PORT = parseInt(process.env.UI_PORT || '8501', 10);
app.listen(PORT, () => console.log(`Interface (UI) stub rodando em http://0.0.0.0:${PORT}`));
