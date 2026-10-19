import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('../', import.meta.url));
export const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

function integer(name, fallback, minimum = 1, maximum = 65535) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`Invalid ${name}`);
  return value;
}

export function config() {
  const backendPort = integer('BACKEND_PORT', 8010);
  const researchPort = integer('RESEARCH_PORT', 8011);
  const backendHost = process.env.BACKEND_HOST || '127.0.0.1';
  const researchHost = process.env.RESEARCH_HOST || '127.0.0.1';
  const mode = process.env.RESEARCH_MODE || 'mock';
  if (!['mock', 'external'].includes(mode)) throw new Error('RESEARCH_MODE must be mock or external');
  return {
    backendHost, backendPort, researchHost, researchPort,
    backendUrl: process.env.BACKEND_URL || `http://${backendHost}:${backendPort}`,
    researchUrl: process.env.RESEARCH_URL || `http://${researchHost}:${researchPort}`,
    webHost: process.env.PI_WEB_HOSTNAME || '127.0.0.1',
    webPort: integer('PI_WEB_PORT', 30141),
    agentDir: path.resolve(root, process.env.PI_CODING_AGENT_DIR || '.local/pi-agent'),
    workspaceDir: path.resolve(root, process.env.WORKSPACE_DIR || 'workspace'),
    mode,
  };
}

export async function prepareRuntime() {
  const cfg = config();
  await mkdir(cfg.agentDir, { recursive: true });
  await mkdir(cfg.workspaceDir, { recursive: true });
  const readJson = async name => {
    try { return JSON.parse(await readFile(path.join(cfg.agentDir, name), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return {}; throw error; }
  };
  const models = await readJson('models.json');
  // Preserve other providers and UI preferences in the dedicated runtime directory.
  models.providers = { ...models.providers, routerai: {
    baseUrl: process.env.LLM_BASE_URL || 'https://routerai.ru/api/v1',
    api: 'openai-completions', apiKey: '${ROUTERAI_API_KEY}',
    models: [{ id: process.env.LLM_MODEL || 'deepseek/deepseek-v4.1-flash-1017' }],
  } };
  await writeFile(path.join(cfg.agentDir, 'models.json'), JSON.stringify(models, null, 2) + '\n');
  await writeFile(path.join(cfg.agentDir, 'settings.json'), JSON.stringify({
    ...await readJson('settings.json'),
    defaultProvider: 'routerai', defaultModel: process.env.LLM_MODEL || 'deepseek/deepseek-v4.1-flash-1017',
  }, null, 2) + '\n');
  process.env.PI_CODING_AGENT_DIR = cfg.agentDir;
  process.env.BACKEND_URL = cfg.backendUrl;
  process.env.RESEARCH_URL = cfg.researchUrl;
  process.env.WORKSPACE_DIR = cfg.workspaceDir;
  return cfg;
}
