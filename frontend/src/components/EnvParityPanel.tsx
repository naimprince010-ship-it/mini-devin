import { useState } from 'react'

const API = (window as any).__DEVIN_API_BASE__ || '/api'

type Tab = 'diff' | 'dockerfile' | 'env_example' | 'compose'

interface DiffResult {
  only_local: string[]
  only_production: string[]
  value_mismatch: { key: string; local: string; production: string }[]
  identical_count: number
  summary: string
}

export default function EnvParityPanel() {
  const [tab, setTab] = useState<Tab>('diff')
  const [loading, setLoading] = useState(false)
  const [projectRoot, setProjectRoot] = useState('.')
  const [toast, setToast] = useState('')

  // Diff state
  const [diffResult, setDiffResult] = useState<DiffResult | null>(null)

  // Dockerfile state
  const [dfProjectType, setDfProjectType] = useState('auto')
  const [dfPort, setDfPort] = useState('')
  const [dfOutput, setDfOutput] = useState<{ content: string; path: string } | null>(null)

  // .env.example state
  const [envOutput, setEnvOutput] = useState<{ content: string; path: string } | null>(null)
  const [includeValues, setIncludeValues] = useState(false)

  // docker-compose state
  const [composeRedis, setComposeRedis] = useState(false)
  const [composePg, setComposePg] = useState(false)
  const [composeOutput, setComposeOutput] = useState<{ content: string; path: string } | null>(null)

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3500)
  }

  const runDiff = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/env-parity/diff`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_root: projectRoot }),
      })
      if (res.ok) setDiffResult(await res.json())
      else showToast('Diff failed: ' + res.statusText)
    } finally {
      setLoading(false)
    }
  }

  const generateDockerfile = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/env-parity/generate-dockerfile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_root: projectRoot,
          project_type: dfProjectType,
          port: dfPort ? Number(dfPort) : undefined,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setDfOutput(data)
        showToast(`Dockerfile generated at ${data.path}`)
      } else {
        showToast('Failed: ' + res.statusText)
      }
    } finally {
      setLoading(false)
    }
  }

  const generateEnvExample = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/env-parity/generate-env-example`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_root: projectRoot, include_current_values: includeValues }),
      })
      if (res.ok) {
        const data = await res.json()
        setEnvOutput(data)
        showToast(`.env.example generated at ${data.path}`)
      } else {
        showToast('Failed: ' + res.statusText)
      }
    } finally {
      setLoading(false)
    }
  }

  const generateCompose = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/env-parity/generate-docker-compose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_root: projectRoot,
          include_redis: composeRedis,
          include_postgres: composePg,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setComposeOutput(data)
        showToast(`docker-compose.yml generated at ${data.path}`)
      } else {
        showToast('Failed: ' + res.statusText)
      }
    } finally {
      setLoading(false)
    }
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: 'diff', label: 'Env Diff' },
    { id: 'dockerfile', label: 'Dockerfile' },
    { id: 'env_example', label: '.env.example' },
    { id: 'compose', label: 'docker-compose' },
  ]

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-white overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Environment Parity</h2>
      </div>

      {toast && (
        <div className="text-xs bg-[#1a2e1a] text-green-300 border border-green-800 rounded px-3 py-2">
          {toast}
        </div>
      )}

      {/* Project root */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500 w-24 flex-shrink-0">Project root</label>
        <input
          className="flex-1 bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1"
          value={projectRoot}
          onChange={e => setProjectRoot(e.target.value)}
          placeholder="/path/to/project"
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-[#262626]">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-xs px-3 py-1.5 border-b-2 transition-colors ${
              tab === t.id
                ? 'border-blue-500 text-white'
                : 'border-transparent text-gray-500 hover:text-gray-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab: Env Diff */}
      {tab === 'diff' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            Compare your local <code className="text-blue-400">.env</code> with the current process environment (or production env vars).
          </p>
          <button
            onClick={runDiff}
            disabled={loading}
            className="text-xs px-4 py-1.5 rounded bg-blue-900 hover:bg-blue-800 text-blue-200"
          >
            {loading ? 'Running…' : 'Run Diff'}
          </button>
          {diffResult && (
            <div className="space-y-2">
              <div className="text-xs bg-[#141414] border border-[#262626] rounded p-3">
                <p className="text-gray-300 font-medium">{diffResult.summary}</p>
                <p className="text-gray-500 mt-1">{diffResult.identical_count} identical vars</p>
              </div>

              {diffResult.only_local.length > 0 && (
                <div className="bg-[#141414] border border-yellow-900 rounded p-2">
                  <p className="text-[10px] text-yellow-400 font-medium mb-1">Only in local ({diffResult.only_local.length})</p>
                  {diffResult.only_local.map(k => (
                    <div key={k} className="text-[10px] text-yellow-300 font-mono">{k}</div>
                  ))}
                </div>
              )}

              {diffResult.only_production.length > 0 && (
                <div className="bg-[#141414] border border-orange-900 rounded p-2">
                  <p className="text-[10px] text-orange-400 font-medium mb-1">Only in production ({diffResult.only_production.length})</p>
                  {diffResult.only_production.map(k => (
                    <div key={k} className="text-[10px] text-orange-300 font-mono">{k}</div>
                  ))}
                </div>
              )}

              {diffResult.value_mismatch.length > 0 && (
                <div className="bg-[#141414] border border-red-900 rounded p-2">
                  <p className="text-[10px] text-red-400 font-medium mb-1">Value mismatches ({diffResult.value_mismatch.length})</p>
                  {diffResult.value_mismatch.map(m => (
                    <div key={m.key} className="text-[10px] font-mono mb-1">
                      <span className="text-red-300">{m.key}</span>
                      <div className="pl-2 text-gray-500">local: <span className="text-yellow-600">{m.local}</span></div>
                      <div className="pl-2 text-gray-500">prod:  <span className="text-orange-600">{m.production}</span></div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Tab: Dockerfile */}
      {tab === 'dockerfile' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            Auto-generate a production-ready Dockerfile with health checks and non-root user.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-gray-500 mb-1 block">Project type</label>
              <select
                className="w-full bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1"
                value={dfProjectType}
                onChange={e => setDfProjectType(e.target.value)}
              >
                <option value="auto">Auto-detect</option>
                <option value="python">Python</option>
                <option value="node">Node.js</option>
                <option value="fullstack">Fullstack (Python + Node)</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 mb-1 block">Port (optional)</label>
              <input
                type="number"
                className="w-full bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1"
                placeholder="8000"
                value={dfPort}
                onChange={e => setDfPort(e.target.value)}
              />
            </div>
          </div>
          <button
            onClick={generateDockerfile}
            disabled={loading}
            className="text-xs px-4 py-1.5 rounded bg-blue-900 hover:bg-blue-800 text-blue-200"
          >
            {loading ? 'Generating…' : 'Generate Dockerfile'}
          </button>
          {dfOutput && (
            <div>
              <p className="text-[10px] text-green-400 mb-1">Saved to {dfOutput.path}</p>
              <pre className="bg-[#141414] border border-[#262626] rounded p-3 text-[10px] text-gray-300 overflow-auto max-h-64">
                {dfOutput.content}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Tab: .env.example */}
      {tab === 'env_example' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            Generate <code className="text-blue-400">.env.example</code> from your <code className="text-blue-400">.env</code> — secrets are always blanked.
          </p>
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={includeValues}
              onChange={e => setIncludeValues(e.target.checked)}
              className="rounded"
            />
            Include non-secret values
          </label>
          <button
            onClick={generateEnvExample}
            disabled={loading}
            className="text-xs px-4 py-1.5 rounded bg-blue-900 hover:bg-blue-800 text-blue-200"
          >
            {loading ? 'Generating…' : 'Generate .env.example'}
          </button>
          {envOutput && (
            <div>
              <p className="text-[10px] text-green-400 mb-1">Saved to {envOutput.path}</p>
              <pre className="bg-[#141414] border border-[#262626] rounded p-3 text-[10px] text-gray-300 overflow-auto max-h-64">
                {envOutput.content}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Tab: docker-compose */}
      {tab === 'compose' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            Generate a <code className="text-blue-400">docker-compose.yml</code> for local development that mirrors production.
          </p>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input type="checkbox" checked={composeRedis} onChange={e => setComposeRedis(e.target.checked)} />
              Include Redis
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input type="checkbox" checked={composePg} onChange={e => setComposePg(e.target.checked)} />
              Include PostgreSQL
            </label>
          </div>
          <button
            onClick={generateCompose}
            disabled={loading}
            className="text-xs px-4 py-1.5 rounded bg-blue-900 hover:bg-blue-800 text-blue-200"
          >
            {loading ? 'Generating…' : 'Generate docker-compose.yml'}
          </button>
          {composeOutput && (
            <div>
              <p className="text-[10px] text-green-400 mb-1">Saved to {composeOutput.path}</p>
              <pre className="bg-[#141414] border border-[#262626] rounded p-3 text-[10px] text-gray-300 overflow-auto max-h-64">
                {composeOutput.content}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
