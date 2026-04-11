import { useState, useEffect } from 'react'

const API = (window as any).__DEVIN_API_BASE__ || '/api'

// ─── Types ───────────────────────────────────────────────────────────────────

interface TestStep {
  type: string
  selector?: string
  url?: string
  text?: string
  value?: string
  script?: string
  screenshot_name?: string
  set_baseline?: boolean
  threshold_percent?: number
  ms?: number
  description?: string
}

interface StepResult {
  step_index: number
  step_type: string
  description: string
  passed: boolean
  duration_ms: number
  error?: string
  screenshot_b64?: string
  diff?: {
    changed_percent: number
    threshold_percent: number
    passed: boolean
    diff_path?: string
  }
}

interface SuiteResult {
  suite_name: string
  url: string
  passed: boolean
  total_steps: number
  passed_steps: number
  failed_steps: number
  duration_ms: number
  steps: StepResult[]
  js_errors: string[]
  ran_at: string
}

interface Baseline {
  name: string
  url?: string
  width?: number
  height?: number
  captured_at?: string
  png_path?: string
}

// ─── Step type options ────────────────────────────────────────────────────────

const STEP_TYPES = [
  { value: 'navigate', label: 'Navigate to URL' },
  { value: 'assert_element', label: 'Assert element exists' },
  { value: 'assert_text', label: 'Assert element text' },
  { value: 'assert_url', label: 'Assert URL contains' },
  { value: 'assert_title', label: 'Assert page title' },
  { value: 'click', label: 'Click element' },
  { value: 'fill', label: 'Fill input' },
  { value: 'select', label: 'Select option' },
  { value: 'press_key', label: 'Press key' },
  { value: 'wait', label: 'Wait (ms)' },
  { value: 'wait_for_selector', label: 'Wait for element' },
  { value: 'screenshot', label: 'Screenshot (visual regression)' },
  { value: 'evaluate', label: 'Run JavaScript' },
  { value: 'assert_no_js_errors', label: 'Assert no JS errors' },
  { value: 'hover', label: 'Hover element' },
  { value: 'scroll', label: 'Scroll' },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function stepIcon(type: string): string {
  const map: Record<string, string> = {
    navigate: '🌐', click: '🖱', fill: '⌨️', assert_element: '🔍',
    assert_text: '📝', assert_url: '🔗', assert_title: '🏷', screenshot: '📷',
    evaluate: '⚙️', assert_no_js_errors: '🛡', wait: '⏱', scroll: '↕️',
    hover: '👆', select: '📋', press_key: '⌨️', wait_for_selector: '⏳',
  }
  return map[type] || '▶'
}

function emptyStep(): TestStep {
  return { type: 'navigate', description: '' }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function UITestPanel() {
  const [tab, setTab] = useState<'run' | 'baselines'>('run')
  const [suiteName, setSuiteName] = useState('My UI Test')
  const [startUrl, setStartUrl] = useState('')
  const [steps, setSteps] = useState<TestStep[]>([
    { type: 'assert_element', selector: 'body', description: 'Page loaded' },
    { type: 'screenshot', screenshot_name: 'homepage', description: 'Capture homepage' },
  ])
  const [threshold, setThreshold] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SuiteResult | null>(null)
  const [baselines, setBaselines] = useState<Baseline[]>([])
  const [expandedStep, setExpandedStep] = useState<number | null>(null)
  const [toast, setToast] = useState('')

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3500)
  }

  const fetchBaselines = async () => {
    try {
      const res = await fetch(`${API}/visual-regression/baselines`)
      if (res.ok) {
        const data = await res.json()
        setBaselines(data.baselines || [])
      }
    } catch { /* ignore */ }
  }

  useEffect(() => {
    if (tab === 'baselines') fetchBaselines()
  }, [tab])

  const addStep = () => setSteps([...steps, emptyStep()])
  const removeStep = (i: number) => setSteps(steps.filter((_, idx) => idx !== i))
  const updateStep = (i: number, patch: Partial<TestStep>) =>
    setSteps(steps.map((s, idx) => idx === i ? { ...s, ...patch } : s))
  const moveStep = (i: number, dir: -1 | 1) => {
    const next = [...steps]
    const j = i + dir
    if (j < 0 || j >= next.length) return
    ;[next[i], next[j]] = [next[j], next[i]]
    setSteps(next)
  }

  const runTest = async () => {
    if (!startUrl) return showToast('Enter a start URL')
    if (steps.length === 0) return showToast('Add at least one step')
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${API}/ui-test/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          suite_name: suiteName,
          url: startUrl,
          steps,
          threshold_percent: threshold,
          working_dir: '.',
        }),
      })
      if (res.ok) {
        const data: SuiteResult = await res.json()
        setResult(data)
        if (tab === 'baselines') fetchBaselines()
      } else {
        const err = await res.text()
        showToast('Test failed: ' + err.slice(0, 200))
      }
    } finally {
      setLoading(false)
    }
  }

  const deleteBaseline = async (name: string) => {
    await fetch(`${API}/visual-regression/baseline/${name}`, { method: 'DELETE' })
    showToast(`Deleted baseline: ${name}`)
    fetchBaselines()
  }

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-white overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e1e1e] flex-shrink-0">
        <h2 className="text-sm font-semibold flex items-center gap-2">
          🧪 Browser UI Testing
        </h2>
        <div className="flex gap-1">
          {(['run', 'baselines'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-xs px-3 py-1 rounded transition-colors ${
                tab === t ? 'bg-[#1e1e1e] text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {t === 'run' ? '▶ Run Tests' : '📸 Baselines'}
            </button>
          ))}
        </div>
      </div>

      {toast && (
        <div className="mx-4 mt-2 text-xs bg-[#1a2e1a] text-green-300 border border-green-800 rounded px-3 py-2 flex-shrink-0">
          {toast}
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {tab === 'run' && (
          <div className="p-4 space-y-4">
            {/* Suite config */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-gray-500 block mb-1">Suite name</label>
                <input
                  className="w-full bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                  value={suiteName}
                  onChange={e => setSuiteName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-[10px] text-gray-500 block mb-1">Visual diff threshold (%)</label>
                <input
                  type="number"
                  step="0.1"
                  className="w-full bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                  value={threshold}
                  onChange={e => setThreshold(Number(e.target.value))}
                />
              </div>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 block mb-1">Start URL</label>
              <input
                className="w-full bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                placeholder="https://your-app.com"
                value={startUrl}
                onChange={e => setStartUrl(e.target.value)}
              />
            </div>

            {/* Steps */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-[10px] text-gray-500 font-medium uppercase tracking-wide">Test Steps</p>
                <button
                  onClick={addStep}
                  className="text-[10px] px-2 py-0.5 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]"
                >
                  + Add Step
                </button>
              </div>

              {steps.map((step, i) => (
                <div key={i} className="bg-[#141414] border border-[#262626] rounded">
                  <div
                    className="flex items-center gap-2 px-3 py-2 cursor-pointer"
                    onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                  >
                    <span className="text-xs w-5 text-center">{stepIcon(step.type)}</span>
                    <span className="text-[10px] text-gray-300 font-medium flex-1 truncate">
                      {step.description || `${step.type}${step.selector ? ` [${step.selector}]` : ''}`}
                    </span>
                    <div className="flex gap-1 ml-auto">
                      <button onClick={e => { e.stopPropagation(); moveStep(i, -1) }} className="text-[10px] text-gray-600 hover:text-gray-300 px-1">↑</button>
                      <button onClick={e => { e.stopPropagation(); moveStep(i, 1) }} className="text-[10px] text-gray-600 hover:text-gray-300 px-1">↓</button>
                      <button onClick={e => { e.stopPropagation(); removeStep(i) }} className="text-[10px] text-gray-600 hover:text-red-400 px-1">✕</button>
                    </div>
                  </div>

                  {expandedStep === i && (
                    <div className="px-3 pb-3 space-y-2 border-t border-[#1e1e1e] pt-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">Step type</label>
                          <select
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            value={step.type}
                            onChange={e => updateStep(i, { type: e.target.value })}
                          >
                            {STEP_TYPES.map(t => (
                              <option key={t.value} value={t.value}>{t.label}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">Description</label>
                          <input
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            placeholder="Human-readable label"
                            value={step.description || ''}
                            onChange={e => updateStep(i, { description: e.target.value })}
                          />
                        </div>
                      </div>

                      {/* Conditional fields */}
                      {['assert_element', 'assert_text', 'click', 'fill', 'select', 'wait_for_selector', 'hover', 'scroll', 'press_key'].includes(step.type) && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">CSS Selector</label>
                          <input
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1 font-mono"
                            placeholder="button#submit, .hero-title, [data-testid='btn']"
                            value={step.selector || ''}
                            onChange={e => updateStep(i, { selector: e.target.value })}
                          />
                        </div>
                      )}

                      {['navigate', 'assert_url'].includes(step.type) && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">URL</label>
                          <input
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            placeholder="https://..."
                            value={step.url || ''}
                            onChange={e => updateStep(i, { url: e.target.value })}
                          />
                        </div>
                      )}

                      {['assert_text', 'assert_title'].includes(step.type) && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">Expected text (contains)</label>
                          <input
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            value={step.text || ''}
                            onChange={e => updateStep(i, { text: e.target.value })}
                          />
                        </div>
                      )}

                      {['fill', 'select', 'press_key'].includes(step.type) && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">Value</label>
                          <input
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            placeholder={step.type === 'press_key' ? 'Enter, Tab, Escape...' : 'value'}
                            value={step.value || ''}
                            onChange={e => updateStep(i, { value: e.target.value })}
                          />
                        </div>
                      )}

                      {step.type === 'wait' && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">Wait (ms)</label>
                          <input
                            type="number"
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                            value={step.ms || 500}
                            onChange={e => updateStep(i, { ms: Number(e.target.value) })}
                          />
                        </div>
                      )}

                      {step.type === 'evaluate' && (
                        <div>
                          <label className="text-[10px] text-gray-500 block mb-0.5">JavaScript</label>
                          <textarea
                            className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1 font-mono h-16 resize-none"
                            placeholder="document.title"
                            value={step.script || ''}
                            onChange={e => updateStep(i, { script: e.target.value })}
                          />
                        </div>
                      )}

                      {step.type === 'screenshot' && (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-[10px] text-gray-500 block mb-0.5">Screenshot name</label>
                            <input
                              className="w-full bg-[#0a0a0a] border border-[#333] text-[10px] text-white rounded px-2 py-1"
                              placeholder="homepage"
                              value={step.screenshot_name || ''}
                              onChange={e => updateStep(i, { screenshot_name: e.target.value })}
                            />
                          </div>
                          <div className="flex items-end gap-2 pb-1">
                            <label className="flex items-center gap-1.5 text-[10px] text-gray-400 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={!!step.set_baseline}
                                onChange={e => updateStep(i, { set_baseline: e.target.checked })}
                              />
                              Set as new baseline
                            </label>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Run button */}
            <button
              onClick={runTest}
              disabled={loading}
              className="w-full py-2 rounded text-sm font-semibold transition-colors bg-green-900 hover:bg-green-800 text-green-200 disabled:opacity-50"
            >
              {loading ? '⏳ Running tests…' : '▶ Run Test Suite'}
            </button>

            {/* Results */}
            {result && (
              <div className="space-y-3">
                <div className={`rounded p-3 border text-sm font-semibold ${result.passed ? 'bg-[#0d1f0d] border-green-800 text-green-300' : 'bg-[#1f0d0d] border-red-800 text-red-300'}`}>
                  {result.passed ? '✅' : '❌'} {result.suite_name}: {result.passed_steps}/{result.total_steps} passed
                  <span className="text-[10px] font-normal ml-2 text-gray-500">in {result.duration_ms}ms</span>
                </div>

                {result.js_errors.length > 0 && (
                  <div className="bg-[#1a1000] border border-yellow-900 rounded p-2 text-[10px] text-yellow-300">
                    <p className="font-medium mb-1">JS Errors ({result.js_errors.length})</p>
                    {result.js_errors.slice(0, 3).map((e, i) => <div key={i} className="truncate">{e}</div>)}
                  </div>
                )}

                <div className="space-y-1">
                  {result.steps.map((sr, i) => (
                    <div key={i} className={`rounded border text-[10px] ${sr.passed ? 'bg-[#0a1a0a] border-green-900' : 'bg-[#1a0a0a] border-red-900'}`}>
                      <div className="flex items-start gap-2 px-3 py-2">
                        <span>{sr.passed ? '✅' : '❌'}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-gray-400 font-mono">[{sr.step_type}]</span>
                            <span className="text-gray-300 truncate">{sr.description}</span>
                            <span className="text-gray-600 ml-auto flex-shrink-0">{sr.duration_ms}ms</span>
                          </div>
                          {sr.error && <div className="text-red-400 mt-0.5">{sr.error}</div>}
                          {sr.diff && (
                            <div className={`mt-0.5 ${sr.diff.passed ? 'text-green-600' : 'text-red-400'}`}>
                              Visual: {sr.diff.changed_percent?.toFixed(3)}% changed (limit {sr.diff.threshold_percent}%)
                            </div>
                          )}
                        </div>
                      </div>
                      {sr.screenshot_b64 && (
                        <div className="px-3 pb-2">
                          <img
                            src={`data:image/png;base64,${sr.screenshot_b64}`}
                            alt="step screenshot"
                            className="w-full max-w-sm rounded border border-[#333] opacity-80 hover:opacity-100 transition-opacity"
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {tab === 'baselines' && (
          <div className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-400">Stored visual regression baselines</p>
              <button onClick={fetchBaselines} className="text-[10px] text-gray-500 hover:text-gray-300 px-2 py-1 rounded bg-[#141414] border border-[#262626]">↺ Refresh</button>
            </div>

            {baselines.length === 0 ? (
              <div className="text-xs text-gray-600 text-center py-8">
                No baselines yet.<br />
                Run a test with a <code className="text-gray-400">screenshot</code> step to capture baselines.
              </div>
            ) : (
              <div className="space-y-2">
                {baselines.map(b => (
                  <div key={b.name} className="bg-[#141414] border border-[#262626] rounded p-3 flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-medium text-white">{b.name}</p>
                      <p className="text-[10px] text-gray-500">
                        {b.width}×{b.height} &nbsp;|&nbsp; {b.captured_at?.slice(0, 19).replace('T', ' ')}
                      </p>
                      {b.url && <p className="text-[10px] text-blue-600 truncate max-w-xs">{b.url}</p>}
                    </div>
                    <button
                      onClick={() => deleteBaseline(b.name)}
                      className="text-[10px] text-gray-600 hover:text-red-400 flex-shrink-0"
                    >
                      🗑 Delete
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
