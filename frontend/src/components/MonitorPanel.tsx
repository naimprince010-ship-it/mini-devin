import { useState, useEffect, useCallback } from 'react'

const API = (window as any).__DEVIN_API_BASE__ || '/api'

interface AppMonitorState {
  name: string
  health_url: string
  platform: string
  status: 'healthy' | 'degraded' | 'down' | 'unknown'
  consecutive_failures: number
  last_heal_at: string | null
  check_interval_seconds: number
  enabled: boolean
}

interface HistoryEvent {
  type: string
  app: string
  status?: string
  status_code?: number
  response_ms?: number
  error?: string
  at: string
  logs_excerpt?: string
}

interface MonitorStatus {
  running: boolean
  apps: Record<string, AppMonitorState>
  history: HistoryEvent[]
}

interface RegisterForm {
  name: string
  health_url: string
  platform: string
  interval: number
  failure_threshold: number
}

const defaultForm: RegisterForm = {
  name: '',
  health_url: '',
  platform: 'generic',
  interval: 60,
  failure_threshold: 3,
}

const STATUS_COLORS: Record<string, string> = {
  healthy: 'text-green-400',
  degraded: 'text-yellow-400',
  down: 'text-red-400',
  unknown: 'text-gray-400',
}

const STATUS_DOTS: Record<string, string> = {
  healthy: 'bg-green-400',
  degraded: 'bg-yellow-400',
  down: 'bg-red-400',
  unknown: 'bg-gray-500',
}

export default function MonitorPanel() {
  const [status, setStatus] = useState<MonitorStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState<RegisterForm>(defaultForm)
  const [showForm, setShowForm] = useState(false)
  const [toast, setToast] = useState('')
  const [oneshotUrl, setOneshotUrl] = useState('')
  const [oneshotResult, setOneshotResult] = useState<any>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/monitor/status`)
      if (res.ok) setStatus(await res.json())
    } catch {
      /* network error — silently ignore */
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const t = setInterval(fetchStatus, 10_000)
    return () => clearInterval(t)
  }, [fetchStatus])

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3000)
  }

  const startMonitor = async () => {
    setLoading(true)
    try {
      await fetch(`${API}/monitor/start`, { method: 'POST' })
      showToast('Monitor started')
      await fetchStatus()
    } finally {
      setLoading(false)
    }
  }

  const stopMonitor = async () => {
    setLoading(true)
    try {
      await fetch(`${API}/monitor/stop`, { method: 'POST' })
      showToast('Monitor stopped')
      await fetchStatus()
    } finally {
      setLoading(false)
    }
  }

  const registerApp = async () => {
    if (!form.name || !form.health_url) return showToast('Name and health URL are required')
    setLoading(true)
    try {
      const res = await fetch(`${API}/monitor/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (res.ok) {
        showToast(`Registered ${form.name}`)
        setForm(defaultForm)
        setShowForm(false)
        await fetchStatus()
      } else {
        showToast('Failed to register app')
      }
    } finally {
      setLoading(false)
    }
  }

  const unregisterApp = async (name: string) => {
    await fetch(`${API}/monitor/${name}`, { method: 'DELETE' })
    showToast(`Removed ${name}`)
    await fetchStatus()
  }

  const oneshotCheck = async () => {
    if (!oneshotUrl) return
    setLoading(true)
    try {
      const res = await fetch(`${API}/monitor/health-check?url=${encodeURIComponent(oneshotUrl)}`, {
        method: 'POST',
      })
      setOneshotResult(await res.json())
    } finally {
      setLoading(false)
    }
  }

  const apps = Object.values(status?.apps ?? {})
  const healEvents = (status?.history ?? []).filter(e => e.type === 'heal_triggered').slice(-10)
  const recentChecks = (status?.history ?? []).filter(e => e.type === 'health_check').slice(-20)

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-white overflow-y-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full inline-block ${status?.running ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          Self-Healing Monitor
        </h2>
        <div className="flex gap-2">
          <button
            onClick={status?.running ? stopMonitor : startMonitor}
            disabled={loading}
            className={`text-xs px-3 py-1 rounded ${status?.running ? 'bg-red-900 hover:bg-red-800 text-red-200' : 'bg-green-900 hover:bg-green-800 text-green-200'}`}
          >
            {status?.running ? 'Stop Monitor' : 'Start Monitor'}
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]"
          >
            + Add App
          </button>
          <button onClick={fetchStatus} className="text-xs px-2 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-400 border border-[#333]">
            ↺
          </button>
        </div>
      </div>

      {toast && (
        <div className="text-xs bg-[#1a2e1a] text-green-300 border border-green-800 rounded px-3 py-2">
          {toast}
        </div>
      )}

      {/* Register form */}
      {showForm && (
        <div className="bg-[#141414] border border-[#262626] rounded p-3 space-y-2">
          <p className="text-xs text-gray-400 font-medium">Register App for Monitoring</p>
          <div className="grid grid-cols-2 gap-2">
            <input
              className="col-span-2 bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
              placeholder="App name (e.g. mini-devin-prod)"
              value={form.name}
              onChange={e => setForm({ ...form, name: e.target.value })}
            />
            <input
              className="col-span-2 bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
              placeholder="Health URL (e.g. https://mini-devin.app/api/health)"
              value={form.health_url}
              onChange={e => setForm({ ...form, health_url: e.target.value })}
            />
            <select
              className="bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
              value={form.platform}
              onChange={e => setForm({ ...form, platform: e.target.value })}
            >
              <option value="generic">Generic</option>
              <option value="digitalocean">DigitalOcean</option>
              <option value="railway">Railway</option>
              <option value="docker">Docker</option>
            </select>
            <input
              type="number"
              className="bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
              placeholder="Poll interval (seconds)"
              value={form.interval}
              onChange={e => setForm({ ...form, interval: Number(e.target.value) })}
            />
          </div>
          <div className="flex gap-2 justify-end">
            <button onClick={() => setShowForm(false)} className="text-xs text-gray-500 hover:text-gray-300">
              Cancel
            </button>
            <button
              onClick={registerApp}
              disabled={loading}
              className="text-xs px-3 py-1 rounded bg-blue-900 hover:bg-blue-800 text-blue-200"
            >
              Register
            </button>
          </div>
        </div>
      )}

      {/* Monitored apps */}
      {apps.length === 0 ? (
        <div className="text-xs text-gray-500 text-center py-4">
          No apps registered. Click "+ Add App" to start monitoring a deployment.
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Monitored Apps</p>
          {apps.map(app => (
            <div key={app.name} className="bg-[#141414] border border-[#262626] rounded p-3 flex items-start justify-between gap-2">
              <div className="flex items-start gap-2">
                <span className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${STATUS_DOTS[app.status] || STATUS_DOTS.unknown}`} />
                <div>
                  <p className="text-xs font-medium text-white">{app.name}</p>
                  <p className="text-[10px] text-gray-500">{app.health_url}</p>
                  <div className="flex gap-3 mt-1">
                    <span className={`text-[10px] font-semibold ${STATUS_COLORS[app.status]}`}>
                      {app.status.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-gray-600">{app.platform}</span>
                    {app.consecutive_failures > 0 && (
                      <span className="text-[10px] text-red-400">
                        {app.consecutive_failures} consecutive failures
                      </span>
                    )}
                    {app.last_heal_at && (
                      <span className="text-[10px] text-yellow-600">
                        Last heal: {new Date(app.last_heal_at).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <button
                onClick={() => unregisterApp(app.name)}
                className="text-[10px] text-gray-600 hover:text-red-400 flex-shrink-0"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* One-shot health check */}
      <div className="space-y-1">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Quick Health Check</p>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1"
            placeholder="https://your-app.com/api/health"
            value={oneshotUrl}
            onChange={e => setOneshotUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && oneshotCheck()}
          />
          <button
            onClick={oneshotCheck}
            disabled={loading || !oneshotUrl}
            className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]"
          >
            Check
          </button>
        </div>
        {oneshotResult && (
          <div className={`text-xs rounded p-2 border ${oneshotResult.status === 'healthy' ? 'bg-[#0d1f0d] border-green-800 text-green-300' : 'bg-[#1f0d0d] border-red-800 text-red-300'}`}>
            <span className="font-semibold">{oneshotResult.status?.toUpperCase()}</span>
            {oneshotResult.status_code && <span className="ml-2 text-gray-400">HTTP {oneshotResult.status_code}</span>}
            <span className="ml-2 text-gray-500">{oneshotResult.response_time_ms}ms</span>
            {oneshotResult.error && <span className="ml-2 text-red-400">{oneshotResult.error}</span>}
          </div>
        )}
      </div>

      {/* Auto-heal events */}
      {healEvents.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-yellow-600 font-medium uppercase tracking-wide">⚡ Auto-Heal Events</p>
          {healEvents.map((e, i) => (
            <div key={i} className="bg-[#1a1500] border border-yellow-900 rounded p-2 text-[10px] text-yellow-300">
              <div className="flex justify-between">
                <span className="font-semibold">{e.app}</span>
                <span className="text-gray-500">{new Date(e.at).toLocaleString()}</span>
              </div>
              {e.logs_excerpt && (
                <pre className="mt-1 text-[9px] text-yellow-600 whitespace-pre-wrap line-clamp-3 overflow-hidden">
                  {e.logs_excerpt.slice(0, 300)}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Recent health check history */}
      {recentChecks.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Recent Checks</p>
          <div className="bg-[#141414] border border-[#1e1e1e] rounded divide-y divide-[#1e1e1e]">
            {recentChecks.map((e, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-1.5 text-[10px]">
                <div className="flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOTS[e.status || 'unknown']}`} />
                  <span className="text-gray-400">{e.app}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-600">
                  {e.status_code && <span>HTTP {e.status_code}</span>}
                  {e.response_ms !== undefined && <span>{e.response_ms}ms</span>}
                  {e.error && <span className="text-red-500 truncate max-w-[120px]">{e.error}</span>}
                  <span>{new Date(e.at).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
