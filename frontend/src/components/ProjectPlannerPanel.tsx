import { useState, useEffect, useCallback } from 'react'

const API = (window as any).__DEVIN_API_BASE__ || '/api'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Project {
  id: string
  name: string
  description: string
  repo_url?: string
  tech_stack: string[]
  created_at: string
}

interface Milestone {
  id: string
  index: number
  name: string
  description: string
  acceptance_criteria: string[]
  depends_on: string[]
  estimated_hours: number
  tags: string[]
}

interface MilestoneResult {
  milestone_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  summary: string
  task_id?: string
  error?: string
  started_at?: string
  completed_at?: string
}

interface Plan {
  id: string
  project_id: string
  goal: string
  milestones: Milestone[]
  results: Record<string, MilestoneResult>
  status: string
  created_at: string
}

interface MemoryEntry {
  id: string
  project_id: string
  category: string
  title: string
  content: string
  tags: string[]
  importance: number
  created_at: string
}

type MainTab = 'projects' | 'plans' | 'memory'

const CATEGORIES = ['architecture', 'decision', 'constraint', 'api_contract', 'lesson', 'milestone', 'user_preference', 'code_snippet', 'context']
const CAT_COLORS: Record<string, string> = {
  architecture: 'text-blue-400', decision: 'text-purple-400', constraint: 'text-red-400',
  api_contract: 'text-cyan-400', lesson: 'text-yellow-400', milestone: 'text-green-400',
  user_preference: 'text-pink-400', code_snippet: 'text-orange-400', context: 'text-gray-400',
}
const STATUS_ICONS: Record<string, string> = {
  completed: '✅', failed: '❌', running: '🔄', pending: '⏳', skipped: '⏭',
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ProjectPlannerPanel() {
  const [tab, setTab] = useState<MainTab>('projects')
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [plans, setPlans] = useState<Plan[]>([])
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null)
  const [memories, setMemories] = useState<MemoryEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState('')

  // Create project form
  const [showCreateProject, setShowCreateProject] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const [newProjectStack, setNewProjectStack] = useState('')

  // Create plan form
  const [showCreatePlan, setShowCreatePlan] = useState(false)
  const [planGoal, setPlanGoal] = useState('')

  // Add memory form
  const [showAddMemory, setShowAddMemory] = useState(false)
  const [memTitle, setMemTitle] = useState('')
  const [memContent, setMemContent] = useState('')
  const [memCategory, setMemCategory] = useState('decision')
  const [memImportance, setMemImportance] = useState(5)
  const [memTags, setMemTags] = useState('')

  // Memory search
  const [memSearchQ, setMemSearchQ] = useState('')
  const [memSearchResults, setMemSearchResults] = useState<any[]>([])

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 3500) }

  // ── Data fetching ─────────────────────────────────────────────────────────

  const fetchProjects = useCallback(async () => {
    try {
      const res = await fetch(`${API}/projects`)
      if (res.ok) setProjects((await res.json()).projects || [])
    } catch { /* ignore */ }
  }, [])

  const fetchPlans = useCallback(async (projectId?: string) => {
    try {
      const url = projectId ? `${API}/project-plans?project_id=${projectId}` : `${API}/project-plans`
      const res = await fetch(url)
      if (res.ok) setPlans((await res.json()).plans || [])
    } catch { /* ignore */ }
  }, [])

  const fetchMemory = useCallback(async (projectId: string) => {
    try {
      const res = await fetch(`${API}/projects/${projectId}/memory`)
      if (res.ok) setMemories((await res.json()).entries || [])
    } catch { /* ignore */ }
  }, [])

  const refreshPlan = useCallback(async (planId: string) => {
    try {
      const res = await fetch(`${API}/project-plans/${planId}`)
      if (res.ok) {
        const p = await res.json()
        setSelectedPlan(p)
        setPlans(prev => prev.map(x => x.id === planId ? p : x))
      }
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { fetchProjects() }, [fetchProjects])

  useEffect(() => {
    if (tab === 'plans' && selectedProject) fetchPlans(selectedProject.id)
    if (tab === 'memory' && selectedProject) fetchMemory(selectedProject.id)
  }, [tab, selectedProject, fetchPlans, fetchMemory])

  // Auto-refresh running plan
  useEffect(() => {
    if (!selectedPlan || selectedPlan.status !== 'running') return
    const t = setInterval(() => refreshPlan(selectedPlan.id), 5000)
    return () => clearInterval(t)
  }, [selectedPlan, refreshPlan])

  // ── Actions ───────────────────────────────────────────────────────────────

  const createProject = async () => {
    if (!newProjectName) return showToast('Project name required')
    setLoading(true)
    try {
      const res = await fetch(`${API}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProjectName,
          description: newProjectDesc,
          tech_stack: newProjectStack ? newProjectStack.split(',').map(s => s.trim()) : [],
        }),
      })
      if (res.ok) {
        const proj = await res.json()
        showToast(`Project created: ${proj.name}`)
        setShowCreateProject(false)
        setNewProjectName(''); setNewProjectDesc(''); setNewProjectStack('')
        await fetchProjects()
      }
    } finally { setLoading(false) }
  }

  const deleteProject = async (id: string) => {
    await fetch(`${API}/projects/${id}`, { method: 'DELETE' })
    if (selectedProject?.id === id) setSelectedProject(null)
    showToast('Project deleted')
    fetchProjects()
  }

  const createPlan = async () => {
    if (!selectedProject || !planGoal) return showToast('Select a project and enter a goal')
    setLoading(true)
    try {
      const res = await fetch(`${API}/project-plans`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: selectedProject.id, goal: planGoal }),
      })
      if (res.ok) {
        const plan = await res.json()
        showToast(`Plan created with ${plan.milestones?.length ?? 0} milestones`)
        setShowCreatePlan(false)
        setPlanGoal('')
        setSelectedPlan(plan)
        fetchPlans(selectedProject.id)
      } else {
        showToast('Failed to create plan: ' + res.statusText)
      }
    } finally { setLoading(false) }
  }

  const executePlan = async (planId: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/project-plans/${planId}/execute`, { method: 'POST' })
      if (res.ok) {
        showToast('Execution started')
        setTimeout(() => refreshPlan(planId), 2000)
      }
    } finally { setLoading(false) }
  }

  const retryMilestone = async (planId: string, milestoneId: string) => {
    await fetch(`${API}/project-plans/${planId}/retry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ milestone_id: milestoneId }),
    })
    showToast('Milestone reset to pending')
    refreshPlan(planId)
  }

  const addMemory = async () => {
    if (!selectedProject || !memTitle || !memContent) return showToast('Title and content required')
    setLoading(true)
    try {
      const res = await fetch(`${API}/projects/${selectedProject.id}/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: selectedProject.id,
          category: memCategory,
          title: memTitle,
          content: memContent,
          tags: memTags ? memTags.split(',').map(t => t.trim()) : [],
          importance: memImportance,
        }),
      })
      if (res.ok) {
        showToast('Memory stored')
        setShowAddMemory(false)
        setMemTitle(''); setMemContent(''); setMemTags('')
        fetchMemory(selectedProject.id)
      }
    } finally { setLoading(false) }
  }

  const deleteMemory = async (entryId: string) => {
    await fetch(`${API}/memory/${entryId}`, { method: 'DELETE' })
    showToast('Deleted')
    if (selectedProject) fetchMemory(selectedProject.id)
  }

  const searchMemory = async () => {
    if (!selectedProject || !memSearchQ) return
    const res = await fetch(`${API}/projects/${selectedProject.id}/memory/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: selectedProject.id, query: memSearchQ }),
    })
    if (res.ok) setMemSearchResults((await res.json()).results || [])
  }

  // ── Render helpers ────────────────────────────────────────────────────────

  const progressBar = (plan: Plan) => {
    const total = plan.milestones.length
    const done = Object.values(plan.results).filter(r => r.status === 'completed').length
    const pct = total ? Math.round(done / total * 100) : 0
    return { total, done, pct }
  }

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-white overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e1e1e] flex-shrink-0">
        <h2 className="text-sm font-semibold">🗂 Project Planner</h2>
        <div className="flex gap-1">
          {(['projects', 'plans', 'memory'] as MainTab[]).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`text-xs px-3 py-1 rounded capitalize ${tab === t ? 'bg-[#1e1e1e] text-white' : 'text-gray-500 hover:text-gray-300'}`}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {toast && (
        <div className="mx-4 mt-2 text-xs bg-[#1a2e1a] text-green-300 border border-green-800 rounded px-3 py-1.5 flex-shrink-0">{toast}</div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">

        {/* ── PROJECTS TAB ─────────────────────────────────────────────── */}
        {tab === 'projects' && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <p className="text-xs text-gray-500">Select or create a project to get started.</p>
              <button onClick={() => setShowCreateProject(!showCreateProject)}
                className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]">
                + New Project
              </button>
            </div>

            {showCreateProject && (
              <div className="bg-[#141414] border border-[#262626] rounded p-3 space-y-2">
                <input className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                  placeholder="Project name" value={newProjectName} onChange={e => setNewProjectName(e.target.value)} />
                <input className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                  placeholder="Description" value={newProjectDesc} onChange={e => setNewProjectDesc(e.target.value)} />
                <input className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1.5"
                  placeholder="Tech stack (comma-separated): React, FastAPI, PostgreSQL" value={newProjectStack}
                  onChange={e => setNewProjectStack(e.target.value)} />
                <div className="flex justify-end gap-2">
                  <button onClick={() => setShowCreateProject(false)} className="text-xs text-gray-500 hover:text-gray-300">Cancel</button>
                  <button onClick={createProject} disabled={loading}
                    className="text-xs px-3 py-1 rounded bg-blue-900 hover:bg-blue-800 text-blue-200">Create</button>
                </div>
              </div>
            )}

            {projects.length === 0 ? (
              <p className="text-xs text-gray-600 text-center py-4">No projects yet.</p>
            ) : (
              projects.map(p => (
                <div key={p.id} onClick={() => setSelectedProject(p)}
                  className={`cursor-pointer rounded border p-3 transition-colors ${selectedProject?.id === p.id ? 'border-blue-700 bg-[#0d1a2a]' : 'border-[#262626] bg-[#141414] hover:border-[#333]'}`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-xs font-semibold text-white">{p.name}</p>
                      <p className="text-[10px] text-gray-500 mt-0.5">{p.description}</p>
                      {p.tech_stack.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {p.tech_stack.map(t => (
                            <span key={t} className="text-[9px] bg-[#1e1e1e] text-gray-400 px-1.5 py-0.5 rounded">{t}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button onClick={e => { e.stopPropagation(); deleteProject(p.id) }}
                      className="text-[10px] text-gray-600 hover:text-red-400">✕</button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* ── PLANS TAB ─────────────────────────────────────────────────── */}
        {tab === 'plans' && (
          <div className="space-y-3">
            {!selectedProject ? (
              <p className="text-xs text-gray-500 text-center py-4">Select a project first from the Projects tab.</p>
            ) : (
              <>
                <div className="flex justify-between items-center">
                  <p className="text-xs text-gray-400">Plans for <span className="text-white">{selectedProject.name}</span></p>
                  <button onClick={() => setShowCreatePlan(!showCreatePlan)}
                    className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]">
                    + New Plan
                  </button>
                </div>

                {showCreatePlan && (
                  <div className="bg-[#141414] border border-[#262626] rounded p-3 space-y-2">
                    <p className="text-[10px] text-gray-500">Describe the project goal. The AI will break it into milestones.</p>
                    <textarea
                      className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1.5 h-20 resize-none"
                      placeholder="Build a SaaS platform with auth, dashboard, billing, and admin panel"
                      value={planGoal} onChange={e => setPlanGoal(e.target.value)} />
                    <div className="flex justify-end gap-2">
                      <button onClick={() => setShowCreatePlan(false)} className="text-xs text-gray-500 hover:text-gray-300">Cancel</button>
                      <button onClick={createPlan} disabled={loading}
                        className="text-xs px-3 py-1 rounded bg-green-900 hover:bg-green-800 text-green-200">
                        {loading ? '⏳ Generating…' : '🤖 Decompose Goal'}
                      </button>
                    </div>
                  </div>
                )}

                {plans.filter(p => p.project_id === selectedProject.id).map(plan => {
                  const { total, done, pct } = progressBar(plan)
                  const isSelected = selectedPlan?.id === plan.id
                  return (
                    <div key={plan.id} className={`rounded border transition-colors ${isSelected ? 'border-green-800 bg-[#0d1f0d]' : 'border-[#262626] bg-[#141414]'}`}>
                      <div className="p-3 cursor-pointer" onClick={() => setSelectedPlan(isSelected ? null : plan)}>
                        <div className="flex justify-between items-start">
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-white truncate">{plan.goal}</p>
                            <div className="flex items-center gap-3 mt-1">
                              <span className={`text-[10px] ${plan.status === 'completed' ? 'text-green-400' : plan.status === 'failed' ? 'text-red-400' : plan.status === 'running' ? 'text-yellow-400' : 'text-gray-500'}`}>
                                {plan.status}
                              </span>
                              <span className="text-[10px] text-gray-500">{done}/{total} milestones</span>
                            </div>
                          </div>
                          <div className="flex gap-2 ml-2">
                            {plan.status !== 'running' && plan.status !== 'completed' && (
                              <button onClick={e => { e.stopPropagation(); executePlan(plan.id) }}
                                className="text-[10px] px-2 py-0.5 rounded bg-green-900 hover:bg-green-800 text-green-200">
                                ▶ Run
                              </button>
                            )}
                          </div>
                        </div>
                        {/* Progress bar */}
                        <div className="mt-2 bg-[#1e1e1e] rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-green-600 transition-all" style={{ width: `${pct}%` }} />
                        </div>
                      </div>

                      {isSelected && (
                        <div className="border-t border-[#1e1e1e] divide-y divide-[#1e1e1e]">
                          {plan.milestones.map(ms => {
                            const res = plan.results[ms.id]
                            const status = res?.status || 'pending'
                            return (
                              <div key={ms.id} className="px-3 py-2">
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex items-start gap-2">
                                    <span className="text-xs mt-0.5">{STATUS_ICONS[status] || '⏳'}</span>
                                    <div>
                                      <p className="text-[10px] font-medium text-white">{ms.index + 1}. {ms.name}</p>
                                      <p className="text-[9px] text-gray-500 mt-0.5">{ms.description}</p>
                                      {res?.error && <p className="text-[9px] text-red-400 mt-0.5">{res.error.slice(0, 120)}</p>}
                                      {res?.summary && <p className="text-[9px] text-green-500 mt-0.5">{res.summary.slice(0, 120)}</p>}
                                    </div>
                                  </div>
                                  {status === 'failed' && (
                                    <button onClick={() => retryMilestone(plan.id, ms.id)}
                                      className="text-[9px] text-yellow-600 hover:text-yellow-400 flex-shrink-0">↺ Retry</button>
                                  )}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}

                {plans.filter(p => p.project_id === selectedProject.id).length === 0 && (
                  <p className="text-xs text-gray-600 text-center py-4">No plans yet. Create one above.</p>
                )}
              </>
            )}
          </div>
        )}

        {/* ── MEMORY TAB ───────────────────────────────────────────────── */}
        {tab === 'memory' && (
          <div className="space-y-3">
            {!selectedProject ? (
              <p className="text-xs text-gray-500 text-center py-4">Select a project first from the Projects tab.</p>
            ) : (
              <>
                <div className="flex justify-between items-center">
                  <p className="text-xs text-gray-400">Long-term memory for <span className="text-white">{selectedProject.name}</span></p>
                  <button onClick={() => setShowAddMemory(!showAddMemory)}
                    className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]">
                    + Remember
                  </button>
                </div>

                {showAddMemory && (
                  <div className="bg-[#141414] border border-[#262626] rounded p-3 space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <select className="bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
                        value={memCategory} onChange={e => setMemCategory(e.target.value)}>
                        {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <input type="number" min={1} max={10}
                        className="bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
                        placeholder="Importance (1-10)" value={memImportance}
                        onChange={e => setMemImportance(Number(e.target.value))} />
                    </div>
                    <input className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
                      placeholder="Title (e.g. 'Database choice')" value={memTitle}
                      onChange={e => setMemTitle(e.target.value)} />
                    <textarea className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1 h-16 resize-none"
                      placeholder="Content (e.g. 'We chose PostgreSQL over MySQL because of better JSON support and pgvector')"
                      value={memContent} onChange={e => setMemContent(e.target.value)} />
                    <input className="w-full bg-[#0a0a0a] border border-[#333] text-xs text-white rounded px-2 py-1"
                      placeholder="Tags (comma-separated)" value={memTags}
                      onChange={e => setMemTags(e.target.value)} />
                    <div className="flex justify-end gap-2">
                      <button onClick={() => setShowAddMemory(false)} className="text-xs text-gray-500 hover:text-gray-300">Cancel</button>
                      <button onClick={addMemory} disabled={loading}
                        className="text-xs px-3 py-1 rounded bg-purple-900 hover:bg-purple-800 text-purple-200">Save to Memory</button>
                    </div>
                  </div>
                )}

                {/* Semantic search */}
                <div className="flex gap-2">
                  <input className="flex-1 bg-[#141414] border border-[#333] text-xs text-white rounded px-2 py-1"
                    placeholder="Search memory semantically…"
                    value={memSearchQ} onChange={e => setMemSearchQ(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && searchMemory()} />
                  <button onClick={searchMemory}
                    className="text-xs px-3 py-1 rounded bg-[#1e1e1e] hover:bg-[#2a2a2a] text-gray-300 border border-[#333]">
                    Search
                  </button>
                </div>

                {memSearchResults.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] text-gray-500">Search results:</p>
                    {memSearchResults.map((r, i) => {
                      const e = r.entry
                      return (
                        <div key={i} className="bg-[#1a1020] border border-purple-900 rounded p-2 text-[10px]">
                          <div className="flex justify-between">
                            <span className={`font-semibold ${CAT_COLORS[e.category] || 'text-gray-300'}`}>[{e.category}] {e.title}</span>
                            <span className="text-gray-600">score {r.score}</span>
                          </div>
                          <p className="text-gray-400 mt-0.5">{e.content}</p>
                        </div>
                      )
                    })}
                    <button onClick={() => setMemSearchResults([])} className="text-[9px] text-gray-600 hover:text-gray-400">Clear results</button>
                  </div>
                )}

                {/* All entries */}
                <div className="space-y-1">
                  {memories.length === 0 ? (
                    <p className="text-xs text-gray-600 text-center py-4">No memories yet. Store decisions, architecture choices, and lessons learned.</p>
                  ) : (
                    [...memories].sort((a, b) => b.importance - a.importance).map(e => (
                      <div key={e.id} className="bg-[#141414] border border-[#1e1e1e] rounded p-2.5 flex gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className={`text-[9px] font-semibold ${CAT_COLORS[e.category] || 'text-gray-400'}`}>
                              [{e.category}]
                            </span>
                            <span className="text-[10px] text-white font-medium truncate">{e.title}</span>
                            <span className="text-[9px] text-gray-600 ml-auto flex-shrink-0">★{e.importance}</span>
                          </div>
                          <p className="text-[10px] text-gray-400 mt-0.5 line-clamp-2">{e.content}</p>
                          {e.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {e.tags.map(t => <span key={t} className="text-[8px] bg-[#1e1e1e] text-gray-500 px-1 py-0.5 rounded">{t}</span>)}
                            </div>
                          )}
                        </div>
                        <button onClick={() => deleteMemory(e.id)} className="text-[10px] text-gray-600 hover:text-red-400 flex-shrink-0 self-start">✕</button>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
