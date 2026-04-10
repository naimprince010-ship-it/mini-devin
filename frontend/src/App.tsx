import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Session } from './types';
import { SessionList } from './components/SessionList';
import { TaskPanel } from './components/TaskPanel';
import { WorkspacePanel } from './components/WorkspacePanel';
import { LoginForm } from './components/LoginForm';
import { UserMenu } from './components/UserMenu';
import { SkillsManager } from './components/SkillsManager';
import { RepoManager } from './components/RepoManager';
import { PRReview } from './components/PRReview';
import { ProviderSelector } from './components/ProviderSelector';
import { FolderPicker } from './components/FolderPicker';
import { useAuth } from './contexts/AuthContext';
import { SessionEventsProvider, useSessionEvents } from './contexts/SessionEventsContext';
import { useApi } from './hooks/useApi';
import { ToastContainer, useToastState } from './components/Toast';
import {
  Bot,
  Github,
  LogIn,
  Loader2,
  MessageSquare,
  Wrench,
  GitFork,
  GitPullRequest,
  BookOpen,
  X,
  Cpu,
  Target,
  Zap,
} from 'lucide-react';

type TabType = 'sessions' | 'skills' | 'repos' | 'reviews';

function NewSessionModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (session: Session) => void;
}) {
  const [workingDir, setWorkingDir] = useState('.');
  const [provider, setProvider] = useState('openai');
  const [model, setModel] = useState('gpt-4o-mini');
  const [maxIterations, setMaxIterations] = useState(50);
  const [acceptanceCriteria, setAcceptanceCriteria] = useState('');
  const [autoGitCommit, setAutoGitCommit] = useState(false);
  const [gitPush, setGitPush] = useState(false);
  const api = useApi();
  const events = useSessionEvents();

  const handleCreate = async () => {
    try {
      const session = await api.createSession({
        working_directory: workingDir,
        model,
        max_iterations: maxIterations,
        auto_git_commit: autoGitCommit,
        git_push: gitPush,
      });
      if (acceptanceCriteria.trim()) {
        events.setAcceptanceCriteria(acceptanceCriteria.trim());
      }
      onCreated(session);
      onClose();
    } catch (e) {
      console.error('Failed to create session:', e);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md bg-[#111111] border border-[#262626] rounded-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262626]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#00ff99]/10 flex items-center justify-center border border-[#00ff99]/20">
              <Bot size={16} className="text-[#00ff99]" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">New Session</h2>
              <p className="text-[10px] text-[#525252]">Configure your agent</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 text-[#525252] hover:text-white rounded-lg hover:bg-[#1a1a1a] transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          {/* Working Directory */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-[#a3a3a3] mb-2">
              <Cpu size={12} className="text-yellow-500/70" />
              Working Directory
            </label>
            <FolderPicker value={workingDir} onChange={setWorkingDir} />
          </div>

          {/* Model */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-[#a3a3a3] mb-2">
              <Cpu size={12} />
              Model
            </label>
            <ProviderSelector
              selectedProvider={provider}
              selectedModel={model}
              onProviderChange={setProvider}
              onModelChange={setModel}
            />
          </div>

          {/* Acceptance Criteria */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-[#a3a3a3] mb-2">
              <Target size={12} />
              What counts as success? <span className="text-[#525252]">(optional)</span>
            </label>
            <textarea
              value={acceptanceCriteria}
              onChange={(e) => setAcceptanceCriteria(e.target.value)}
              className="w-full px-3 py-2.5 bg-[#0a0a0a] text-white text-sm rounded-lg border border-[#262626] focus:border-[#00ff99]/40 focus:outline-none transition-colors placeholder-[#525252] resize-none"
              placeholder="e.g. A working Python script that prints Hello World"
              rows={2}
            />
          </div>

          {/* Max Iterations */}
          <div>
            <label className="flex items-center justify-between text-xs font-medium text-[#a3a3a3] mb-2">
              <span>Max Iterations</span>
              <span className="text-[#00ff99] font-mono">{maxIterations}</span>
            </label>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value))}
              className="w-full accent-[#00ff99]"
            />
            <div className="flex justify-between text-[10px] text-[#525252] mt-1">
              <span>5</span>
              <span>100</span>
            </div>
          </div>

          {/* Git Options */}
          <div className="space-y-2 pt-1 border-t border-[#1a1a1a]">
            <label className="flex items-center gap-1.5 text-xs font-medium text-[#a3a3a3]">
              <Github size={12} />
              Git Options
            </label>
            <label className="flex items-center gap-2.5 cursor-pointer group">
              <div
                onClick={() => setAutoGitCommit(v => !v)}
                className={`relative w-8 h-4 rounded-full transition-colors ${autoGitCommit ? 'bg-[#00ff99]' : 'bg-[#262626]'}`}
              >
                <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform ${autoGitCommit ? 'translate-x-4' : ''}`} />
              </div>
              <span className="text-xs text-[#a3a3a3] group-hover:text-white transition-colors">Auto git commit on task complete</span>
            </label>
            {autoGitCommit && (
              <label className="flex items-center gap-2.5 cursor-pointer group ml-2">
                <div
                  onClick={() => setGitPush(v => !v)}
                  className={`relative w-8 h-4 rounded-full transition-colors ${gitPush ? 'bg-[#00ff99]' : 'bg-[#262626]'}`}
                >
                  <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform ${gitPush ? 'translate-x-4' : ''}`} />
                </div>
                <span className="text-xs text-[#a3a3a3] group-hover:text-white transition-colors">Push to remote after commit</span>
              </label>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex gap-3 px-6 py-4 border-t border-[#262626]">
          <button
            onClick={onClose}
            className="flex-1 py-2.5 rounded-lg border border-[#262626] text-sm text-[#a3a3a3] hover:text-white hover:border-[#363636] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={api.loading}
            className="flex-1 py-2.5 rounded-lg bg-[#00ff99] text-[#0f0f0f] text-sm font-semibold hover:bg-[#00e589] disabled:opacity-50 transition-colors"
          >
            {api.loading ? 'Creating...' : 'Start Session'}
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const { user, loading } = useAuth();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const { toasts, dismiss } = useToastState();
  const [showNewSession, setShowNewSession] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('sessions');
  const api = useApi();

  const handleSelfEvolve = async () => {
    try {
      const result = await api.triggerSelfEvolution();
      // Handle success - maybe select the new session
      console.log('Self-evolution started:', result);
    } catch (e) {
      console.error('Failed to trigger self-evolution:', e);
    }
  };

  // Real-time title from WebSocket (session_title_updated)
  const handleTitleUpdated = (title: string) => {
    setSelectedSession(prev => prev ? { ...prev, title } : prev);
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-[#0f0f0f]">
        <Loader2 className="animate-spin text-[#00ff99]" size={48} />
      </div>
    );
  }

  if (showLogin && !user) {
    return <LoginForm />;
  }

  const navItems: { id: TabType; icon: React.ReactNode; label: string }[] = [
    { id: 'sessions', icon: <MessageSquare size={18} />, label: 'Sessions' },
    { id: 'skills', icon: <Wrench size={18} />, label: 'Skills' },
    { id: 'repos', icon: <GitFork size={18} />, label: 'Repos' },
    { id: 'reviews', icon: <GitPullRequest size={18} />, label: 'Reviews' },
  ];

  const apiBase = import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL}/api`
    : 'http://localhost:8000/api';

  return (
    <SessionEventsProvider>
      {showNewSession && (
        <NewSessionModal
          onClose={() => setShowNewSession(false)}
          onCreated={(session) => {
            setSelectedSession(session);
            setActiveTab('sessions');
          }}
        />
      )}

      <div className="h-screen flex bg-[#0f0f0f] text-white overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[220px] flex flex-col bg-[#111111] border-r border-[#1a1a1a] flex-shrink-0">
          {/* Logo */}
          <div className="p-4 flex items-center gap-2.5 mb-2">
            <div className="w-7 h-7 rounded-lg bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center justify-center">
              <Bot className="text-[#00ff99]" size={16} />
            </div>
            <h1 className="font-bold text-sm tracking-tight text-white">Mini-Devin</h1>
          </div>

          {/* Nav */}
          <nav className="px-2 space-y-0.5">
            {navItems.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === item.id
                  ? 'bg-[#1a1a1a] text-white'
                  : 'text-[#737373] hover:text-white hover:bg-[#1a1a1a]/50'
                  }`}
              >
                <span className={activeTab === item.id ? 'text-[#00ff99]' : ''}>{item.icon}</span>
                {item.label}
              </button>
            ))}
            <button
              onClick={handleSelfEvolve}
              disabled={api.loading}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-[#00ff99] hover:bg-[#00ff99]/10 transition-colors mt-2"
            >
              <Zap size={18} fill="currentColor" fillOpacity={0.2} />
              Self-Evolve
            </button>
          </nav>

          {/* Sessions list */}
          {activeTab === 'sessions' && (
            <div className="flex-1 mt-4 px-2 overflow-y-auto min-h-0 border-t border-[#1a1a1a] pt-4 custom-scrollbar">
              <SessionList
                onSelectSession={setSelectedSession}
                selectedSessionId={selectedSession?.session_id}
                onNewSession={() => setShowNewSession(true)}
              />
            </div>
          )}

          {/* Bottom links */}
          <div className="p-2 border-t border-[#1a1a1a] space-y-0.5">
            <a
              href="/docs"
              target="_blank"
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-[#737373] hover:text-white hover:bg-[#1a1a1a]/50 transition-colors"
            >
              <BookOpen size={16} />
              API Docs
            </a>
            <a
              href="https://github.com"
              target="_blank"
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-[#737373] hover:text-white hover:bg-[#1a1a1a]/50 transition-colors"
            >
              <Github size={16} />
              View Source
            </a>
            <div className="pt-1">
              {user ? (
                <UserMenu />
              ) : (
                <button
                  onClick={() => setShowLogin(true)}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium bg-[#1a1a1a] hover:bg-[#262626] transition-colors text-[#a3a3a3]"
                >
                  <LogIn size={16} />
                  Sign In
                </button>
              )}
            </div>
          </div>
        </aside>

        {/* Main Area */}
        <div className="flex-1 flex overflow-hidden">
          {activeTab === 'sessions' && selectedSession ? (
            <PanelGroup direction="horizontal">
              <Panel defaultSize={45} minSize={30}>
                <div className="h-full flex flex-col bg-[#0f0f0f] relative overflow-hidden">
                  <TaskPanel session={selectedSession} onTitleUpdated={handleTitleUpdated} />
                </div>
              </Panel>

              <PanelResizeHandle className="w-px bg-[#1a1a1a] hover:bg-[#00ff99]/30 transition-colors cursor-col-resize" />

              <Panel minSize={30}>
                <WorkspacePanel sessionId={selectedSession.session_id} />
              </Panel>
            </PanelGroup>
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden">
              {activeTab === 'sessions' ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-6 text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-[#00ff99]/5 border border-[#00ff99]/10 flex items-center justify-center">
                    <Bot size={32} className="text-[#00ff99]/40" />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-xl font-bold tracking-tight">Welcome to Mini-Devin</h2>
                    <p className="text-[#737373] text-sm leading-relaxed max-w-xs">
                      Start a new session to work with your AI agent, or select an existing one from the sidebar.
                    </p>
                  </div>
                  <button
                    onClick={() => setShowNewSession(true)}
                    className="px-6 py-3 bg-[#00ff99] text-[#0f0f0f] font-semibold text-sm rounded-xl hover:bg-[#00e589] transition-colors"
                  >
                    + New Session
                  </button>
                </div>
              ) : activeTab === 'skills' ? (
                <SkillsManager apiBaseUrl={apiBase} />
              ) : activeTab === 'repos' ? (
                <RepoManager
                  apiBaseUrl={apiBase}
                  sessionId={selectedSession?.session_id}
                  onRepoLinked={() => { }}
                />
              ) : (
                <PRReview apiBaseUrl={apiBase} />
              )}
            </div>
          )}
        </div>
      </div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </SessionEventsProvider>
  );
}

export default App;
