import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { ErrorBoundary } from './components/ErrorBoundary';
import { useAuth } from './contexts/AuthContext';
import { useTheme } from './contexts/ThemeContext';
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
  Sun,
  Moon,
  Menu,
} from 'lucide-react';

type TabType = 'sessions' | 'skills' | 'repos' | 'reviews';

function NewSessionModal({
  onClose,
  onCreated,
  initialWorkingDir = '',
}: {
  onClose: () => void;
  onCreated: (session: Session) => void;
  initialWorkingDir?: string;
}) {
  const [workingDir, setWorkingDir] = useState(initialWorkingDir);
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
              <span className="ml-auto text-[10px] text-[#525252] font-normal">empty = safe default workspace</span>
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

const LAST_SESSION_KEY = 'mini-devin:last-session-id';

function App() {
  const { user, loading } = useAuth();
  const { theme, toggleTheme, isDark } = useTheme();
  const navigate = useNavigate();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const { toasts, dismiss } = useToastState();
  const [showNewSession, setShowNewSession] = useState(false);
  const [newSessionInitialDir, setNewSessionInitialDir] = useState('');
  const [activeTab, setActiveTab] = useState<TabType>('sessions');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const api = useApi();

  // Restore last selected session from localStorage on mount
  useEffect(() => {
    const savedId = localStorage.getItem(LAST_SESSION_KEY);
    if (savedId && !selectedSession) {
      api.getSession(savedId)
        .then(s => { if (s) setSelectedSession(s); })
        .catch(() => localStorage.removeItem(LAST_SESSION_KEY));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  const accentColor = isDark ? '#00ff99' : '#00aa66';
  const bgPrimary = isDark ? 'bg-[#0f0f0f]' : 'bg-[#f5f5f5]';
  const bgSidebar = isDark ? 'bg-[#111111]' : 'bg-white';
  const borderColor = isDark ? 'border-[#1a1a1a]' : 'border-[#e5e5e5]';
  const textPrimary = isDark ? 'text-white' : 'text-[#0f0f0f]';
  const textMuted = isDark ? 'text-[#737373]' : 'text-[#737373]';
  const bgHover = isDark ? 'hover:bg-[#1a1a1a]/50' : 'hover:bg-[#f0f0f0]';
  const bgActive = isDark ? 'bg-[#1a1a1a]' : 'bg-[#ebebeb]';

  return (
    <SessionEventsProvider>
      {showNewSession && (
        <NewSessionModal
          onClose={() => { setShowNewSession(false); setNewSessionInitialDir(''); }}
          initialWorkingDir={newSessionInitialDir}
          onCreated={(session) => {
            setSelectedSession(session);
            localStorage.setItem(LAST_SESSION_KEY, session.session_id);
            setActiveTab('sessions');
            setSidebarOpen(false);
          }}
        />
      )}

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className={`h-screen flex ${bgPrimary} ${textPrimary} overflow-hidden`}>
        {/* Left Sidebar — hidden on mobile, slide-in on toggle */}
        <aside
          className={`
            fixed md:relative z-50 md:z-auto
            w-[220px] flex flex-col flex-shrink-0
            ${bgSidebar} border-r ${borderColor}
            h-full
            transition-transform duration-300 ease-in-out
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          `}
        >
          {/* Logo + close button (mobile) */}
          <div className="p-4 flex items-center justify-between mb-2">
            <button
              onClick={() => navigate('/')}
              className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
              title="Back to home"
            >
              <div className="w-7 h-7 rounded-lg bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center justify-center">
                <Bot style={{ color: accentColor }} size={16} />
              </div>
              <h1 className={`font-bold text-sm tracking-tight ${textPrimary}`}>Mini-Devin</h1>
            </button>
            <button
              className={`md:hidden p-1 rounded-lg ${textMuted} ${bgHover} transition-colors`}
              onClick={() => setSidebarOpen(false)}
            >
              <X size={16} />
            </button>
          </div>

          {/* Nav */}
          <nav className="px-2 space-y-0.5">
            {navItems.map(item => (
              <button
                key={item.id}
                onClick={() => { setActiveTab(item.id); setSidebarOpen(false); }}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === item.id
                    ? `${bgActive} ${textPrimary}`
                    : `${textMuted} ${bgHover} hover:${textPrimary}`
                }`}
              >
                <span style={activeTab === item.id ? { color: accentColor } : {}}>{item.icon}</span>
                {item.label}
              </button>
            ))}
            <button
              onClick={handleSelfEvolve}
              disabled={api.loading}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium hover:bg-[#00ff99]/10 transition-colors mt-2"
              style={{ color: accentColor }}
            >
              <Zap size={18} fill="currentColor" fillOpacity={0.2} />
              Self-Evolve
            </button>
          </nav>

          {/* Sessions list */}
          {activeTab === 'sessions' && (
            <div className={`flex-1 mt-4 px-2 overflow-y-auto min-h-0 border-t ${borderColor} pt-4 custom-scrollbar`}>
              <SessionList
                onSelectSession={(s) => {
                  setSelectedSession(s);
                  localStorage.setItem(LAST_SESSION_KEY, s.session_id);
                  setSidebarOpen(false);
                }}
                selectedSessionId={selectedSession?.session_id}
                onNewSession={() => setShowNewSession(true)}
              />
            </div>
          )}

          {/* Bottom links */}
          <div className={`p-2 border-t ${borderColor} space-y-0.5`}>
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${textMuted} ${bgHover} transition-colors`}
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
              {isDark ? 'Light Mode' : 'Dark Mode'}
            </button>
            <a
              href="/docs"
              target="_blank"
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${textMuted} ${bgHover} transition-colors`}
            >
              <BookOpen size={16} />
              API Docs
            </a>
            <a
              href="https://github.com"
              target="_blank"
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${textMuted} ${bgHover} transition-colors`}
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
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${bgActive} hover:${isDark ? 'bg-[#262626]' : 'bg-[#e0e0e0]'} transition-colors ${textMuted}`}
                >
                  <LogIn size={16} />
                  Sign In
                </button>
              )}
            </div>
          </div>
        </aside>

        {/* Main Area */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Mobile top bar */}
          <div className={`flex items-center gap-3 px-4 py-3 border-b ${borderColor} md:hidden flex-shrink-0`}>
            <button
              onClick={() => setSidebarOpen(true)}
              className={`p-2 rounded-lg ${textMuted} ${bgHover} transition-colors`}
            >
              <Menu size={18} />
            </button>
            <div className="flex items-center gap-2">
              <Bot style={{ color: accentColor }} size={16} />
              <span className={`font-bold text-sm ${textPrimary}`}>Mini-Devin</span>
            </div>
            {selectedSession && (
              <span className={`ml-auto text-xs ${textMuted} truncate max-w-[140px]`}>
                {selectedSession.title || 'Session'}
              </span>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 flex overflow-hidden">
            {activeTab === 'sessions' && selectedSession ? (
              <PanelGroup direction="horizontal">
                <Panel defaultSize={45} minSize={30}>
                  <div className={`h-full flex flex-col ${bgPrimary} relative overflow-hidden`}>
                    <ErrorBoundary>
                      <TaskPanel session={selectedSession} onTitleUpdated={handleTitleUpdated} />
                    </ErrorBoundary>
                  </div>
                </Panel>

                <PanelResizeHandle className={`w-px ${isDark ? 'bg-[#1a1a1a]' : 'bg-[#e5e5e5]'} hover:bg-[#00ff99]/30 transition-colors cursor-col-resize hidden md:block`} />

                <Panel minSize={30} className="hidden md:block">
                  <ErrorBoundary>
                    <WorkspacePanel sessionId={selectedSession.session_id} />
                  </ErrorBoundary>
                </Panel>
              </PanelGroup>
            ) : (
              <div className="flex-1 flex flex-col overflow-hidden">
                {activeTab === 'sessions' ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-6 text-center p-8">
                    <div className="w-16 h-16 rounded-2xl bg-[#00ff99]/5 border border-[#00ff99]/10 flex items-center justify-center">
                      <Bot size={32} style={{ color: accentColor, opacity: 0.4 }} />
                    </div>
                    <div className="space-y-2">
                      <h2 className={`text-xl font-bold tracking-tight ${textPrimary}`}>Welcome to Mini-Devin</h2>
                      <p className={`${textMuted} text-sm leading-relaxed max-w-xs`}>
                        Start a new session to work with your AI agent, or select an existing one from the sidebar.
                      </p>
                    </div>
                    <button
                      onClick={() => setShowNewSession(true)}
                      className="px-6 py-3 font-semibold text-sm rounded-xl transition-colors"
                      style={{ backgroundColor: accentColor, color: '#0f0f0f' }}
                    >
                      + New Session
                    </button>
                  </div>
                ) : activeTab === 'skills' ? (
                  <ErrorBoundary>
                    <SkillsManager apiBaseUrl={apiBase} />
                  </ErrorBoundary>
                ) : activeTab === 'repos' ? (
                  <ErrorBoundary>
                    <RepoManager
                      apiBaseUrl={apiBase}
                      sessionId={selectedSession?.session_id}
                      onRepoLinked={() => { }}
                      onOpenInSession={(localPath, repoName) => {
                        setNewSessionInitialDir(localPath);
                        setShowNewSession(true);
                        setActiveTab('sessions');
                      }}
                    />
                  </ErrorBoundary>
                ) : (
                  <ErrorBoundary>
                    <PRReview apiBaseUrl={apiBase} />
                  </ErrorBoundary>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </SessionEventsProvider>
  );
}

export default App;
