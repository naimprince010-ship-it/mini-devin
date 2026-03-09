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
import { useAuth } from './contexts/AuthContext';
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
  Settings
} from 'lucide-react';

type TabType = 'sessions' | 'skills' | 'repos' | 'reviews';

function App() {
  const { user, loading } = useAuth();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('sessions');

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

  return (
    <div className="h-screen flex bg-[#0f0f0f] text-white overflow-hidden">
      {/* 1. Left Sidebar (Navigation) */}
      <aside className="w-[240px] flex flex-col bg-[#121212] border-r border-[#262626] flex-shrink-0">
        <div className="p-4 flex items-center gap-3 mb-4">
          <Bot className="text-[#00ff99]" size={28} />
          <h1 className="font-bold text-lg tracking-tight">Mini-Devin</h1>
        </div>

        <nav className="flex-1 px-2 space-y-1">
          <button
            onClick={() => setActiveTab('sessions')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'sessions'
              ? 'bg-[#00ff99]/10 text-[#00ff99] shadow-sm'
              : 'text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50'
              }`}
          >
            <MessageSquare size={18} />
            Sessions
          </button>
          <button
            onClick={() => setActiveTab('skills')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'skills'
              ? 'bg-[#00ff99]/10 text-[#00ff99] shadow-sm'
              : 'text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50'
              }`}
          >
            <Wrench size={18} />
            Skills
          </button>
          <button
            onClick={() => setActiveTab('repos')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'repos'
              ? 'bg-[#00ff99]/10 text-[#00ff99] shadow-sm'
              : 'text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50'
              }`}
          >
            <GitFork size={18} />
            Repos
          </button>
          <button
            onClick={() => setActiveTab('reviews')}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'reviews'
              ? 'bg-[#00ff99]/10 text-[#00ff99] shadow-sm'
              : 'text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50'
              }`}
          >
            <GitPullRequest size={18} />
            Reviews
          </button>
        </nav>

        {/* Recent Sessions List in Sidebar */}
        {activeTab === 'sessions' && (
          <div className="flex-1 mt-6 px-2 overflow-y-auto min-h-0 border-t border-[#262626] pt-4">
            <SessionList
              onSelectSession={setSelectedSession}
              selectedSessionId={selectedSession?.session_id}
            />
          </div>
        )}

        <div className="p-2 border-t border-[#262626] space-y-1">
          <a
            href="/docs"
            target="_blank"
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50 transition-colors"
          >
            <BookOpen size={18} />
            API Docs
          </a>
          <a
            href="https://github.com"
            target="_blank"
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a]/50 transition-colors"
          >
            <Github size={18} />
            View Source
          </a>
          <div className="pt-2">
            {user ? (
              <UserMenu />
            ) : (
              <button
                onClick={() => setShowLogin(true)}
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium bg-[#1a1a1a] hover:bg-[#262626] transition-colors"
              >
                <LogIn size={18} />
                Sign In
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* 2. Main Panel Container (Center Chat + Right Workspace) */}
      <div className="flex-1 flex overflow-hidden">
        {activeTab === 'sessions' && selectedSession ? (
          <PanelGroup direction="horizontal">
            {/* Center Chat Column */}
            <Panel defaultSize={45} minSize={30}>
              <div className="h-full flex flex-col bg-[#0f0f0f] relative overflow-hidden">
                <TaskPanel session={selectedSession} />
              </div>
            </Panel>

            <PanelResizeHandle className="w-1 bg-[#262626] hover:bg-[#00ff99]/50 transition-colors cursor-col-resize active:bg-[#00ff99]" />

            {/* Right Workspace Column */}
            <Panel minSize={30}>
              <WorkspacePanel sessionId={selectedSession.session_id} />
            </Panel>
          </PanelGroup>
        ) : (
          /* Placeholder / Feature Managers */
          <div className="flex-1 flex flex-col overflow-hidden">
            {activeTab === 'sessions' ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center max-w-sm">
                  <Bot className="mx-auto text-[#1a1a1a] mb-6" size={80} />
                  <h2 className="text-2xl font-bold mb-3 tracking-tight">
                    Welcome to Mini-Devin
                  </h2>
                  <p className="text-[#a3a3a3] text-sm leading-relaxed">
                    Create a new session from the sidebar or select an existing one to start building with your AI agent.
                  </p>
                </div>
              </div>
            ) : activeTab === 'skills' ? (
              <SkillsManager apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'} />
            ) : activeTab === 'repos' ? (
              <RepoManager
                apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'}
                sessionId={selectedSession?.session_id}
                onRepoLinked={() => { }}
              />
            ) : (
              <PRReview
                apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

