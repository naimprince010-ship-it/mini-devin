import { useState } from 'react';
import { Session } from './types';
import { SessionList } from './components/SessionList';
import { TaskPanel } from './components/TaskPanel';
import { StatusBar } from './components/StatusBar';
import { LoginForm } from './components/LoginForm';
import { UserMenu } from './components/UserMenu';
import { SkillsManager } from './components/SkillsManager';
import { RepoManager } from './components/RepoManager';
import { PRReview } from './components/PRReview';
import { useAuth } from './contexts/AuthContext';
import { Bot, Github, LogIn, Loader2, MessageSquare, Wrench, GitFork, GitPullRequest } from 'lucide-react';

type TabType = 'sessions' | 'skills' | 'repos' | 'reviews';

function App() {
  const { user, loading } = useAuth();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('sessions');

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-900">
        <Loader2 className="animate-spin text-blue-400" size={48} />
      </div>
    );
  }

  if (showLogin && !user) {
    return <LoginForm />;
  }

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Bot className="text-blue-400" size={28} />
          <h1 className="text-xl font-bold">Mini-Devin</h1>
          <span className="text-gray-400 text-sm">Autonomous AI Software Engineer</span>
        </div>
        <div className="flex items-center gap-4">
          <a
            href="/docs"
            target="_blank"
            className="text-gray-400 hover:text-white text-sm"
          >
            API Docs
          </a>
          <a
            href="https://github.com"
            target="_blank"
            className="text-gray-400 hover:text-white"
          >
            <Github size={20} />
          </a>
          {user ? (
            <UserMenu />
          ) : (
            <button
              onClick={() => setShowLogin(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 transition-colors text-sm"
            >
              <LogIn size={16} />
              Sign In
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar with Tabs */}
        <aside className="w-80 border-r border-gray-700 flex flex-col">
          {/* Tab Navigation */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setActiveTab('sessions')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'sessions'
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800/50'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/30'
              }`}
            >
              <MessageSquare size={16} />
              Sessions
            </button>
                      <button
                        onClick={() => setActiveTab('skills')}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                          activeTab === 'skills'
                            ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800/50'
                            : 'text-gray-400 hover:text-white hover:bg-gray-800/30'
                        }`}
                      >
                        <Wrench size={16} />
                        Skills
                      </button>
                      <button
                        onClick={() => setActiveTab('repos')}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                          activeTab === 'repos'
                            ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800/50'
                            : 'text-gray-400 hover:text-white hover:bg-gray-800/30'
                        }`}
                      >
                        <GitFork size={16} />
                        Repos
                      </button>
                      <button
                        onClick={() => setActiveTab('reviews')}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                          activeTab === 'reviews'
                            ? 'text-purple-400 border-b-2 border-purple-400 bg-gray-800/50'
                            : 'text-gray-400 hover:text-white hover:bg-gray-800/30'
                        }`}
                      >
                        <GitPullRequest size={16} />
                        Reviews
                      </button>
                    </div>
          
                    {/* Tab Content */}
                    <div className="flex-1 overflow-y-auto">
                      {activeTab === 'sessions' ? (
                        <div className="p-4">
                          <SessionList
                            onSelectSession={setSelectedSession}
                            selectedSessionId={selectedSession?.session_id}
                          />
                        </div>
                      ) : activeTab === 'skills' ? (
                        <div className="p-4">
                          <SkillsManager apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'} />
                        </div>
                      ) : activeTab === 'repos' ? (
                        <RepoManager 
                          apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'}
                          sessionId={selectedSession?.session_id}
                          onRepoLinked={() => {}}
                        />
                      ) : (
                        <PRReview 
                          apiBaseUrl={import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : 'http://localhost:8000/api'}
                        />
                      )}
                    </div>
        </aside>

                {/* Main Panel */}
                <main className="flex-1 flex flex-col overflow-hidden">
                  {activeTab === 'sessions' && selectedSession ? (
                    <TaskPanel session={selectedSession} />
                  ) : activeTab === 'sessions' ? (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <Bot className="mx-auto text-gray-600 mb-4" size={64} />
                        <h2 className="text-xl font-medium text-gray-400 mb-2">
                          Welcome to Mini-Devin
                        </h2>
                        <p className="text-gray-500 max-w-md">
                          Create a new session or select an existing one to start working with your autonomous AI software engineer.
                        </p>
                      </div>
                    </div>
                  ) : activeTab === 'skills' ? (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <Wrench className="mx-auto text-gray-600 mb-4" size={64} />
                        <h2 className="text-xl font-medium text-gray-400 mb-2">
                          Skills Manager
                        </h2>
                        <p className="text-gray-500 max-w-md">
                          Create, edit, and manage custom skills for your AI agent. Skills are reusable procedures that can be executed on demand.
                        </p>
                      </div>
                    </div>
                  ) : activeTab === 'repos' ? (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <GitFork className="mx-auto text-gray-600 mb-4" size={64} />
                        <h2 className="text-xl font-medium text-gray-400 mb-2">
                          GitHub Repositories
                        </h2>
                        <p className="text-gray-500 max-w-md">
                          Connect GitHub repositories to Mini-Devin. The AI agent can clone, edit, commit, push, and create pull requests on your repos.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <GitPullRequest className="mx-auto text-purple-600 mb-4" size={64} />
                        <h2 className="text-xl font-medium text-gray-400 mb-2">
                          AI Code Review
                        </h2>
                        <p className="text-gray-500 max-w-md">
                          Analyze pull requests with AI-powered code review. Get inline suggestions, approve or request changes, and submit reviews directly to GitHub.
                        </p>
                      </div>
                    </div>
                  )}
                </main>
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}

export default App;
