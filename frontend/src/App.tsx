import { useState } from 'react';
import { Session } from './types';
import { SessionList } from './components/SessionList';
import { TaskPanel } from './components/TaskPanel';
import { StatusBar } from './components/StatusBar';
import { LoginForm } from './components/LoginForm';
import { UserMenu } from './components/UserMenu';
import { useAuth } from './contexts/AuthContext';
import { Bot, Github, LogIn, Loader2 } from 'lucide-react';

function App() {
  const { user, loading } = useAuth();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showLogin, setShowLogin] = useState(false);

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
        {/* Sidebar - Sessions */}
        <aside className="w-80 border-r border-gray-700 overflow-y-auto p-4">
          <SessionList
            onSelectSession={setSelectedSession}
            selectedSessionId={selectedSession?.session_id}
          />
        </aside>

        {/* Main Panel */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {selectedSession ? (
            <TaskPanel session={selectedSession} />
          ) : (
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
          )}
        </main>
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}

export default App;
