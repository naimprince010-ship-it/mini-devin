import { useState } from 'react';
import { Session } from './types';
import { SessionList } from './components/SessionList';
import { TaskPanel } from './components/TaskPanel';
import { StatusBar } from './components/StatusBar';
import { Bot, Github } from 'lucide-react';

function App() {
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);

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
