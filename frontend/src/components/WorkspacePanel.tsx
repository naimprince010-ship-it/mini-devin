import React, { useState, useRef, useEffect } from 'react';
import { Terminal, FileCode, History, Maximize2, Globe } from 'lucide-react';
import { MemoryView } from './MemoryView';
import { FileExplorer } from './FileExplorer';
import { ToolCallLog } from './ToolCallLog';
import { useSessionEvents } from '../contexts/SessionEventsContext';

interface WorkspacePanelProps {
    sessionId?: string;
}

type TabType = 'shell' | 'worklog' | 'editor' | 'browser';

export const WorkspacePanel: React.FC<WorkspacePanelProps> = ({ sessionId }) => {
    const [activeTab, setActiveTab] = useState<TabType>('shell');
    const shellRef = useRef<HTMLDivElement>(null);
    const events = useSessionEvents();

    // Auto-scroll shell when new lines come in
    useEffect(() => {
        if (shellRef.current) {
            shellRef.current.scrollTop = shellRef.current.scrollHeight;
        }
    }, [events.shellLines]);

    // Switch to worklog when a tool starts
    useEffect(() => {
        if (events.toolCalls.length > 0 && events.isRunning) {
            // Don't auto-switch away from editor/browser
            if (activeTab === 'shell') setActiveTab('worklog');
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [events.toolCalls.length, events.isRunning]);

    const tabs: { id: TabType; label: string; icon: React.ReactNode; badge?: number }[] = [
        { id: 'shell', label: 'Shell', icon: <Terminal size={13} /> },
        {
            id: 'worklog',
            label: 'Worklog',
            icon: <History size={13} />,
            badge: events.toolCalls.length > 0 ? events.toolCalls.length : undefined,
        },
        { id: 'editor', label: 'IDE', icon: <FileCode size={13} /> },
        { id: 'browser', label: 'Browser', icon: <Globe size={13} /> },
    ];

    return (
        <div className="flex-1 flex flex-col bg-[#0a0a0a] border-l border-[#262626]">
            {/* Tab Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#262626] bg-[#111111]">
                <div className="flex gap-1">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${activeTab === tab.id
                                    ? 'bg-[#1e1e1e] text-white shadow-sm'
                                    : 'text-[#737373] hover:text-[#a3a3a3] hover:bg-[#1a1a1a]'
                                }`}
                        >
                            <span className={activeTab === tab.id ? 'text-[#00ff99]' : ''}>{tab.icon}</span>
                            {tab.label}
                            {tab.badge !== undefined && (
                                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-[#00ff99] text-[#0f0f0f] text-[9px] font-bold flex items-center justify-center">
                                    {tab.badge > 9 ? '9+' : tab.badge}
                                </span>
                            )}
                        </button>
                    ))}
                </div>
                <button className="p-1.5 text-[#525252] hover:text-[#a3a3a3] rounded-md hover:bg-[#1a1a1a] transition-colors">
                    <Maximize2 size={13} />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden relative">

                {/* SHELL */}
                {activeTab === 'shell' && (
                    <div
                        ref={shellRef}
                        className="absolute inset-0 p-4 font-mono text-xs overflow-y-auto bg-[#050505] custom-scrollbar"
                    >
                        <div className="text-[#00ff99] mb-3 text-[11px]">
                            ubuntu@mini-devin:~$ <span className="animate-pulse">_</span>
                        </div>
                        {events.shellLines.length === 0 ? (
                            <div className="text-[#3a3a3a] italic text-[11px]">
                                Shell output will appear here when the agent runs commands...
                            </div>
                        ) : (
                            events.shellLines.map((line, i) => (
                                <div
                                    key={i}
                                    className={`leading-5 ${line.startsWith('$')
                                            ? 'text-[#00ff99]'
                                            : line.toLowerCase().includes('error') || line.toLowerCase().includes('fail')
                                                ? 'text-red-400'
                                                : 'text-[#c0c0c0]'
                                        }`}
                                >
                                    {line}
                                </div>
                            ))
                        )}
                    </div>
                )}

                {/* WORKLOG */}
                {activeTab === 'worklog' && (
                    <div className="absolute inset-0 overflow-y-auto custom-scrollbar">
                        {events.toolCalls.length === 0 && !events.isRunning ? (
                            <div className="p-4">
                                <h3 className="text-[#525252] text-[10px] font-semibold mb-3 uppercase tracking-wider">
                                    Agent Worklog
                                </h3>
                                <p className="text-[#3a3a3a] text-xs italic">
                                    Tool calls and actions will appear here in real time...
                                </p>
                            </div>
                        ) : (
                            <>
                                <div className="px-3 pt-3 pb-1 flex items-center justify-between">
                                    <h3 className="text-[#525252] text-[10px] font-semibold uppercase tracking-wider">
                                        Agent Worklog
                                    </h3>
                                    <span className="text-[10px] text-[#525252]">
                                        {events.toolCalls.filter(t => t.status === 'completed').length}/
                                        {events.toolCalls.length} done
                                    </span>
                                </div>
                                <ToolCallLog toolCalls={events.toolCalls} />
                            </>
                        )}
                    </div>
                )}

                {/* IDE */}
                {activeTab === 'editor' && (
                    <div className="absolute inset-0 flex flex-col">
                        <FileExplorer
                            sessionId={sessionId}
                            onFileSelect={(path: string) => console.log('Selected file:', path)}
                        />
                    </div>
                )}

                {/* BROWSER */}
                {activeTab === 'browser' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8 gap-4">
                        <Globe size={48} className="text-[#2a2a2a]" />
                        <div>
                            <p className="text-[#525252] text-sm font-medium">Browser</p>
                            <p className="text-[#3a3a3a] text-xs mt-1">
                                Web browser activity will be visible here when the agent browses the internet.
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
