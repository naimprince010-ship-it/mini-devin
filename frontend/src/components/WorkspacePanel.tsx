import React, { useState } from 'react';
import { Terminal, FileCode, History, Eye, Maximize2, Plus } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';
import { MemoryView } from './MemoryView';
import { FileExplorer } from './FileExplorer';

interface WorkspacePanelProps {
    sessionId?: string;
}

type TabType = 'shell' | 'worklog' | 'editor';

export const WorkspacePanel: React.FC<WorkspacePanelProps> = ({ sessionId }) => {
    const [activeTab, setActiveTab] = useState<TabType>('shell');

    return (
        <div className="flex-1 flex flex-col bg-[#0f0f0f] border-l border-[#262626]">
            {/* Tab Header */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-[#262626] bg-[#121212]">
                <div className="flex gap-2">
                    <button
                        onClick={() => setActiveTab('shell')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${activeTab === 'shell'
                            ? 'bg-[#1a1a1a] text-[#00ff99]'
                            : 'text-[#a3a3a3] hover:text-white'
                            }`}
                    >
                        <Terminal size={14} />
                        Shell
                    </button>
                    <button
                        onClick={() => setActiveTab('worklog')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${activeTab === 'worklog'
                            ? 'bg-[#1a1a1a] text-[#00ff99]'
                            : 'text-[#a3a3a3] hover:text-white'
                            }`}
                    >
                        <History size={14} />
                        Worklog
                    </button>
                    <button
                        onClick={() => setActiveTab('editor')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${activeTab === 'editor'
                            ? 'bg-[#1a1a1a] text-[#00ff99]'
                            : 'text-[#a3a3a3] hover:text-white'
                            }`}
                    >
                        <FileCode size={14} />
                        IDE
                    </button>
                    <button className="p-1.5 text-[#a3a3a3] hover:text-white rounded-full hover:bg-[#1a1a1a]">
                        <Plus size={14} />
                    </button>
                </div>
                <div className="flex gap-2">
                    <button className="p-1.5 text-[#a3a3a3] hover:text-white rounded-full hover:bg-[#1a1a1a]">
                        <Maximize2 size={14} />
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden relative">
                {activeTab === 'shell' && (
                    <div className="absolute inset-0 p-4 font-mono text-sm overflow-y-auto bg-black">
                        <div className="text-[#00ff99] mb-2">ubuntu@devin-box:~$</div>
                        <StreamingOutput content="" isStreaming={false} sessionId={sessionId || ''} />
                    </div>
                )}

                {activeTab === 'worklog' && (
                    <div className="absolute inset-0 p-4 overflow-y-auto">
                        <h3 className="text-[#a3a3a3] text-xs font-semibold mb-4 uppercase tracking-wider">Session Memory</h3>
                        <MemoryView sessionId={sessionId || ''} />
                    </div>
                )}

                {activeTab === 'editor' && (
                    <div className="absolute inset-0 flex flex-col">
                        <FileExplorer
                            sessionId={sessionId}
                            onFileSelect={(path: string) => console.log('Selected file:', path)}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};
