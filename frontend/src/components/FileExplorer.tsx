import React, { useState, useEffect, useCallback } from 'react';
import { Folder, File, ChevronRight, ChevronDown, RefreshCw, FolderOpen } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useSessionEvents } from '../contexts/SessionEventsContext';

interface FileExplorerProps {
    sessionId?: string;
    onFileSelect?: (path: string) => void;
}

interface FileEntry {
    name: string;
    path: string;
    is_directory: boolean;
    size: number;
    modified_at: string;
}

interface FileNode {
    name: string;
    path: string;
    isDir: boolean;
    size?: number;
    children?: FileNode[];
    isOpen?: boolean;
}

function buildTree(entries: FileEntry[], prefix: string = '.'): FileNode[] {
    return entries.map(e => ({
        name: e.name,
        path: e.path,
        isDir: e.is_directory,
        size: e.size,
        isOpen: false,
    }));
}

export function FileExplorer({ sessionId, onFileSelect }: FileExplorerProps) {
    const [files, setFiles] = useState<FileNode[]>([]);
    const [loading, setLoading] = useState(false);
    const [openDirs, setOpenDirs] = useState<Record<string, FileNode[]>>({});
    const api = useApi();
    const events = useSessionEvents();

    const fetchFiles = useCallback(async (dir: string = '.') => {
        if (!sessionId) return;
        setLoading(true);
        try {
            const data: FileEntry[] = await api.listWorkspaceFiles(sessionId, dir);
            if (dir === '.') {
                setFiles(buildTree(data));
            } else {
                setOpenDirs(prev => ({ ...prev, [dir]: buildTree(data) }));
            }
        } catch (e) {
            console.error('Failed to fetch files:', e);
        } finally {
            setLoading(false);
        }
    }, [sessionId, api]);

    useEffect(() => {
        fetchFiles('.');
    }, [sessionId]);

    // Auto-refresh file list when task ends
    useEffect(() => {
        if (!events.isRunning && sessionId) {
            fetchFiles('.');
        }
    }, [events.isRunning]);

    const toggleFolder = async (node: FileNode) => {
        if (!node.isDir) return;
        const isOpen = !!openDirs[node.path];
        if (!isOpen) {
            await fetchFiles(node.path);
        } else {
            setOpenDirs(prev => {
                const next = { ...prev };
                delete next[node.path];
                return next;
            });
        }
    };

    const getFileIcon = (name: string) => {
        const ext = name.split('.').pop()?.toLowerCase();
        const colors: Record<string, string> = {
            py: 'text-yellow-400', ts: 'text-blue-400', tsx: 'text-blue-300',
            js: 'text-yellow-300', jsx: 'text-yellow-300', json: 'text-green-400',
            md: 'text-[#a3a3a3]', txt: 'text-[#737373]', html: 'text-orange-400',
            css: 'text-purple-400', yml: 'text-red-400', yaml: 'text-red-400',
        };
        return colors[ext || ''] || 'text-[#737373]';
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes}B`;
        return `${(bytes / 1024).toFixed(1)}KB`;
    };

    const renderNode = (node: FileNode, depth: number = 0) => {
        const isOpen = !!openDirs[node.path];
        const children = openDirs[node.path] || [];

        return (
            <div key={node.path} className="select-none">
                <div
                    className="flex items-center gap-1.5 px-2 py-[3px] hover:bg-[#1a1a1a] cursor-pointer rounded transition-colors group"
                    style={{ paddingLeft: `${depth * 12 + 8}px` }}
                    onClick={() => node.isDir ? toggleFolder(node) : onFileSelect?.(node.path)}
                >
                    {node.isDir ? (
                        <div className="flex items-center gap-1 text-[#a3a3a3] group-hover:text-[#00ff99]">
                            {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                            {isOpen
                                ? <FolderOpen size={13} className="text-[#00ff99]" />
                                : <Folder size={13} />
                            }
                        </div>
                    ) : (
                        <div className={`flex items-center gap-1 pl-5 ${getFileIcon(node.name)} group-hover:brightness-125`}>
                            <File size={12} />
                        </div>
                    )}
                    <span className={`flex-1 text-[11px] truncate ${node.isDir ? 'text-[#d1d1d1] font-medium' : 'text-[#a3a3a3]'} group-hover:text-white`}>
                        {node.name}
                    </span>
                    {!node.isDir && node.size !== undefined && node.size > 0 && (
                        <span className="text-[9px] text-[#3a3a3a] group-hover:text-[#525252] flex-shrink-0">
                            {formatSize(node.size)}
                        </span>
                    )}
                </div>

                {node.isDir && isOpen && children.length > 0 && (
                    <div>
                        {children.map(child => renderNode(child, depth + 1))}
                    </div>
                )}
            </div>
        );
    };

    const fileCount = files.filter(f => !f.isDir).length;
    const dirCount = files.filter(f => f.isDir).length;

    return (
        <div className="flex flex-col h-full bg-[#0f0f0f]">
            <div className="flex items-center justify-between px-3 py-2.5 border-b border-[#262626] bg-[#121212]">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-[#737373]">Files</span>
                    {files.length > 0 && (
                        <span className="text-[9px] text-[#525252]">
                            {fileCount}f · {dirCount}d
                        </span>
                    )}
                </div>
                <button
                    onClick={() => fetchFiles('.')}
                    className="p-1 hover:bg-[#1a1a1a] rounded text-[#737373] hover:text-[#00ff99] transition-all"
                    title="Refresh"
                >
                    <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto py-1 custom-scrollbar">
                {loading && files.length === 0 ? (
                    <div className="flex items-center justify-center h-20 opacity-20">
                        <span className="text-[10px] font-bold uppercase tracking-widest animate-pulse">Scanning...</span>
                    </div>
                ) : files.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center p-6 gap-2">
                        <FolderOpen size={28} className="text-[#2a2a2a]" />
                        <p className="text-[#3a3a3a] text-xs">No files yet.</p>
                        <p className="text-[#2a2a2a] text-[10px]">Files created by the agent will appear here.</p>
                    </div>
                ) : (
                    <div className="space-y-0">
                        {files.map(node => renderNode(node))}
                    </div>
                )}
            </div>
        </div>
    );
}
