import React, { useState, useEffect } from 'react';
import { Folder, File, ChevronRight, ChevronDown, RefreshCw } from 'lucide-react';
import { useApi } from '../hooks/useApi';

interface FileExplorerProps {
    sessionId?: string;
    onFileSelect?: (path: string) => void;
}

interface FileNode {
    name: string;
    path: string;
    isDir: boolean;
    children?: FileNode[];
    isOpen?: boolean;
}

export function FileExplorer({ sessionId, onFileSelect }: FileExplorerProps) {
    const [files, setFiles] = useState<FileNode[]>([]);
    const [loading, setLoading] = useState(false);
    const api = useApi();

    const fetchFiles = async (dir: string = '.') => {
        setLoading(true);
        try {
            // Fetch project files from the new /ls endpoint
            if (sessionId) {
                const data = await api.listWorkspaceFiles(sessionId);
                setFiles(data);
            }
        } catch (e) {
            console.error('Failed to fetch files:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFiles();
    }, [sessionId]);

    const toggleFolder = (path: string) => {
        setFiles(prev => {
            const updateNode = (nodes: FileNode[]): FileNode[] => {
                return nodes.map(node => {
                    if (node.path === path) {
                        return { ...node, isOpen: !node.isOpen };
                    }
                    if (node.children) {
                        return { ...node, children: updateNode(node.children) };
                    }
                    return node;
                });
            };
            return updateNode(prev);
        });
    };

    const renderNode = (node: FileNode, depth: number = 0) => {
        const isExpanded = node.isOpen;

        return (
            <div key={node.path} className="select-none">
                <div
                    className="flex items-center gap-2 px-2 py-1 hover:bg-[#1a1a1a] cursor-pointer rounded transition-colors group"
                    style={{ paddingLeft: `${depth * 12 + 8}px` }}
                    onClick={() => node.isDir ? toggleFolder(node.path) : onFileSelect?.(node.path)}
                >
                    {node.isDir ? (
                        <div className="flex items-center gap-1.5 text-[#a3a3a3] group-hover:text-[#00ff99]">
                            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                            <Folder size={14} className={isExpanded ? 'fill-[#00ff99]/20 text-[#00ff99]' : ''} />
                        </div>
                    ) : (
                        <div className="flex items-center gap-1.5 text-[#737373] group-hover:text-white pl-4">
                            <File size={14} />
                        </div>
                    )}
                    <span className={`text-xs ${node.isDir ? 'text-[#d1d1d1]' : 'text-[#a3a3a3]'} group-hover:text-white truncate`}>
                        {node.name}
                    </span>
                </div>

                {node.isDir && isExpanded && node.children && (
                    <div>
                        {node.children.map(child => renderNode(child, depth + 1))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full bg-[#0f0f0f]">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#262626] bg-[#121212]">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[#737373]">Explorer</span>
                <button
                    onClick={() => fetchFiles()}
                    className="p-1 hover:bg-[#1a1a1a] rounded text-[#737373] hover:text-[#00ff99] transition-all"
                    title="Refresh"
                >
                    <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                {loading && files.length === 0 ? (
                    <div className="flex items-center justify-center h-20 opacity-20">
                        <span className="text-[10px] font-bold uppercase tracking-widest animate-pulse">Scanning...</span>
                    </div>
                ) : (
                    <div className="space-y-0.5">
                        {files.map(node => renderNode(node))}
                    </div>
                )}
            </div>
        </div>
    );
}
