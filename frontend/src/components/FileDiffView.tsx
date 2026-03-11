import React, { useState } from 'react';
import { FileCode, ChevronDown, ChevronRight } from 'lucide-react';
import type { FileEdit } from '../contexts/SessionEventsContext';

interface FileDiffViewProps {
    fileEdits: FileEdit[];
    onFileSelect?: (path: string) => void;
}

function getLanguage(path: string): string {
    const ext = path.split('.').pop()?.toLowerCase() || '';
    const map: Record<string, string> = {
        py: 'python', js: 'javascript', ts: 'typescript', tsx: 'tsx', jsx: 'jsx',
        json: 'json', md: 'markdown', html: 'html', css: 'css', sh: 'bash',
        yml: 'yaml', yaml: 'yaml', toml: 'toml', rs: 'rust', go: 'go',
        java: 'java', cpp: 'cpp', c: 'c', rb: 'ruby', php: 'php',
    };
    return map[ext] || 'text';
}

function computeDiff(before: string, after: string): { type: 'add' | 'remove' | 'same'; line: string }[] {
    const beforeLines = before.split('\n');
    const afterLines = after.split('\n');

    // Simple LCS-based diff
    const result: { type: 'add' | 'remove' | 'same'; line: string }[] = [];
    const m = beforeLines.length;
    const n = afterLines.length;

    // Build LCS table (limit size for performance)
    const MAX = 200;
    const bSlice = beforeLines.slice(0, MAX);
    const aSlice = afterLines.slice(0, MAX);
    const bm = bSlice.length;
    const am = aSlice.length;

    const dp: number[][] = Array.from({ length: bm + 1 }, () => new Array(am + 1).fill(0));
    for (let i = 1; i <= bm; i++) {
        for (let j = 1; j <= am; j++) {
            if (bSlice[i - 1] === aSlice[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    // Backtrack
    let i = bm, j = am;
    const ops: { type: 'add' | 'remove' | 'same'; line: string }[] = [];
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && bSlice[i - 1] === aSlice[j - 1]) {
            ops.unshift({ type: 'same', line: bSlice[i - 1] });
            i--; j--;
        } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
            ops.unshift({ type: 'add', line: aSlice[j - 1] });
            j--;
        } else {
            ops.unshift({ type: 'remove', line: bSlice[i - 1] });
            i--;
        }
    }

    // Add remaining lines if truncated
    if (m > MAX) {
        afterLines.slice(MAX).forEach(l => ops.push({ type: 'add', line: l }));
    }

    return ops;
}

function FileItem({ edit, isSelected, onClick }: { edit: FileEdit; isSelected: boolean; onClick: () => void }) {
    const filename = edit.path.split(/[\\/]/).pop() || edit.path;
    const isNew = !edit.before;

    return (
        <button
            onClick={onClick}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors text-left ${isSelected ? 'bg-[#1a1a1a] text-white' : 'text-[#737373] hover:text-[#a3a3a3] hover:bg-[#141414]'
                }`}
        >
            <FileCode size={12} className={isSelected ? 'text-[#00ff99]' : 'text-[#525252]'} />
            <span className="flex-1 truncate font-mono">{filename}</span>
            {isNew && (
                <span className="text-[9px] text-[#00ff99] bg-[#00ff99]/10 px-1.5 py-0.5 rounded font-bold uppercase">NEW</span>
            )}
        </button>
    );
}

export function FileDiffView({ fileEdits, onFileSelect }: FileDiffViewProps) {
    const [selectedPath, setSelectedPath] = useState<string | null>(null);
    const [showFullFile, setShowFullFile] = useState(false);

    if (fileEdits.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-3 p-8 text-center">
                <FileCode size={40} className="text-[#2a2a2a]" />
                <div>
                    <p className="text-[#525252] text-sm font-medium">No file changes yet</p>
                    <p className="text-[#3a3a3a] text-xs mt-1">Files created or edited by the agent will appear here</p>
                </div>
            </div>
        );
    }

    const selectedEdit = fileEdits.find(f => f.path === selectedPath) || fileEdits[fileEdits.length - 1];
    const diffLines = selectedEdit?.before
        ? computeDiff(selectedEdit.before, selectedEdit.content)
        : selectedEdit?.content.split('\n').map(l => ({ type: 'add' as const, line: l })) || [];

    const lang = getLanguage(selectedEdit?.path || '');

    // For new files or full view, show all lines
    const displayLines = showFullFile && !selectedEdit?.before
        ? selectedEdit.content.split('\n').map(l => ({ type: 'same' as const, line: l }))
        : diffLines;

    const addCount = diffLines.filter(l => l.type === 'add').length;
    const removeCount = diffLines.filter(l => l.type === 'remove').length;

    return (
        <div className="flex h-full">
            {/* File list sidebar */}
            <div className="w-44 flex-shrink-0 border-r border-[#1a1a1a] p-2 overflow-y-auto custom-scrollbar">
                <p className="text-[9px] uppercase tracking-wider text-[#3a3a3a] font-bold px-2 mb-2">
                    Changed Files
                </p>
                {fileEdits.map(edit => (
                    <FileItem
                        key={edit.path}
                        edit={edit}
                        isSelected={(selectedPath || fileEdits[fileEdits.length - 1]?.path) === edit.path}
                        onClick={() => { setSelectedPath(edit.path); setShowFullFile(false); }}
                    />
                ))}
            </div>

            {/* Diff viewer */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {selectedEdit && (
                    <>
                        {/* File header */}
                        <div className="flex items-center justify-between px-4 py-2 border-b border-[#1a1a1a] bg-[#0d0d0d]">
                            <div className="flex items-center gap-2">
                                <span className="text-[10px] font-mono text-[#a3a3a3] truncate max-w-[200px]">
                                    {selectedEdit.path}
                                </span>
                                <span className="text-[9px] text-[#525252]">{lang}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                {addCount > 0 && (
                                    <span className="text-[10px] text-green-400 font-mono">+{addCount}</span>
                                )}
                                {removeCount > 0 && (
                                    <span className="text-[10px] text-red-400 font-mono">-{removeCount}</span>
                                )}
                                {onFileSelect && selectedEdit && (
                                    <button
                                        onClick={() => onFileSelect(selectedEdit.path)}
                                        className="text-[9px] text-[#00ff99] hover:text-white border border-[#00ff99]/20 px-1.5 py-0.5 rounded uppercase tracking-wider"
                                    >
                                        Edit
                                    </button>
                                )}
                                {selectedEdit.before && (
                                    <button
                                        onClick={() => setShowFullFile(v => !v)}
                                        className="text-[9px] text-[#525252] hover:text-[#a3a3a3] uppercase tracking-wider"
                                    >
                                        {showFullFile ? 'Diff' : 'Full'}
                                    </button>
                                )}
                            </div>
                        </div>

                        {/* Lines */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar font-mono text-[11px]">
                            {displayLines.map((line, i) => (
                                <div
                                    key={i}
                                    className={`flex items-start gap-0 leading-5 ${line.type === 'add'
                                        ? 'bg-green-900/20'
                                        : line.type === 'remove'
                                            ? 'bg-red-900/20'
                                            : ''
                                        }`}
                                >
                                    {/* Line number */}
                                    <span className="w-10 text-right pr-3 text-[#2a2a2a] select-none flex-shrink-0 py-0.5">
                                        {i + 1}
                                    </span>
                                    {/* Prefix */}
                                    <span className={`w-4 flex-shrink-0 py-0.5 font-bold ${line.type === 'add' ? 'text-green-500'
                                        : line.type === 'remove' ? 'text-red-500'
                                            : 'text-transparent'
                                        }`}>
                                        {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
                                    </span>
                                    {/* Content */}
                                    <span className={`flex-1 py-0.5 pr-4 whitespace-pre-wrap break-all ${line.type === 'add' ? 'text-green-300'
                                        : line.type === 'remove' ? 'text-red-400 line-through opacity-70'
                                            : 'text-[#c0c0c0]'
                                        }`}>
                                        {line.line || ' '}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
