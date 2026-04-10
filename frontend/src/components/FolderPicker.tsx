import { useState, useEffect } from 'react';
import { FolderOpen, Folder, ChevronRight, X, Check, ArrowLeft, Loader2 } from 'lucide-react';

interface DirEntry {
    name: string;
    path: string;
    is_directory: boolean;
}

interface BrowseResult {
    current: string;
    parent: string | null;
    entries: DirEntry[];
}

interface FolderPickerProps {
    value: string;
    onChange: (path: string) => void;
}

export function FolderPicker({ value, onChange }: FolderPickerProps) {
    const [open, setOpen] = useState(false);
    const [data, setData] = useState<BrowseResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const browse = async (path: string) => {
        setLoading(true);
        setError('');
        try {
            const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
            if (!res.ok) throw new Error('Cannot open this folder');
            const json: BrowseResult = await res.json();
            setData(json);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : 'Error loading directory');
        } finally {
            setLoading(false);
        }
    };

    const handleOpen = () => {
        setOpen(true);
        browse(value && value !== '.' ? value : '.');
    };

    const handleSelect = () => {
        if (data) {
            onChange(data.current);
            setOpen(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape') setOpen(false);
    };

    return (
        <div className="relative">
            {/* Input row with Browse button */}
            <div className="flex gap-2">
                <input
                    type="text"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    className="flex-1 px-3 py-2.5 bg-[#0a0a0a] text-white text-sm rounded-lg border border-[#262626] focus:border-[#00ff99]/40 focus:outline-none transition-colors placeholder-[#525252] font-mono"
                    placeholder="Leave empty for default workspace"
                />
                <button
                    type="button"
                    onClick={handleOpen}
                    className="px-3 py-2.5 bg-[#1a1a1a] border border-[#262626] rounded-lg text-[#a3a3a3] hover:text-white hover:border-[#00ff99]/30 transition-colors flex items-center gap-1.5 text-xs font-medium flex-shrink-0"
                    title="Browse folders"
                >
                    <FolderOpen size={13} />
                    Browse
                </button>
            </div>

            {/* Folder picker dropdown */}
            {open && (
                <div
                    className="absolute z-50 top-full left-0 right-0 mt-1 bg-[#111111] border border-[#262626] rounded-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-1 duration-150"
                    onKeyDown={handleKeyDown}
                >
                    {/* Toolbar */}
                    <div className="flex items-center justify-between px-3 py-2 border-b border-[#1a1a1a] bg-[#0a0a0a]">
                        <button
                            onClick={() => data?.parent && browse(data.parent)}
                            disabled={!data?.parent || loading}
                            className="flex items-center gap-1 text-[10px] text-[#737373] hover:text-white disabled:opacity-30 transition-colors"
                        >
                            <ArrowLeft size={11} /> Up
                        </button>
                        <span className="text-[10px] font-mono text-[#525252] truncate max-w-[220px] px-2">
                            {data?.current ?? '…'}
                        </span>
                        <button onClick={() => setOpen(false)} className="text-[#525252] hover:text-white transition-colors">
                            <X size={13} />
                        </button>
                    </div>

                    {/* Directory listing */}
                    <div className="max-h-52 overflow-y-auto custom-scrollbar">
                        {loading && (
                            <div className="flex items-center justify-center py-6 text-[#525252]">
                                <Loader2 size={16} className="animate-spin" />
                            </div>
                        )}
                        {error && (
                            <div className="px-3 py-3 text-xs text-red-400">{error}</div>
                        )}
                        {!loading && !error && data && (
                            <>
                                {data.entries.length === 0 && (
                                    <div className="px-3 py-4 text-xs text-[#525252] text-center">Empty folder</div>
                                )}
                                {data.entries.map((entry) => (
                                    <button
                                        key={entry.path}
                                        onClick={() => entry.is_directory ? browse(entry.path) : null}
                                        className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors text-left ${entry.is_directory
                                                ? 'text-[#d1d1d1] hover:bg-[#1a1a1a] cursor-pointer'
                                                : 'text-[#525252] cursor-default'
                                            }`}
                                    >
                                        {entry.is_directory ? (
                                            <Folder size={13} className="text-yellow-500/70 flex-shrink-0" />
                                        ) : (
                                            <span className="w-3 h-3 flex-shrink-0" />
                                        )}
                                        <span className="truncate">{entry.name}</span>
                                        {entry.is_directory && (
                                            <ChevronRight size={11} className="ml-auto text-[#525252] flex-shrink-0" />
                                        )}
                                    </button>
                                ))}
                            </>
                        )}
                    </div>

                    {/* Footer — Select current dir */}
                    <div className="px-3 py-2 border-t border-[#1a1a1a] bg-[#0a0a0a] flex items-center justify-between gap-2">
                        <span className="text-[10px] text-[#525252] truncate flex-1">
                            Select: <span className="text-[#a3a3a3] font-mono">{data?.current}</span>
                        </span>
                        <button
                            onClick={handleSelect}
                            disabled={!data}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#00ff99] text-[#0f0f0f] text-xs font-bold rounded-lg hover:bg-[#00e589] disabled:opacity-40 transition-colors flex-shrink-0"
                        >
                            <Check size={12} /> Select
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
