/**
 * MonacoEditorPanel
 * Live code editor backed by Monaco. Loads file content from the backend
 * and saves changes via PUT /sessions/{id}/file.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import { Save, X, AlertCircle, Loader2 } from 'lucide-react';

interface MonacoEditorPanelProps {
    sessionId: string;
    filePath: string;          // relative to working dir
    initialContent?: string;   // if already known (e.g. from file_changed event)
    onClose?: () => void;
    onSaved?: (path: string, content: string) => void;
}

function getLanguage(path: string): string {
    const ext = path.split('.').pop()?.toLowerCase() ?? '';
    const map: Record<string, string> = {
        ts: 'typescript', tsx: 'typescript',
        js: 'javascript', jsx: 'javascript',
        py: 'python',
        rs: 'rust',
        go: 'go',
        java: 'java',
        cpp: 'cpp', cc: 'cpp', cxx: 'cpp',
        c: 'c', h: 'c',
        css: 'css', scss: 'scss',
        html: 'html',
        json: 'json',
        yaml: 'yaml', yml: 'yaml',
        md: 'markdown',
        sh: 'shell',
        toml: 'toml',
        sql: 'sql',
        rb: 'ruby',
        php: 'php',
    };
    return map[ext] ?? 'plaintext';
}

const API_BASE = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : '/api';

export function MonacoEditorPanel({
    sessionId, filePath, initialContent, onClose, onSaved,
}: MonacoEditorPanelProps) {
    const [content, setContent] = useState<string>(initialContent ?? '');
    const [loading, setLoading] = useState(!initialContent);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dirty, setDirty] = useState(false);
    const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

    // Load file content
    useEffect(() => {
        if (initialContent !== undefined) {
            setContent(initialContent);
            setLoading(false);
            return;
        }
        setLoading(true);
        setError(null);
        fetch(`${API_BASE}/sessions/${sessionId}/file?path=${encodeURIComponent(filePath)}`)
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(d => {
                setContent(d.content ?? '');
                setLoading(false);
                setDirty(false);
            })
            .catch(e => {
                setError(e.message);
                setLoading(false);
            });
    }, [sessionId, filePath, initialContent]);

    const handleSave = useCallback(async () => {
        const current = editorRef.current?.getValue() ?? content;
        setSaving(true);
        setError(null);
        try {
            const r = await fetch(`${API_BASE}/sessions/${sessionId}/file`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: filePath, content: current }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            setDirty(false);
            setContent(current);
            onSaved?.(filePath, current);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Save failed');
        } finally {
            setSaving(false);
        }
    }, [sessionId, filePath, content, onSaved]);

    // Ctrl+S to save
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                handleSave();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [handleSave]);

    const filename = filePath.split('/').pop() ?? filePath;

    return (
        <div className="flex flex-col h-full bg-[#0f0f0f] border border-[#1a1a1a] rounded-lg overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 bg-[#111] border-b border-[#1a1a1a] flex-shrink-0">
                <div className="flex items-center gap-2 min-w-0">
                    <span className="text-[#a3a3a3] text-xs font-mono truncate" title={filePath}>{filePath}</span>
                    {dirty && <span className="w-2 h-2 rounded-full bg-[#f59e0b] flex-shrink-0" title="Unsaved changes" />}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                    {error && (
                        <span className="flex items-center gap-1 text-red-400 text-xs">
                            <AlertCircle size={12} /> {error}
                        </span>
                    )}
                    <button
                        onClick={handleSave}
                        disabled={saving || loading}
                        className="flex items-center gap-1.5 px-2.5 py-1 text-xs bg-[#00ff99]/10 text-[#00ff99] hover:bg-[#00ff99]/20 border border-[#00ff99]/20 rounded-md transition-colors disabled:opacity-40"
                        title="Save (Ctrl+S)"
                    >
                        {saving ? <Loader2 size={11} className="animate-spin" /> : <Save size={11} />}
                        Save
                    </button>
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="p-1 text-[#525252] hover:text-white rounded transition-colors"
                        >
                            <X size={14} />
                        </button>
                    )}
                </div>
            </div>

            {/* Editor */}
            <div className="flex-1 overflow-hidden">
                {loading ? (
                    <div className="h-full flex items-center justify-center text-[#525252]">
                        <Loader2 size={20} className="animate-spin mr-2" />
                        <span className="text-sm">Loading {filename}…</span>
                    </div>
                ) : (
                    <Editor
                        height="100%"
                        language={getLanguage(filePath)}
                        value={content}
                        theme="vs-dark"
                        onMount={(editor) => { editorRef.current = editor; }}
                        onChange={(val) => {
                            setDirty(true);
                            setContent(val ?? '');
                        }}
                        options={{
                            fontSize: 13,
                            fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
                            fontLigatures: true,
                            minimap: { enabled: false },
                            scrollBeyondLastLine: false,
                            lineNumbers: 'on',
                            renderLineHighlight: 'gutter',
                            bracketPairColorization: { enabled: true },
                            padding: { top: 8, bottom: 8 },
                            scrollbar: { verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
                            tabSize: 2,
                            wordWrap: 'off',
                        }}
                    />
                )}
            </div>
        </div>
    );
}
