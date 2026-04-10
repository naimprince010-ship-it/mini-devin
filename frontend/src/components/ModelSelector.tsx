import { useState, useEffect, useRef } from 'react';
import { ChevronDown, Cpu, Cloud, Server, Zap, CheckCircle2 } from 'lucide-react';

export interface Model {
  id: string;
  name: string;
  provider: string;
  context_window: number;
  supports_tools: boolean;
  supports_vision: boolean;
  max_output_tokens: number;
  description: string;
}

export interface Provider {
  id: string;
  name: string;
  configured: boolean;
  enabled: boolean;
}

interface ModelSelectorProps {
  value: string;
  onChange: (modelId: string) => void;
  className?: string;
  showDetails?: boolean;
}

const providerMeta: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  openai:    { icon: <Cloud size={12} />,  color: 'text-green-400',  label: 'OpenAI' },
  anthropic: { icon: <Cpu size={12} />,   color: 'text-orange-400', label: 'Anthropic' },
  google:    { icon: <Zap size={12} />,   color: 'text-blue-400',   label: 'Google' },
  ollama:    { icon: <Server size={12} />, color: 'text-purple-400', label: 'Ollama' },
};

const formatCtx = (n: number) => n >= 1_000_000 ? `${(n/1_000_000).toFixed(1)}M` : `${(n/1000).toFixed(0)}K`;

export function ModelSelector({ value, onChange, className = '', showDetails = false }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [filterProvider, setFilterProvider] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    Promise.all([
      fetch(`${apiUrl}/api/models`).then(r => r.json()).catch(() => ({ models: [] })),
      fetch(`${apiUrl}/api/providers`).then(r => r.json()).catch(() => ({ providers: [] })),
    ]).then(([m, p]) => {
      setModels(m.models || []);
      setProviders(p.providers || []);
    }).finally(() => setLoading(false));
  }, []);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const selectedModel = models.find(m => m.id === value);
  const filtered = filterProvider ? models.filter(m => m.provider === filterProvider) : models;
  const grouped = filtered.reduce((acc, m) => {
    if (!acc[m.provider]) acc[m.provider] = [];
    acc[m.provider].push(m);
    return acc;
  }, {} as Record<string, Model[]>);

  if (loading) {
    return (
      <div className={`px-3 py-2 bg-[#1a1a1a] text-[#737373] rounded-lg border border-[#262626] text-sm ${className}`}>
        Loading models...
      </div>
    );
  }

  const meta = selectedModel ? providerMeta[selectedModel.provider] : null;

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2.5 bg-[#1a1a1a] text-white rounded-lg border border-[#262626] hover:border-[#363636] focus:border-[#00ff99]/50 focus:outline-none flex items-center justify-between gap-2 transition-colors"
      >
        <div className="flex items-center gap-2 min-w-0">
          {meta && <span className={meta.color}>{meta.icon}</span>}
          <span className="text-sm truncate">{selectedModel?.name || 'Select a model'}</span>
        </div>
        <ChevronDown size={14} className={`text-[#737373] flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full min-w-[280px] mt-1 bg-[#111111] border border-[#262626] rounded-xl shadow-2xl overflow-hidden">
          {/* Provider filter tabs */}
          {providers.length > 1 && (
            <div className="flex border-b border-[#1a1a1a] px-1 pt-1 gap-0.5">
              <button
                onClick={() => setFilterProvider(null)}
                className={`px-3 py-1.5 text-xs font-medium rounded-t-lg transition-colors ${
                  filterProvider === null ? 'bg-[#1e1e1e] text-white' : 'text-[#737373] hover:text-white'
                }`}
              >
                All
              </button>
              {providers.filter(p => p.enabled).map(p => {
                const m = providerMeta[p.id];
                return (
                  <button
                    key={p.id}
                    onClick={() => setFilterProvider(p.id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-t-lg transition-colors ${
                      filterProvider === p.id ? 'bg-[#1e1e1e] text-white' : 'text-[#737373] hover:text-white'
                    }`}
                  >
                    {m && <span className={m.color}>{m.icon}</span>}
                    {m?.label || p.name}
                  </button>
                );
              })}
            </div>
          )}

          {/* Model list */}
          <div className="overflow-y-auto max-h-72 custom-scrollbar">
            {Object.entries(grouped).map(([provider, providerModels]) => (
              <div key={provider}>
                {!filterProvider && (
                  <div className="px-3 pt-3 pb-1 flex items-center gap-2">
                    {providerMeta[provider] && (
                      <span className={providerMeta[provider].color}>{providerMeta[provider].icon}</span>
                    )}
                    <span className="text-[10px] font-bold uppercase tracking-widest text-[#525252]">
                      {providerMeta[provider]?.label || provider}
                    </span>
                  </div>
                )}
                {providerModels.map(model => (
                  <button
                    key={model.id}
                    onClick={() => { onChange(model.id); setIsOpen(false); }}
                    className={`w-full px-3 py-2.5 text-left hover:bg-[#1a1a1a] flex items-start justify-between gap-3 transition-colors ${
                      model.id === value ? 'bg-[#00ff99]/5 border-l-2 border-[#00ff99]' : ''
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-white font-medium">{model.name}</span>
                        {model.id === value && <CheckCircle2 size={12} className="text-[#00ff99] flex-shrink-0" />}
                      </div>
                      {model.description && (
                        <p className="text-[11px] text-[#525252] mt-0.5">{model.description}</p>
                      )}
                      {showDetails && (
                        <div className="flex items-center gap-2 mt-1 text-[10px] text-[#3a3a3a]">
                          <span>{formatCtx(model.context_window)} ctx</span>
                          <span>·</span>
                          <span>{formatCtx(model.max_output_tokens)} out</span>
                        </div>
                      )}
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      {model.supports_tools && (
                        <span className="text-[9px] px-1.5 py-0.5 bg-green-500/15 text-green-400 rounded font-bold">TOOLS</span>
                      )}
                      {model.supports_vision && (
                        <span className="text-[9px] px-1.5 py-0.5 bg-purple-500/15 text-purple-400 rounded font-bold">VISION</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
            {Object.keys(grouped).length === 0 && (
              <p className="p-4 text-[#525252] text-sm text-center">No models available</p>
            )}
          </div>
        </div>
      )}

      {selectedModel && showDetails && (
        <div className="mt-2 px-3 py-2 rounded-lg bg-[#1a1a1a] border border-[#262626]">
          <div className="flex items-center gap-2 text-xs text-[#737373]">
            {meta && <span className={meta.color}>{meta.icon}</span>}
            <span>{meta?.label || selectedModel.provider}</span>
            <span className="text-[#2a2a2a]">·</span>
            <span>{formatCtx(selectedModel.context_window)} context</span>
            {selectedModel.supports_tools && <span className="text-green-400">+ Tools</span>}
          </div>
        </div>
      )}
    </div>
  );
}
