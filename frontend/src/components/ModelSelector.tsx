import { useState, useEffect } from 'react';
import { ChevronDown, Cpu, Cloud, Server, Building2 } from 'lucide-react';

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

const providerIcons: Record<string, React.ReactNode> = {
  openai: <Cloud size={14} className="text-green-400" />,
  anthropic: <Cpu size={14} className="text-orange-400" />,
  ollama: <Server size={14} className="text-blue-400" />,
  azure: <Building2 size={14} className="text-cyan-400" />,
};

const providerColors: Record<string, string> = {
  openai: 'border-green-500/30 bg-green-500/10',
  anthropic: 'border-orange-500/30 bg-orange-500/10',
  ollama: 'border-blue-500/30 bg-blue-500/10',
  azure: 'border-cyan-500/30 bg-cyan-500/10',
};

export function ModelSelector({ value, onChange, className = '', showDetails = false }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        
        const [modelsRes, providersRes] = await Promise.all([
          fetch(`${apiUrl}/api/models`),
          fetch(`${apiUrl}/api/providers`),
        ]);
        
        if (modelsRes.ok) {
          const data = await modelsRes.json();
          setModels(data.models || []);
        }
        
        if (providersRes.ok) {
          const data = await providersRes.json();
          setProviders(data.providers || []);
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
        setModels([
          { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', context_window: 128000, supports_tools: true, supports_vision: true, max_output_tokens: 4096, description: 'Most capable GPT-4 model' },
          { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openai', context_window: 128000, supports_tools: true, supports_vision: true, max_output_tokens: 16384, description: 'Smaller, faster GPT-4o' },
          { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', provider: 'anthropic', context_window: 200000, supports_tools: true, supports_vision: true, max_output_tokens: 8192, description: 'Most intelligent Claude' },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  const selectedModel = models.find(m => m.id === value);
  
  const filteredModels = selectedProvider 
    ? models.filter(m => m.provider === selectedProvider)
    : models;

  const groupedModels = filteredModels.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, Model[]>);

  const formatContextWindow = (tokens: number) => {
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`;
    return tokens.toString();
  };

  if (loading) {
    return (
      <div className={`px-3 py-2 bg-gray-600 text-gray-400 rounded border border-gray-500 ${className}`}>
        Loading models...
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none flex items-center justify-between"
      >
        <div className="flex items-center gap-2">
          {selectedModel && providerIcons[selectedModel.provider]}
          <span>{selectedModel?.name || 'Select a model'}</span>
        </div>
        <ChevronDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-gray-700 border border-gray-600 rounded-lg shadow-xl max-h-80 overflow-hidden">
          <div className="flex border-b border-gray-600">
            <button
              onClick={() => setSelectedProvider(null)}
              className={`flex-1 px-3 py-2 text-xs font-medium ${
                selectedProvider === null ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              All
            </button>
            {providers.filter(p => p.enabled).map(provider => (
              <button
                key={provider.id}
                onClick={() => setSelectedProvider(provider.id)}
                className={`flex-1 px-3 py-2 text-xs font-medium flex items-center justify-center gap-1 ${
                  selectedProvider === provider.id ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                {providerIcons[provider.id]}
                <span className="capitalize">{provider.id}</span>
              </button>
            ))}
          </div>
          
          <div className="overflow-y-auto max-h-64">
            {Object.entries(groupedModels).map(([provider, providerModels]) => (
              <div key={provider}>
                {!selectedProvider && (
                  <div className="px-3 py-1.5 text-xs font-semibold text-gray-400 bg-gray-750 uppercase tracking-wider flex items-center gap-2">
                    {providerIcons[provider]}
                    {provider}
                  </div>
                )}
                {providerModels.map(model => (
                  <button
                    key={model.id}
                    onClick={() => {
                      onChange(model.id);
                      setIsOpen(false);
                    }}
                    className={`w-full px-3 py-2 text-left hover:bg-gray-600 flex flex-col ${
                      model.id === value ? 'bg-blue-600/20 border-l-2 border-blue-500' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-white font-medium">{model.name}</span>
                      <div className="flex items-center gap-2">
                        {model.supports_tools && (
                          <span className="text-xs px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded">Tools</span>
                        )}
                        {model.supports_vision && (
                          <span className="text-xs px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded">Vision</span>
                        )}
                      </div>
                    </div>
                    {showDetails && (
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                        <span>{formatContextWindow(model.context_window)} context</span>
                        <span>{formatContextWindow(model.max_output_tokens)} max output</span>
                      </div>
                    )}
                    {model.description && (
                      <span className="text-xs text-gray-500 mt-0.5">{model.description}</span>
                    )}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedModel && showDetails && (
        <div className={`mt-2 p-2 rounded border ${providerColors[selectedModel.provider]}`}>
          <div className="flex items-center gap-2 text-xs text-gray-300">
            {providerIcons[selectedModel.provider]}
            <span className="capitalize">{selectedModel.provider}</span>
            <span className="text-gray-500">|</span>
            <span>{formatContextWindow(selectedModel.context_window)} context</span>
            {selectedModel.supports_tools && <span className="text-green-400">+ Tools</span>}
            {selectedModel.supports_vision && <span className="text-purple-400">+ Vision</span>}
          </div>
        </div>
      )}
    </div>
  );
}
