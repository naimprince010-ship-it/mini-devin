import { useState, useEffect } from 'react';
import { Provider } from '../types';
import { useApi } from '../hooks/useApi';
import { Server, ChevronDown, Loader2 } from 'lucide-react';

interface ProviderSelectorProps {
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;
}

export function ProviderSelector({
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
}: ProviderSelectorProps) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const api = useApi();

  useEffect(() => {
    let cancelled = false;
    const loadProviders = async () => {
      setLoadError(null);
      try {
        const data = await api.listProviders();
        if (!cancelled) {
          setProviders(data);
          if (data.length > 0 && !selectedProvider) {
            onProviderChange(data[0].id);
            if (data[0].models.length > 0) {
              onModelChange(data[0].models[0]);
            }
          }
        }
      } catch (e) {
        console.error('Failed to load providers:', e);
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : 'Failed to load providers');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadProviders();
    return () => {
      cancelled = true;
    };
  }, []);

  const currentProvider = providers.find(p => p.id === selectedProvider);
  const availableModels = currentProvider?.models || [];

  const handleProviderSelect = (providerId: string) => {
    onProviderChange(providerId);
    const provider = providers.find(p => p.id === providerId);
    if (provider && provider.models.length > 0) {
      onModelChange(provider.models[0]);
    }
    setShowDropdown(false);
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-400">
        <Loader2 className="animate-spin" size={16} />
        <span className="text-sm">Loading providers...</span>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="space-y-2 text-sm">
        <p className="text-red-400">{loadError}</p>
        <p className="text-[#737373] text-xs leading-relaxed">
          Dev: run the API on port 8000 (<code className="text-[#a3a3a3]">uvicorn mini_devin.api.app:app</code>) so
          Vite can proxy <code className="text-[#a3a3a3]">/api/providers</code>. Set{' '}
          <code className="text-[#a3a3a3]">OPENAI_API_KEY</code>, <code className="text-[#a3a3a3]">ANTHROPIC_API_KEY</code>, or{' '}
          <code className="text-[#a3a3a3]">GEMINI_API_KEY</code> for models to appear.
        </p>
      </div>
    );
  }

  if (providers.length === 0) {
    return (
      <div className="text-sm text-yellow-500">
        No LLM providers configured. Add API keys to enable AI features.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-sm text-gray-300 mb-1">Provider</label>
        <div className="relative">
          <button
            type="button"
            onClick={() => setShowDropdown(!showDropdown)}
            className="w-full flex items-center justify-between px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
          >
            <div className="flex items-center gap-2">
              <Server size={16} className="text-gray-400" />
              <span>{currentProvider?.name || 'Select Provider'}</span>
            </div>
            <ChevronDown size={16} className={`text-gray-400 transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
          </button>
          
          {showDropdown && (
            <div className="absolute z-10 w-full mt-1 bg-gray-700 border border-gray-600 rounded-lg shadow-lg overflow-hidden">
              {providers.map((provider) => (
                <button
                  key={provider.id}
                  type="button"
                  onClick={() => handleProviderSelect(provider.id)}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-600 transition-colors ${
                    provider.id === selectedProvider ? 'bg-blue-600' : ''
                  }`}
                >
                  <Server size={16} className="text-gray-400" />
                  <div>
                    <div className="text-white">{provider.name}</div>
                    <div className="text-xs text-gray-400">{provider.models.length} models</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Model</label>
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
