import { useState, useEffect, useCallback } from 'react';
import { Memory } from '../types';
import { useApi } from '../hooks/useApi';
import { Brain, Trash2, Plus, RefreshCw, Clock, Hash, X } from 'lucide-react';

interface MemoryViewProps {
  sessionId: string;
}

export function MemoryView({ sessionId }: MemoryViewProps) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [loading, setLoading] = useState(false);
  const api = useApi();

  const loadMemories = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listMemories(sessionId);
      setMemories(data);
    } catch (e) {
      console.error('Failed to load memories:', e);
    } finally {
      setLoading(false);
    }
  }, [sessionId, api]);

  useEffect(() => {
    loadMemories();
  }, [loadMemories]);

  const handleAddMemory = async () => {
    if (!newKey.trim() || !newValue.trim()) return;
    try {
      await api.storeMemory(sessionId, newKey, newValue);
      setNewKey('');
      setNewValue('');
      setShowAddForm(false);
      loadMemories();
    } catch (e) {
      console.error('Failed to store memory:', e);
    }
  };

  const handleDeleteMemory = async (memoryId: string) => {
    try {
      await api.deleteMemory(sessionId, memoryId);
      setMemories(prev => prev.filter(m => m.memory_id !== memoryId));
    } catch (e) {
      console.error('Failed to delete memory:', e);
    }
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#525252] flex items-center gap-2">
          <Brain size={13} className="text-[#00ff99]" />
          Agent Memory
        </h3>
        <div className="flex gap-1">
          <button
            onClick={loadMemories}
            className="p-1.5 text-[#525252] hover:text-white hover:bg-[#1a1a1a] rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="p-1.5 text-[#525252] hover:text-[#00ff99] hover:bg-[#00ff99]/10 rounded-lg transition-colors"
            title="Add Memory"
          >
            <Plus size={12} />
          </button>
        </div>
      </div>

      {/* Add form */}
      {showAddForm && (
        <div className="p-3 bg-[#111111] border border-[#262626] rounded-xl space-y-2">
          <div className="flex items-center justify-between mb-1">
            <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold">New Memory</p>
            <button onClick={() => setShowAddForm(false)} className="text-[#525252] hover:text-white">
              <X size={12} />
            </button>
          </div>
          <input
            type="text"
            value={newKey}
            onChange={e => setNewKey(e.target.value)}
            placeholder="Key (e.g. user_preference)"
            className="w-full px-3 py-2 bg-[#0f0f0f] border border-[#262626] focus:border-[#00ff99]/50 text-white rounded-lg text-xs outline-none transition-colors"
          />
          <textarea
            value={newValue}
            onChange={e => setNewValue(e.target.value)}
            placeholder="Value (e.g. prefers TypeScript)"
            className="w-full px-3 py-2 bg-[#0f0f0f] border border-[#262626] focus:border-[#00ff99]/50 text-white rounded-lg text-xs outline-none resize-none transition-colors"
            rows={2}
          />
          <button
            onClick={handleAddMemory}
            disabled={!newKey.trim() || !newValue.trim()}
            className="w-full py-1.5 bg-[#00ff99] text-[#0f0f0f] rounded-lg text-xs font-bold disabled:opacity-40 hover:bg-[#00e589] transition-colors"
          >
            Store Memory
          </button>
        </div>
      )}

      {/* Memory list */}
      {memories.length === 0 ? (
        <div className="py-8 text-center">
          <Brain size={24} className="text-[#2a2a2a] mx-auto mb-2" />
          <p className="text-[#3a3a3a] text-xs">
            No memories yet. The agent stores important context here during tasks.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {memories.map(memory => (
            <div key={memory.memory_id} className="p-3 bg-[#111111] border border-[#1a1a1a] rounded-xl hover:border-[#262626] transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[#00ff99] font-mono text-xs truncate">{memory.content.key}</span>
                    <span className="text-[9px] text-[#525252] px-1.5 py-0.5 bg-[#1a1a1a] rounded font-bold uppercase">
                      {memory.type}
                    </span>
                  </div>
                  <p className="text-[#a3a3a3] text-xs break-words leading-relaxed">{memory.content.value}</p>
                  <div className="flex items-center gap-3 mt-2 text-[10px] text-[#3a3a3a]">
                    <span className="flex items-center gap-1">
                      <Clock size={9} />
                      {new Date(memory.created_at).toLocaleString()}
                    </span>
                    <span className="flex items-center gap-1">
                      <Hash size={9} />
                      {memory.access_count} uses
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteMemory(memory.memory_id)}
                  className="p-1.5 text-[#3a3a3a] hover:text-red-400 rounded-lg flex-shrink-0 hover:bg-red-500/10 transition-colors"
                  title="Delete"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
