import { useState, useEffect, useCallback } from 'react';
import { Memory } from '../types';
import { useApi } from '../hooks/useApi';
import { Brain, Trash2, Plus, RefreshCw, Clock, Hash } from 'lucide-react';

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

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Brain size={16} />
          Agent Memory
        </h3>
        <div className="flex gap-1">
          <button
            onClick={loadMemories}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Refresh"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Add Memory"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      {showAddForm && (
        <div className="mb-4 p-3 bg-gray-700 rounded-lg space-y-2">
          <input
            type="text"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="Key (e.g., user_preference)"
            className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none text-sm"
          />
          <textarea
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            placeholder="Value (e.g., prefers dark mode)"
            className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none text-sm resize-none"
            rows={2}
          />
          <div className="flex gap-2">
            <button
              onClick={handleAddMemory}
              disabled={!newKey.trim() || !newValue.trim()}
              className="flex-1 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium disabled:opacity-50"
            >
              Store
            </button>
            <button
              onClick={() => setShowAddForm(false)}
              className="px-3 py-1.5 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {memories.length === 0 ? (
        <p className="text-gray-500 text-sm text-center py-4">
          No memories stored yet. The agent will store important information here during task execution.
        </p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {memories.map((memory) => (
            <div
              key={memory.memory_id}
              className="p-3 bg-gray-700 rounded-lg"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-blue-400 font-mono text-sm">{memory.content.key}</span>
                    <span className="text-xs text-gray-500 px-1.5 py-0.5 bg-gray-600 rounded">
                      {memory.type}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm break-words">{memory.content.value}</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                      <Clock size={10} />
                      {formatDate(memory.created_at)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Hash size={10} />
                      {memory.access_count} accesses
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteMemory(memory.memory_id)}
                  className="p-1.5 text-gray-400 hover:text-red-400 rounded flex-shrink-0"
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
