import { useState, useEffect } from 'react';
import { SystemStatus } from '../types';
import { useApi } from '../hooks/useApi';
import { Activity, Clock, CheckCircle, Server } from 'lucide-react';

export function StatusBar() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const api = useApi();

  const loadStatus = async () => {
    try {
      const data = await api.getStatus();
      setStatus(data);
    } catch (e) {
      console.error('Failed to load status:', e);
    }
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (!status) {
    return (
      <div className="bg-gray-800 border-t border-gray-700 px-4 py-2">
        <span className="text-gray-500 text-sm">Loading...</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border-t border-gray-700 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <Server size={14} className="text-gray-400" />
          <span className="text-gray-300">v{status.version}</span>
        </div>
        <div className="flex items-center gap-2">
          <Activity size={14} className={status.status === 'running' ? 'text-green-400' : 'text-red-400'} />
          <span className={status.status === 'running' ? 'text-green-400' : 'text-red-400'}>
            {status.status}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Clock size={14} className="text-gray-400" />
          <span className="text-gray-300">Uptime: {formatUptime(status.uptime_seconds)}</span>
        </div>
      </div>
      <div className="flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-gray-400">Active Sessions:</span>
          <span className="text-white font-medium">{status.active_sessions}</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle size={14} className="text-green-400" />
          <span className="text-gray-400">Tasks Completed:</span>
          <span className="text-white font-medium">{status.total_tasks_completed}</span>
        </div>
      </div>
    </div>
  );
}
