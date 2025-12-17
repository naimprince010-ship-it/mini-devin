import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { FileJson, FileText, File, Loader2 } from 'lucide-react';

interface ExportButtonsProps {
  sessionId: string;
}

export function ExportButtons({ sessionId }: ExportButtonsProps) {
  const [exporting, setExporting] = useState<string | null>(null);
  const api = useApi();

  const handleExport = async (format: 'json' | 'markdown' | 'txt') => {
    setExporting(format);
    try {
      const data = await api.exportSession(sessionId, format);
      
      let content: string;
      let filename: string;
      let mimeType: string;
      
      if (format === 'json') {
        content = JSON.stringify(data, null, 2);
        filename = `session-${sessionId}.json`;
        mimeType = 'application/json';
      } else if (format === 'markdown') {
        content = data.content || '';
        filename = `session-${sessionId}.md`;
        mimeType = 'text/markdown';
      } else {
        content = data.content || '';
        filename = `session-${sessionId}.txt`;
        mimeType = 'text/plain';
      }
      
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Failed to export session:', e);
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400">Export:</span>
      <button
        onClick={() => handleExport('json')}
        disabled={exporting !== null}
        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors disabled:opacity-50"
        title="Export as JSON"
      >
        {exporting === 'json' ? (
          <Loader2 className="animate-spin" size={12} />
        ) : (
          <FileJson size={12} />
        )}
        JSON
      </button>
      <button
        onClick={() => handleExport('markdown')}
        disabled={exporting !== null}
        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors disabled:opacity-50"
        title="Export as Markdown"
      >
        {exporting === 'markdown' ? (
          <Loader2 className="animate-spin" size={12} />
        ) : (
          <FileText size={12} />
        )}
        MD
      </button>
      <button
        onClick={() => handleExport('txt')}
        disabled={exporting !== null}
        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors disabled:opacity-50"
        title="Export as Text"
      >
        {exporting === 'txt' ? (
          <Loader2 className="animate-spin" size={12} />
        ) : (
          <File size={12} />
        )}
        TXT
      </button>
    </div>
  );
}
