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
        filename = `session-${sessionId.slice(0, 8)}.json`;
        mimeType = 'application/json';
      } else if (format === 'markdown') {
        content = data.content || JSON.stringify(data, null, 2);
        filename = `session-${sessionId.slice(0, 8)}.md`;
        mimeType = 'text/markdown';
      } else {
        content = data.content || JSON.stringify(data, null, 2);
        filename = `session-${sessionId.slice(0, 8)}.txt`;
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

  const buttons: { format: 'json' | 'markdown' | 'txt'; label: string; icon: React.ReactNode }[] = [
    { format: 'json',     label: 'JSON',     icon: <FileJson size={13} /> },
    { format: 'markdown', label: 'Markdown', icon: <FileText size={13} /> },
    { format: 'txt',      label: 'Text',     icon: <File size={13} /> },
  ];

  return (
    <div className="flex flex-col gap-1">
      {buttons.map(({ format, label, icon }) => (
        <button
          key={format}
          onClick={() => handleExport(format)}
          disabled={exporting !== null}
          className="flex items-center gap-2 px-3 py-2 text-xs text-[#a3a3a3] hover:text-white hover:bg-[#1a1a1a] rounded-lg transition-colors disabled:opacity-50 w-full text-left"
          title={`Export as ${label}`}
        >
          {exporting === format ? <Loader2 className="animate-spin" size={13} /> : icon}
          {label}
        </button>
      ))}
    </div>
  );
}
