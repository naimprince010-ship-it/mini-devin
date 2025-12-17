import { useState, useCallback, useRef } from 'react';
import { UploadedFile } from '../types';
import { useApi } from '../hooks/useApi';
import { Upload, Trash2, Download, Loader2 } from 'lucide-react';

interface FileUploadProps {
  sessionId: string;
  onFileUploaded?: (file: UploadedFile) => void;
}

export function FileUpload({ sessionId, onFileUploaded }: FileUploadProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const api = useApi();

  const loadFiles = useCallback(async () => {
    try {
      const data = await api.listFiles(sessionId);
      setFiles(data);
    } catch (e) {
      console.error('Failed to load files:', e);
    }
  }, [sessionId, api]);

  useState(() => {
    loadFiles();
  });

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    for (const file of droppedFiles) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    setUploading(true);
    try {
      const uploaded = await api.uploadFile(sessionId, file);
      setFiles(prev => [...prev, uploaded]);
      onFileUploaded?.(uploaded);
    } catch (e) {
      console.error('Failed to upload file:', e);
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    for (const file of selectedFiles) {
      await uploadFile(file);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDelete = async (fileId: string) => {
    try {
      await api.deleteFile(sessionId, fileId);
      setFiles(prev => prev.filter(f => f.file_id !== fileId));
    } catch (e) {
      console.error('Failed to delete file:', e);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) return '🖼️';
    if (mimeType.startsWith('text/')) return '📄';
    if (mimeType.includes('json')) return '📋';
    if (mimeType.includes('pdf')) return '📕';
    return '📁';
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
        <Upload size={16} />
        Files
      </h3>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
          isDragging
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-gray-600 hover:border-gray-500'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />
        {uploading ? (
          <div className="flex items-center justify-center gap-2 text-gray-400">
            <Loader2 className="animate-spin" size={20} />
            <span>Uploading...</span>
          </div>
        ) : (
          <div className="text-gray-400">
            <Upload className="mx-auto mb-2" size={24} />
            <p className="text-sm">Drop files here or click to upload</p>
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          {files.map((file) => (
            <div
              key={file.file_id}
              className="flex items-center justify-between p-2 bg-gray-700 rounded-lg"
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className="text-lg">{getFileIcon(file.mime_type)}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{file.filename}</p>
                  <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <a
                  href={`${import.meta.env.VITE_API_URL || ''}/api/sessions/${sessionId}/files/${file.file_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-1.5 text-gray-400 hover:text-blue-400 rounded"
                  title="Download"
                >
                  <Download size={14} />
                </a>
                <button
                  onClick={() => handleDelete(file.file_id)}
                  className="p-1.5 text-gray-400 hover:text-red-400 rounded"
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
