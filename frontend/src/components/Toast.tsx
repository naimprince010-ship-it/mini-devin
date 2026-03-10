import { useEffect, useState } from 'react';
import { CheckCircle2, AlertCircle, Info, X, AlertTriangle } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
    id: string;
    type: ToastType;
    title: string;
    message?: string;
    duration?: number; // ms, default 5000. 0 = sticky
}

interface ToastItemProps {
    toast: Toast;
    onDismiss: (id: string) => void;
}

const ICONS = {
    success: CheckCircle2,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
};

const COLORS = {
    success: 'border-[#00ff99]/30 bg-[#00ff99]/5 text-[#00ff99]',
    error: 'border-red-500/30 bg-red-500/5 text-red-400',
    warning: 'border-yellow-500/30 bg-yellow-500/5 text-yellow-400',
    info: 'border-blue-500/30 bg-blue-500/5 text-blue-400',
};

function ToastItem({ toast, onDismiss }: ToastItemProps) {
    const Icon = ICONS[toast.type];
    const colorClass = COLORS[toast.type];
    const duration = toast.duration ?? 5000;

    useEffect(() => {
        if (duration === 0) return;
        const t = setTimeout(() => onDismiss(toast.id), duration);
        return () => clearTimeout(t);
    }, [toast.id, duration, onDismiss]);

    return (
        <div
            className={`flex items-start gap-3 p-3 rounded-lg border backdrop-blur-md shadow-xl max-w-sm animate-in slide-in-from-right-8 duration-300 ${colorClass}`}
            style={{
                background: 'rgba(10,10,10,0.9)',
                borderColor: toast.type === 'success' ? 'rgba(0,255,153,0.2)'
                    : toast.type === 'error' ? 'rgba(239,68,68,0.2)'
                        : toast.type === 'warning' ? 'rgba(234,179,8,0.2)'
                            : 'rgba(59,130,246,0.2)',
            }}
        >
            <Icon size={15} className="flex-shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white">{toast.title}</p>
                {toast.message && (
                    <p className="text-[11px] text-[#a3a3a3] mt-0.5 leading-relaxed">{toast.message}</p>
                )}
            </div>
            <button
                onClick={() => onDismiss(toast.id)}
                className="flex-shrink-0 p-0.5 text-[#525252] hover:text-white transition-colors mt-0.5"
            >
                <X size={12} />
            </button>
        </div>
    );
}

interface ToastContainerProps {
    toasts: Toast[];
    onDismiss: (id: string) => void;
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
    if (toasts.length === 0) return null;

    return (
        <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-none">
            {toasts.map(t => (
                <div key={t.id} className="pointer-events-auto">
                    <ToastItem toast={t} onDismiss={onDismiss} />
                </div>
            ))}
        </div>
    );
}

// Hook for managing toasts
let _addToast: ((toast: Omit<Toast, 'id'>) => void) | null = null;

export function useToast() {
    const add = (toast: Omit<Toast, 'id'>) => {
        if (_addToast) _addToast(toast);
    };

    return {
        success: (title: string, message?: string) => add({ type: 'success', title, message }),
        error: (title: string, message?: string) => add({ type: 'error', title, message }),
        warning: (title: string, message?: string) => add({ type: 'warning', title, message }),
        info: (title: string, message?: string) => add({ type: 'info', title, message }),
    };
}

export function useToastState() {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const addToast = (toast: Omit<Toast, 'id'>) => {
        const id = Math.random().toString(36).slice(2);
        setToasts(prev => [...prev, { ...toast, id }]);
    };

    const dismiss = (id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    // Register global handler
    useEffect(() => {
        _addToast = addToast;
        return () => { _addToast = null; };
    }, []);

    return { toasts, addToast, dismiss };
}
