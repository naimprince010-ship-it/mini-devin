import { useEffect, useState } from 'react';

interface OperatorActionConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  confirmPhrase?: string;
  requireReason?: boolean;
  onCancel: () => void;
  onConfirm: (payload: { reason: string }) => void;
}

export default function OperatorActionConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  confirmPhrase = 'APPROVE',
  requireReason = true,
  onCancel,
  onConfirm,
}: OperatorActionConfirmDialogProps) {
  const [reason, setReason] = useState('');
  const [gateInput, setGateInput] = useState('');

  useEffect(() => {
    if (!open) {
      setReason('');
      setGateInput('');
    }
  }, [open]);

  if (!open) {
    return null;
  }

  const gateOk = gateInput.trim().toUpperCase() === confirmPhrase.toUpperCase();
  const reasonOk = !requireReason || reason.trim().length >= 8;
  const allowConfirm = gateOk && reasonOk;

  return (
    <div className="fixed inset-0 z-[70] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-lg rounded-xl border border-[#2a2a2a] bg-[#111111] p-4">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <p className="mt-2 text-xs text-[#a3a3a3] leading-relaxed">{description}</p>

        <div className="mt-4 space-y-3">
          <label className="block text-xs text-[#a3a3a3]">
            Operator reason
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={3}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-2 text-sm text-white"
              placeholder="Write a short operational reason for audit trail"
            />
          </label>

          <label className="block text-xs text-[#a3a3a3]">
            Type {confirmPhrase} to confirm
            <input
              value={gateInput}
              onChange={(e) => setGateInput(e.target.value)}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-2 text-sm text-white"
              placeholder={confirmPhrase}
            />
          </label>
        </div>

        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded border border-[#2a2a2a] bg-[#161616] px-3 py-1.5 text-xs text-[#d4d4d4]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!allowConfirm}
            onClick={() => onConfirm({ reason: reason.trim() })}
            className="rounded border border-[#00ff99]/30 bg-[#00ff99]/10 px-3 py-1.5 text-xs font-medium text-[#00ff99] disabled:opacity-50"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
