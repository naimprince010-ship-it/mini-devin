import React, { useState } from 'react';
import { CheckCircle2, Circle, Loader2, XCircle, ChevronDown, ChevronRight, Sparkles, Zap } from 'lucide-react';
import type { PlanStep } from '../contexts/SessionEventsContext';

interface PlanStepsViewProps {
    steps: PlanStep[];
    currentIndex: number;
    isRunning: boolean;
}

function StepStatusIcon({ status }: { status: PlanStep['status'] }) {
    if (status === 'done') {
        return (
            <div className="relative flex items-center justify-center w-6 h-6 rounded-full bg-[#00ff99]/15 border border-[#00ff99]/40 shadow-[0_0_8px_rgba(0,255,153,0.2)]">
                <CheckCircle2 size={13} className="text-[#00ff99]" />
            </div>
        );
    }
    if (status === 'running') {
        return (
            <div className="relative flex items-center justify-center w-6 h-6 rounded-full bg-[#3399ff]/15 border border-[#3399ff]/50 shadow-[0_0_10px_rgba(51,153,255,0.3)]">
                <Loader2 size={12} className="text-[#3399ff] animate-spin" />
                {/* Outer pulse ring */}
                <span className="absolute inset-0 rounded-full border border-[#3399ff]/30 animate-ping" />
            </div>
        );
    }
    if (status === 'failed') {
        return (
            <div className="flex items-center justify-center w-6 h-6 rounded-full bg-red-500/15 border border-red-500/40">
                <XCircle size={13} className="text-red-400" />
            </div>
        );
    }
    // pending
    return (
        <div className="flex items-center justify-center w-6 h-6 rounded-full border border-[#2a2a2a] bg-[#111]">
            <Circle size={9} className="text-[#3a3a3a]" />
        </div>
    );
}

function StepRow({ step, index, isLast }: { step: PlanStep; index: number; isLast: boolean }) {
    const isRunning = step.status === 'running';
    const isDone = step.status === 'done';
    const isFailed = step.status === 'failed';
    const isPending = step.status === 'pending';

    return (
        <div className="flex gap-3">
            {/* Left column: icon + connector line */}
            <div className="flex flex-col items-center flex-shrink-0">
                <StepStatusIcon status={step.status} />
                {!isLast && (
                    <div
                        className={`w-px flex-1 mt-1 mb-0.5 transition-colors duration-500 ${isDone ? 'bg-[#00ff99]/30' : 'bg-[#1e1e1e]'
                            }`}
                        style={{ minHeight: '16px' }}
                    />
                )}
            </div>

            {/* Right column: step content */}
            <div className={`flex-1 pb-3 ${isLast ? 'pb-1' : ''}`}>
                {/* Step number + label row */}
                <div className="flex items-center gap-2 mb-0.5">
                    <span
                        className={`text-[9px] font-black uppercase tracking-[0.2em] font-mono ${isDone ? 'text-[#00ff99]/50' :
                            isRunning ? 'text-[#3399ff]' :
                                isFailed ? 'text-red-500/70' :
                                    'text-[#2e2e2e]'
                            }`}
                    >
                        STEP {String(index + 1).padStart(2, '0')}
                    </span>
                    {isRunning && (
                        <span className="text-[9px] font-bold uppercase tracking-widest text-[#3399ff] animate-pulse">
                            · IN PROGRESS
                        </span>
                    )}
                    {isDone && (
                        <span className="text-[9px] font-bold uppercase tracking-widest text-[#00ff99]/60">
                            · DONE
                        </span>
                    )}
                    {isFailed && (
                        <span className="text-[9px] font-bold uppercase tracking-widest text-red-500/70">
                            · FAILED
                        </span>
                    )}
                </div>

                {/* Step text */}
                <p
                    className={`text-[12px] leading-snug transition-colors duration-300 ${isDone ? 'text-[#404040] line-through decoration-[#333]' :
                        isRunning ? 'text-white font-medium' :
                            isFailed ? 'text-red-400/80' :
                                isPending ? 'text-[#555]' : 'text-[#737373]'
                        }`}
                >
                    {step.text}
                </p>

                {/* Running shimmer bar */}
                {isRunning && (
                    <div className="mt-1.5 h-0.5 w-full max-w-[200px] bg-[#1a2a3a] rounded-full overflow-hidden">
                        <div className="h-full w-1/2 bg-[#3399ff]/60 rounded-full animate-[shimmer_1.5s_ease-in-out_infinite]"
                            style={{ animation: 'shimmer 1.5s ease-in-out infinite' }}
                        />
                    </div>
                )}
            </div>
        </div>
    );
}

export function PlanStepsView({ steps, currentIndex, isRunning }: PlanStepsViewProps) {
    const [collapsed, setCollapsed] = useState(false);

    if (steps.length === 0) return null;

    const doneCount = steps.filter(s => s.status === 'done').length;
    const failedCount = steps.filter(s => s.status === 'failed').length;
    const progressPct = steps.length > 0 ? (doneCount / steps.length) * 100 : 0;
    const allDone = doneCount === steps.length;

    return (
        <div className={`mx-6 mb-4 rounded-2xl border overflow-hidden transition-all duration-300 ${allDone
            ? 'border-[#00ff99]/20 bg-gradient-to-b from-[#00ff99]/5 to-[#0d0d0d]'
            : isRunning
                ? 'border-[#3399ff]/20 bg-[#080c12]'
                : 'border-[#1e1e1e] bg-[#0a0a0a]'
            }`}>

            {/* ── Header ─────────────────────────────────────────────────── */}
            <button
                onClick={() => setCollapsed(c => !c)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/[0.02] transition-colors group"
            >
                <div className="flex items-center gap-2.5">
                    {/* Icon */}
                    {allDone ? (
                        <Sparkles size={13} className="text-[#00ff99]" />
                    ) : isRunning ? (
                        <Zap size={13} className="text-[#3399ff] animate-pulse" />
                    ) : (
                        <Zap size={13} className="text-[#525252]" />
                    )}

                    <span className={`text-[10px] font-black uppercase tracking-[0.15em] ${allDone ? 'text-[#00ff99]/80' : isRunning ? 'text-[#3399ff]/80' : 'text-[#525252]'
                        }`}>
                        Execution Plan
                    </span>

                    {/* Progress pill */}
                    <div className={`px-2 py-0.5 rounded-full text-[9px] font-bold font-mono border ${allDone ? 'border-[#00ff99]/30 text-[#00ff99]/70 bg-[#00ff99]/5' :
                        failedCount > 0 ? 'border-red-500/30 text-red-400/70 bg-red-500/5' :
                            'border-[#262626] text-[#525252] bg-[#111]'
                        }`}>
                        {doneCount}/{steps.length}
                    </div>

                    {isRunning && !allDone && (
                        <div className="flex items-center gap-1">
                            <span className="w-1 h-1 rounded-full bg-[#3399ff] animate-ping" />
                            <span className="text-[9px] uppercase tracking-[0.15em] font-bold text-[#3399ff]/60">
                                Running
                            </span>
                        </div>
                    )}
                </div>

                <div className="flex items-center gap-3">
                    {/* Mini progress bar in header */}
                    <div className="hidden sm:flex items-center gap-2">
                        <div className="w-20 h-1 bg-[#1a1a1a] rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-700 ease-out ${allDone ? 'bg-[#00ff99]' : isRunning ? 'bg-[#3399ff]' : 'bg-[#404040]'}`}
                                style={{ width: `${progressPct}%` }}
                            />
                        </div>
                        <span className="text-[9px] font-mono text-[#404040]">{Math.round(progressPct)}%</span>
                    </div>

                    <div className="text-[#3a3a3a] group-hover:text-[#525252] transition-colors">
                        {collapsed
                            ? <ChevronRight size={13} />
                            : <ChevronDown size={13} />
                        }
                    </div>
                </div>
            </button>

            {/* ── Steps ──────────────────────────────────────────────────── */}
            {!collapsed && (
                <div className="px-4 pt-2 pb-4">
                    {/* Full progress bar */}
                    <div className="h-px bg-[#1a1a1a] rounded-full mb-4 overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${allDone ? 'bg-[#00ff99]' : 'bg-[#3399ff]/60'}`}
                            style={{ width: `${progressPct}%` }}
                        />
                    </div>

                    <div>
                        {steps.map((step, i) => (
                            <StepRow
                                key={step.id}
                                step={step}
                                index={i}
                                isLast={i === steps.length - 1}
                            />
                        ))}
                    </div>

                    {/* Done state footer */}
                    {allDone && (
                        <div className="mt-3 pt-3 border-t border-[#00ff99]/10 flex items-center gap-2">
                            <CheckCircle2 size={11} className="text-[#00ff99]/60" />
                            <span className="text-[10px] text-[#00ff99]/50 font-medium">All steps completed</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
