import React, { useState } from 'react';
import { CheckCircle2, Circle, Loader2, XCircle, ChevronDown, ChevronRight, ListChecks } from 'lucide-react';
import type { PlanStep } from '../contexts/SessionEventsContext';

interface PlanStepsViewProps {
    steps: PlanStep[];
    currentIndex: number;
    isRunning: boolean;
}

export function PlanStepsView({ steps, currentIndex, isRunning }: PlanStepsViewProps) {
    const [collapsed, setCollapsed] = useState(false);

    if (steps.length === 0) return null;

    const doneCount = steps.filter(s => s.status === 'done').length;

    return (
        <div className="mx-6 mb-4 rounded-xl border border-[#1e1e1e] bg-[#0d0d0d] overflow-hidden">
            {/* Header */}
            <button
                onClick={() => setCollapsed(c => !c)}
                className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-[#141414] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <ListChecks size={13} className="text-[#00ff99]" />
                    <span className="text-[11px] font-bold uppercase tracking-wider text-[#a3a3a3]">
                        Agent Plan
                    </span>
                    <span className="text-[10px] text-[#525252] font-mono">
                        {doneCount}/{steps.length}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    {isRunning && doneCount < steps.length && (
                        <span className="text-[9px] text-[#00ff99] uppercase tracking-widest font-bold animate-pulse">
                            Running
                        </span>
                    )}
                    {collapsed ? (
                        <ChevronRight size={12} className="text-[#525252]" />
                    ) : (
                        <ChevronDown size={12} className="text-[#525252]" />
                    )}
                </div>
            </button>

            {/* Steps */}
            {!collapsed && (
                <div className="px-4 pb-3 space-y-1.5">
                    {/* Progress bar */}
                    <div className="h-0.5 bg-[#1a1a1a] rounded-full mb-3 overflow-hidden">
                        <div
                            className="h-full bg-[#00ff99] rounded-full transition-all duration-700"
                            style={{ width: `${steps.length ? (doneCount / steps.length) * 100 : 0}%` }}
                        />
                    </div>

                    {steps.map((step, i) => (
                        <div
                            key={step.id}
                            className={`flex items-start gap-2.5 py-1 transition-opacity ${step.status === 'pending' && i > currentIndex + 1 ? 'opacity-40' : 'opacity-100'
                                }`}
                        >
                            {/* Icon */}
                            <div className="flex-shrink-0 mt-0.5">
                                {step.status === 'done' ? (
                                    <CheckCircle2 size={14} className="text-[#00ff99]" />
                                ) : step.status === 'running' ? (
                                    <Loader2 size={14} className="text-[#00ff99] animate-spin" />
                                ) : step.status === 'failed' ? (
                                    <XCircle size={14} className="text-red-500" />
                                ) : (
                                    <Circle size={14} className="text-[#3a3a3a]" />
                                )}
                            </div>

                            {/* Text */}
                            <span
                                className={`text-xs leading-relaxed ${step.status === 'done'
                                        ? 'text-[#525252] line-through'
                                        : step.status === 'running'
                                            ? 'text-white font-medium'
                                            : step.status === 'failed'
                                                ? 'text-red-400'
                                                : 'text-[#737373]'
                                    }`}
                            >
                                <span className="text-[#3a3a3a] font-mono mr-1">{i + 1}.</span>
                                {step.text}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
