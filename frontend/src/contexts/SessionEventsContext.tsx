/**
 * SessionEventsContext
 * Shared state for WebSocket events (tool calls, phase, iteration)
 * so both TaskPanel and WorkspacePanel can access the same data.
 */
import React, { createContext, useContext, useState, useCallback } from 'react';

export interface ToolCallEntry {
    id: string;
    tool: string;
    input: Record<string, unknown>;
    output?: Record<string, unknown>;
    status: 'running' | 'completed' | 'failed';
    startedAt: Date;
    completedAt?: Date;
    durationMs?: number;
}

export type AgentPhase =
    | 'intake'
    | 'explore'
    | 'plan'
    | 'execute'
    | 'verify'
    | 'repair'
    | 'finalize'
    | 'complete'
    | null;

interface SessionEventsState {
    phase: AgentPhase;
    iteration: number;
    maxIterations: number;
    toolCalls: ToolCallEntry[];
    shellLines: string[];
    isRunning: boolean;
}

interface SessionEventsContextValue extends SessionEventsState {
    onPhaseChange: (phase: AgentPhase) => void;
    onToolStarted: (tool: string, input: Record<string, unknown>) => string;
    onToolCompleted: (id: string, output: Record<string, unknown>, durationMs: number) => void;
    onToolFailed: (id: string, error: string) => void;
    onShellLine: (line: string) => void;
    onIterationUpdate: (iteration: number, max?: number) => void;
    onTaskStarted: () => void;
    onTaskEnded: () => void;
    resetForSession: () => void;
}

const defaultState: SessionEventsState = {
    phase: null,
    iteration: 0,
    maxIterations: 50,
    toolCalls: [],
    shellLines: [],
    isRunning: false,
};

const SessionEventsContext = createContext<SessionEventsContextValue>({
    ...defaultState,
    onPhaseChange: () => { },
    onToolStarted: () => '',
    onToolCompleted: () => { },
    onToolFailed: () => { },
    onShellLine: () => { },
    onIterationUpdate: () => { },
    onTaskStarted: () => { },
    onTaskEnded: () => { },
    resetForSession: () => { },
});

export function SessionEventsProvider({ children }: { children: React.ReactNode }) {
    const [state, setState] = useState<SessionEventsState>(defaultState);

    const onPhaseChange = useCallback((phase: AgentPhase) => {
        setState(prev => ({ ...prev, phase }));
    }, []);

    const onToolStarted = useCallback((tool: string, input: Record<string, unknown>): string => {
        const id = `tool-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        const entry: ToolCallEntry = {
            id,
            tool,
            input,
            status: 'running',
            startedAt: new Date(),
        };
        setState(prev => ({ ...prev, toolCalls: [...prev.toolCalls, entry] }));

        // Also add shell line for commands
        if (tool === 'run_command' || tool === 'bash' || tool === 'shell') {
            const cmd = (input.command as string) || (input.cmd as string) || JSON.stringify(input);
            setState(prev => ({
                ...prev,
                shellLines: [...prev.shellLines, `$ ${cmd}`],
            }));
        }
        return id;
    }, []);

    const onToolCompleted = useCallback(
        (id: string, output: Record<string, unknown>, durationMs: number) => {
            setState(prev => ({
                ...prev,
                toolCalls: prev.toolCalls.map(t =>
                    t.id === id
                        ? { ...t, status: 'completed', output, completedAt: new Date(), durationMs }
                        : t
                ),
                shellLines:
                    prev.toolCalls.find(t => t.id === id)?.tool === 'run_command' ||
                        prev.toolCalls.find(t => t.id === id)?.tool === 'bash'
                        ? [
                            ...prev.shellLines,
                            ...(typeof output.stdout === 'string'
                                ? output.stdout.split('\n').filter(Boolean)
                                : typeof output.output === 'string'
                                    ? output.output.split('\n').filter(Boolean)
                                    : []),
                        ]
                        : prev.shellLines,
            }));
        },
        []
    );

    const onToolFailed = useCallback((id: string, error: string) => {
        setState(prev => ({
            ...prev,
            toolCalls: prev.toolCalls.map(t =>
                t.id === id
                    ? {
                        ...t,
                        status: 'failed',
                        output: { error },
                        completedAt: new Date(),
                    }
                    : t
            ),
        }));
    }, []);

    const onShellLine = useCallback((line: string) => {
        setState(prev => ({ ...prev, shellLines: [...prev.shellLines, line] }));
    }, []);

    const onIterationUpdate = useCallback((iteration: number, max?: number) => {
        setState(prev => ({
            ...prev,
            iteration,
            ...(max !== undefined ? { maxIterations: max } : {}),
        }));
    }, []);

    const onTaskStarted = useCallback(() => {
        setState(prev => ({ ...prev, isRunning: true, toolCalls: [], shellLines: [], phase: null, iteration: 0 }));
    }, []);

    const onTaskEnded = useCallback(() => {
        setState(prev => ({ ...prev, isRunning: false }));
    }, []);

    const resetForSession = useCallback(() => {
        setState(defaultState);
    }, []);

    return (
        <SessionEventsContext.Provider
            value={{
                ...state,
                onPhaseChange,
                onToolStarted,
                onToolCompleted,
                onToolFailed,
                onShellLine,
                onIterationUpdate,
                onTaskStarted,
                onTaskEnded,
                resetForSession,
            }}
        >
            {children}
        </SessionEventsContext.Provider>
    );
}

export function useSessionEvents() {
    return useContext(SessionEventsContext);
}
