/**
 * SessionEventsContext
 * Shared state for WebSocket events (tool calls, phase, iteration, plan steps, file edits, browser events)
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

export interface PlanStep {
    id: string;
    text: string;
    status: 'pending' | 'running' | 'done' | 'failed';
}

export interface FileEdit {
    path: string;
    content: string;
    before?: string;
    timestamp: Date;
    toolId: string;
}

export interface BrowserEvent {
    id: string;
    type: 'navigate' | 'screenshot' | 'search' | 'click' | 'other';
    url?: string;
    query?: string;
    screenshotBase64?: string;
    timestamp: Date;
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
    planSteps: PlanStep[];
    currentStepIndex: number;
    fileEdits: FileEdit[];
    browserEvents: BrowserEvent[];
    acceptanceCriteria: string;
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
    onPlanCreated: (steps: string[]) => void;
    onStepStarted: (index: number) => void;
    onStepCompleted: (index: number) => void;
    setAcceptanceCriteria: (criteria: string) => void;
}

const defaultState: SessionEventsState = {
    phase: null,
    iteration: 0,
    maxIterations: 50,
    toolCalls: [],
    shellLines: [],
    isRunning: false,
    planSteps: [],
    currentStepIndex: -1,
    fileEdits: [],
    browserEvents: [],
    acceptanceCriteria: '',
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
    onPlanCreated: () => { },
    onStepStarted: () => { },
    onStepCompleted: () => { },
    setAcceptanceCriteria: () => { },
});

const BROWSER_TOOLS = new Set(['browser_navigate', 'browser_screenshot', 'browser_click', 'web_search', 'search_web', 'browse', 'browser']);
const FILE_WRITE_TOOLS = new Set(['write_file', 'edit_file', 'str_replace_editor', 'create_file', 'write_to_file', 'patch_file', 'overwrite_file']);

export function SessionEventsProvider({ children }: { children: React.ReactNode }) {
    const [state, setState] = useState<SessionEventsState>(defaultState);

    const onPhaseChange = useCallback((phase: AgentPhase) => {
        setState(prev => ({ ...prev, phase }));
    }, []);

    const onPlanCreated = useCallback((steps: string[]) => {
        const planSteps: PlanStep[] = steps.map((text, i) => ({
            id: `step-${i}`,
            text,
            status: 'pending',
        }));
        setState(prev => ({ ...prev, planSteps, currentStepIndex: -1 }));
    }, []);

    const onStepStarted = useCallback((index: number) => {
        setState(prev => ({
            ...prev,
            currentStepIndex: index,
            planSteps: prev.planSteps.map((s, i) =>
                i === index ? { ...s, status: 'running' } : s
            ),
        }));
    }, []);

    const onStepCompleted = useCallback((index: number) => {
        setState(prev => ({
            ...prev,
            planSteps: prev.planSteps.map((s, i) =>
                i === index ? { ...s, status: 'done' } : s
            ),
        }));
    }, []);

    const setAcceptanceCriteria = useCallback((criteria: string) => {
        setState(prev => ({ ...prev, acceptanceCriteria: criteria }));
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
        setState(prev => {
            const newState: SessionEventsState = { ...prev, toolCalls: [...prev.toolCalls, entry] };

            // Shell lines for command tools
            if (tool === 'run_command' || tool === 'bash' || tool === 'shell') {
                const cmd = (input.command as string) || (input.cmd as string) || JSON.stringify(input);
                newState.shellLines = [...prev.shellLines, `$ ${cmd}`];
            }

            // Browser events
            if (BROWSER_TOOLS.has(tool)) {
                const browserEvent: BrowserEvent = {
                    id,
                    type: tool.includes('navigate') || tool === 'browse' ? 'navigate'
                        : tool.includes('screenshot') ? 'screenshot'
                            : tool.includes('search') ? 'search'
                                : tool.includes('click') ? 'click'
                                    : 'other',
                    url: (input.url as string) || (input.query as string) || undefined,
                    query: (input.query as string) || undefined,
                    timestamp: new Date(),
                };
                newState.browserEvents = [...prev.browserEvents, browserEvent];
            }

            return newState;
        });
        return id;
    }, []);

    const onToolCompleted = useCallback(
        (id: string, output: Record<string, unknown>, durationMs: number) => {
            setState(prev => {
                const tool = prev.toolCalls.find(t => t.id === id);
                const newState: SessionEventsState = {
                    ...prev,
                    toolCalls: prev.toolCalls.map(t =>
                        t.id === id
                            ? { ...t, status: 'completed', output, completedAt: new Date(), durationMs }
                            : t
                    ),
                };

                // Shell output
                if (tool?.tool === 'run_command' || tool?.tool === 'bash') {
                    newState.shellLines = [
                        ...prev.shellLines,
                        ...(typeof output.stdout === 'string'
                            ? output.stdout.split('\n').filter(Boolean)
                            : typeof output.output === 'string'
                                ? output.output.split('\n').filter(Boolean)
                                : []),
                    ];
                }

                // File edits
                if (tool && FILE_WRITE_TOOLS.has(tool.tool)) {
                    const path = (tool.input.path as string) || (tool.input.file_path as string) || (tool.input.filename as string) || 'unknown';
                    const content = (tool.input.content as string) || (tool.input.new_content as string) || (output.content as string) || '';
                    const fileEdit: FileEdit = {
                        path,
                        content,
                        timestamp: new Date(),
                        toolId: id,
                    };
                    // Replace if same path already edited
                    const existing = prev.fileEdits.findIndex(f => f.path === path);
                    if (existing >= 0) {
                        const updated = [...prev.fileEdits];
                        fileEdit.before = updated[existing].content;
                        updated[existing] = fileEdit;
                        newState.fileEdits = updated;
                    } else {
                        newState.fileEdits = [...prev.fileEdits, fileEdit];
                    }
                }

                // Browser screenshot
                if (tool && BROWSER_TOOLS.has(tool.tool)) {
                    const screenshot = (output.screenshot as string) || (output.image as string) || undefined;
                    const url = (output.url as string) || (tool.input.url as string) || undefined;
                    if (screenshot || url) {
                        newState.browserEvents = prev.browserEvents.map(ev =>
                            ev.id === id
                                ? { ...ev, screenshotBase64: screenshot, url: url || ev.url }
                                : ev
                        );
                    }
                }

                return newState;
            });
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
        setState(prev => ({
            ...prev,
            isRunning: true,
            toolCalls: [],
            shellLines: [],
            phase: null,
            iteration: 0,
            planSteps: [],
            currentStepIndex: -1,
            fileEdits: [],
            browserEvents: [],
        }));
    }, []);

    const onTaskEnded = useCallback(() => {
        setState(prev => ({
            ...prev,
            isRunning: false,
            planSteps: prev.planSteps.map(s =>
                s.status === 'running' ? { ...s, status: 'done' } : s
            ),
        }));
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
                onPlanCreated,
                onStepStarted,
                onStepCompleted,
                setAcceptanceCriteria,
            }}
        >
            {children}
        </SessionEventsContext.Provider>
    );
}

export function useSessionEvents() {
    return useContext(SessionEventsContext);
}
