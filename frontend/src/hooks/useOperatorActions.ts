import { useCallback, useMemo, useState } from 'react';

export type OperatorActionType =
  | 'acknowledge_incident'
  | 'mark_investigation_started'
  | 'pause_runtime'
  | 'resume_runtime'
  | 'retry_task'
  | 'replay_session'
  | 'quarantine_runtime'
  | 'jump_to_diagnostics';

export type OperatorActionStatus = 'pending' | 'succeeded' | 'failed';

export type RuntimeControlState = 'running' | 'paused' | 'quarantined';

export interface OperatorActionAuditPayload {
  action_id: string;
  action_type: OperatorActionType;
  requested_at: string;
  requested_by: string;
  target: string;
  reason: string;
  confirmation_gate: 'explicit';
  dry_run: boolean;
  metadata?: Record<string, unknown>;
}

export interface OperatorActionRecord {
  audit: OperatorActionAuditPayload;
  status: OperatorActionStatus;
  message: string;
  completed_at?: string;
  stubbed: boolean;
}

interface ExecuteActionInput {
  action_type: OperatorActionType;
  requested_by: string;
  target: string;
  reason: string;
  metadata?: Record<string, unknown>;
}

function nowIso(): string {
  return new Date().toISOString();
}

function makeActionId(): string {
  return `op_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function baseMessage(action: OperatorActionType): string {
  if (action === 'acknowledge_incident') return 'Incident acknowledged by operator.';
  if (action === 'mark_investigation_started') return 'Investigation marked as started.';
  if (action === 'pause_runtime') return 'Runtime pause requested (stubbed).';
  if (action === 'resume_runtime') return 'Runtime resume requested (stubbed).';
  if (action === 'retry_task') return 'Retry task scaffold queued (stubbed).';
  if (action === 'replay_session') return 'Replay session scaffold queued (stubbed).';
  if (action === 'quarantine_runtime') return 'Runtime quarantine scaffold requested (stubbed).';
  return 'Jumped to diagnostics.';
}

export function useOperatorActions() {
  const [records, setRecords] = useState<OperatorActionRecord[]>([]);
  const [runtimeState, setRuntimeState] = useState<RuntimeControlState>('running');
  const [inFlightByAction, setInFlightByAction] = useState<Record<OperatorActionType, boolean>>({
    acknowledge_incident: false,
    mark_investigation_started: false,
    pause_runtime: false,
    resume_runtime: false,
    retry_task: false,
    replay_session: false,
    quarantine_runtime: false,
    jump_to_diagnostics: false,
  });

  const executeAction = useCallback(async (input: ExecuteActionInput): Promise<OperatorActionRecord> => {
    const audit: OperatorActionAuditPayload = {
      action_id: makeActionId(),
      action_type: input.action_type,
      requested_at: nowIso(),
      requested_by: input.requested_by,
      target: input.target,
      reason: input.reason,
      confirmation_gate: 'explicit',
      dry_run: true,
      metadata: input.metadata,
    };

    const pending: OperatorActionRecord = {
      audit,
      status: 'pending',
      message: baseMessage(input.action_type),
      stubbed: true,
    };

    setInFlightByAction((prev) => ({ ...prev, [input.action_type]: true }));
    setRecords((prev) => [pending, ...prev]);

    try {
      await new Promise((resolve) => setTimeout(resolve, 700));

      if (input.action_type === 'pause_runtime') {
        setRuntimeState('paused');
      } else if (input.action_type === 'resume_runtime') {
        setRuntimeState('running');
      } else if (input.action_type === 'quarantine_runtime') {
        setRuntimeState('quarantined');
      }

      const success: OperatorActionRecord = {
        ...pending,
        status: 'succeeded',
        completed_at: nowIso(),
      };
      setRecords((prev) => prev.map((item) => (item.audit.action_id === audit.action_id ? success : item)));
      return success;
    } catch {
      const failed: OperatorActionRecord = {
        ...pending,
        status: 'failed',
        completed_at: nowIso(),
        message: `${baseMessage(input.action_type)} Action failed in scaffold mode.`,
      };
      setRecords((prev) => prev.map((item) => (item.audit.action_id === audit.action_id ? failed : item)));
      return failed;
    } finally {
      setInFlightByAction((prev) => ({ ...prev, [input.action_type]: false }));
    }
  }, []);

  const pendingCount = useMemo(
    () => Object.values(inFlightByAction).filter(Boolean).length,
    [inFlightByAction],
  );

  return {
    records,
    runtimeState,
    inFlightByAction,
    pendingCount,
    executeAction,
  };
}
