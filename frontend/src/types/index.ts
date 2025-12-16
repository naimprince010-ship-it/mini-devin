// API Types

export interface Session {
  session_id: string;
  created_at: string;
  status: string;
  working_directory: string;
  current_task: string | null;
  iteration: number;
  total_tasks: number;
}

export interface Task {
  task_id: string;
  session_id: string;
  description: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  iteration: number;
  error_message: string | null;
}

export interface TaskResult {
  task_id: string;
  status: string;
  summary: string;
  files_modified: string[];
  commands_executed: string[];
  total_tokens: number;
  duration_seconds: number;
}

export interface CreateSessionRequest {
  working_directory: string;
  model: string;
  max_iterations: number;
}

export interface CreateTaskRequest {
  description: string;
  acceptance_criteria: string[];
}

// WebSocket Message Types

export type MessageType =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'task_started'
  | 'task_completed'
  | 'task_failed'
  | 'task_cancelled'
  | 'phase_changed'
  | 'iteration_started'
  | 'iteration_completed'
  | 'token'
  | 'tokens_batch'
  | 'tool_started'
  | 'tool_completed'
  | 'tool_failed'
  | 'tool_output'
  | 'plan_created'
  | 'plan_updated'
  | 'step_completed'
  | 'verification_started'
  | 'verification_completed'
  | 'repair_started'
  | 'repair_completed';

export interface WebSocketMessage {
  type: MessageType;
  data: Record<string, unknown>;
  timestamp: string;
  session_id: string | null;
  task_id: string | null;
}

export interface SystemStatus {
  status: string;
  version: string;
  active_sessions: number;
  total_tasks_completed: number;
  uptime_seconds: number;
}

export interface Model {
  id: string;
  name: string;
  provider: string;
}
