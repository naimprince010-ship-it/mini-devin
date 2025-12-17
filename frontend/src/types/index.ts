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
  | 'iteration'
  | 'token'
  | 'tokens_batch'
  | 'thinking'
  | 'response'
  | 'tool_started'
  | 'tool_completed'
  | 'tool_failed'
  | 'tool_output'
  | 'tool_result'
  | 'plan_created'
  | 'plan_updated'
  | 'step_completed'
  | 'verification_started'
  | 'verification_completed'
  | 'repair_started'
  | 'repair_completed'
  | 'pong';

export interface ToolResultData {
  tool: string;
  success: boolean;
  output: string;
  error?: string;
}

export interface WebSocketMessage {
  type: MessageType;
  data?: Record<string, unknown>;
  timestamp?: string;
  session_id?: string | null;
  task_id?: string | null;
  content?: string;
  iteration?: number;
  max?: number;
  result?: ToolResultData;
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

export interface User {
  user_id: string;
  email: string;
  username: string;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface APIKey {
  api_key_id: string;
  name: string;
  key_prefix: string;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
}

export interface Provider {
  id: string;
  name: string;
  models: string[];
}

export interface UploadedFile {
  file_id: string;
  filename: string;
  size: number;
  mime_type: string;
  uploaded_at: string;
}

export interface Memory {
  memory_id: string;
  type: string;
  content: {
    key: string;
    value: string;
  };
  created_at: string;
  last_accessed: string;
  access_count: number;
}

export interface ExportResponse {
  format?: string;
  content?: string;
  session_id?: string;
  created_at?: string;
  model?: string;
  provider?: string | number;
  tasks?: Task[];
  memories?: Memory[];
  files?: UploadedFile[];
  exported_at?: string;
}
