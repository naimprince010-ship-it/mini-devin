import { useState, useCallback } from 'react';
import {
  Session,
  Task,
  TaskResult,
  CreateSessionRequest,
  CreateTaskRequest,
  SystemStatus,
  Model,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : '/api';

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearError = useCallback(() => setError(null), []);

  // Sessions
  const createSession = useCallback(async (data: CreateSessionRequest): Promise<Session> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<Session>('/sessions', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create session');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const listSessions = useCallback(async (): Promise<Session[]> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<{ sessions: Session[] } | Session[]>('/sessions');
      return Array.isArray(result) ? result : result.sessions;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list sessions');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const getSession = useCallback(async (sessionId: string): Promise<Session> => {
    setLoading(true);
    setError(null);
    try {
      return await fetchApi<Session>(`/sessions/${sessionId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get session');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteSession = useCallback(async (sessionId: string): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      await fetchApi(`/sessions/${sessionId}`, { method: 'DELETE' });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete session');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  // Tasks
  const createTask = useCallback(async (
    sessionId: string,
    data: CreateTaskRequest
  ): Promise<Task> => {
    setLoading(true);
    setError(null);
    try {
      return await fetchApi<Task>(`/sessions/${sessionId}/tasks`, {
        method: 'POST',
        body: JSON.stringify(data),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create task');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const listTasks = useCallback(async (sessionId: string): Promise<Task[]> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<{ tasks: Task[] } | Task[]>(`/sessions/${sessionId}/tasks`);
      return Array.isArray(result) ? result : result.tasks;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list tasks');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const getTask = useCallback(async (sessionId: string, taskId: string): Promise<Task> => {
    setLoading(true);
    setError(null);
    try {
      return await fetchApi<Task>(`/sessions/${sessionId}/tasks/${taskId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get task');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const getTaskResult = useCallback(async (
    sessionId: string,
    taskId: string
  ): Promise<TaskResult> => {
    setLoading(true);
    setError(null);
    try {
      return await fetchApi<TaskResult>(`/sessions/${sessionId}/tasks/${taskId}/result`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get task result');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const getTaskOutput = useCallback(async (
    sessionId: string,
    taskId: string
  ): Promise<{ task_id: string; status: string; outputs: Array<{ type: string; content: string }>; result: string | null }> => {
    try {
      return await fetchApi(`/sessions/${sessionId}/tasks/${taskId}/output`);
    } catch (e) {
      throw e;
    }
  }, []);

  const cancelTask = useCallback(async (sessionId: string, taskId: string): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      await fetchApi(`/sessions/${sessionId}/tasks/${taskId}/cancel`, { method: 'POST' });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to cancel task');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  // Artifacts
  const listArtifacts = useCallback(async (
    sessionId: string,
    taskId: string
  ): Promise<string[]> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<{ artifacts: string[] }>(
        `/sessions/${sessionId}/tasks/${taskId}/artifacts`
      );
      return result.artifacts;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list artifacts');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const getArtifact = useCallback(async (
    sessionId: string,
    taskId: string,
    artifactName: string
  ): Promise<string> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<{ content: string }>(
        `/sessions/${sessionId}/tasks/${taskId}/artifacts/${artifactName}`
      );
      return result.content;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get artifact');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  // System
  const getStatus = useCallback(async (): Promise<SystemStatus> => {
    setLoading(true);
    setError(null);
    try {
      return await fetchApi<SystemStatus>('/status');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get status');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const listModels = useCallback(async (): Promise<Model[]> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchApi<{ models: Model[] }>('/models');
      return result.models;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list models');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    clearError,
    // Sessions
    createSession,
    listSessions,
    getSession,
    deleteSession,
    // Tasks
    createTask,
    listTasks,
    getTask,
    getTaskResult,
    getTaskOutput,
    cancelTask,
    // Artifacts
    listArtifacts,
    getArtifact,
    // System
    getStatus,
    listModels,
  };
}
