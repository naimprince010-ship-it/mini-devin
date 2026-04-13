import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketMessage } from '../types';
import { getApiBase, getApiWsUrl, getSessionSseUrl } from '../config/apiBase';

export type StreamTransport = 'websocket' | 'sse';

function getTransport(): StreamTransport {
  const v = import.meta.env.VITE_STREAM_TRANSPORT;
  if (String(v || '').toLowerCase() === 'sse') {
    return 'sse';
  }
  return 'websocket';
}

interface UseSessionStreamOptions {
  sessionId?: string;
  onMessage?: (message: WebSocketMessage) => void;
}

/**
 * Real-time session events: WebSocket (default) or Server-Sent Events + REST send.
 * Set VITE_STREAM_TRANSPORT=sse when proxies block WebSockets.
 */
export function useSessionStream(options: UseSessionStreamOptions = {}) {
  const { sessionId } = options;
  const transport = getTransport();
  const onMessageRef = useRef(options.onMessage);
  useEffect(() => {
    onMessageRef.current = options.onMessage;
  }, [options.onMessage]);

  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const sendMessage = useCallback(
    async (data: string): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }
      if (transport === 'websocket') {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data);
          return true;
        }
        return false;
      }
      try {
        const base = getApiBase();
        const r = await fetch(`${base}/sessions/${sessionId}/agent/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: data }),
        });
        return r.ok;
      } catch {
        return false;
      }
    },
    [sessionId, transport],
  );

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    if (transport === 'websocket') {
      const connect = () => {
        const ws = new WebSocket(getApiWsUrl(sessionId));
        wsRef.current = ws;
        ws.onopen = () => setIsConnected(true);
        ws.onclose = () => {
          setIsConnected(false);
          if (wsRef.current === ws) {
            reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
          }
        };
        ws.onerror = () => {
          /* onclose handles reconnect */
        };
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as WebSocketMessage;
            onMessageRef.current?.(message);
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
          }
        };
      };
      connect();
      return () => {
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      };
    }

    const es = new EventSource(getSessionSseUrl(sessionId));
    esRef.current = es;
    es.onopen = () => setIsConnected(true);
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setIsConnected(false);
      }
    };
    es.onmessage = (ev) => {
      try {
        const message = JSON.parse(ev.data) as WebSocketMessage;
        onMessageRef.current?.(message);
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };
    return () => {
      es.close();
      esRef.current = null;
      setIsConnected(false);
    };
  }, [sessionId, transport]);

  return { isConnected, sendMessage, transport };
}
