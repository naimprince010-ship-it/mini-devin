import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketMessage } from '../types';
import { getApiWsUrl } from '../config/apiBase';

interface UseWebSocketOptions {
  sessionId?: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { sessionId } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Store latest callbacks in refs to avoid stale closures
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const connect = useCallback(() => {
    const wsUrl = getApiWsUrl(sessionId);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      optionsRef.current.onConnect?.();
    };

    ws.onclose = () => {
      setIsConnected(false);
      optionsRef.current.onDisconnect?.();

      // Only attempt to reconnect if it wasn't an intentional disconnect
      if (wsRef.current === ws) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 3000);
      }
    };

    ws.onerror = (error) => {
      optionsRef.current.onError?.(error);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        setMessages((prev) => [...prev, message]);
        optionsRef.current.onMessage?.(message);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    wsRef.current = ws;
  }, [sessionId]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: string): boolean => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
      return true;
    }
    return false;
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return {
    isConnected,
    messages,
    sendMessage,
    clearMessages,
    connect,
    disconnect,
  };
}
