# Mini-Devin API Documentation

This document describes the REST and WebSocket APIs provided by Mini-Devin's FastAPI backend.

## Base URL

```
http://localhost:8000
```

## Authentication

Mini-Devin supports two authentication methods:

### JWT Token Authentication

Include the JWT token in the Authorization header:

```
Authorization: Bearer <jwt_token>
```

### API Key Authentication

Include the API key in the X-API-Key header:

```
X-API-Key: <api_key>
```

## Endpoints

### Health Check

#### GET /health

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Authentication

#### POST /api/auth/register

Register a new user.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### POST /api/auth/login

Authenticate and receive a JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /api/auth/api-keys

Create a new API key.

**Request Body:**
```json
{
  "name": "My API Key",
  "expires_at": "2025-01-01T00:00:00Z"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "My API Key",
  "key": "md_...",
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2025-01-01T00:00:00Z"
}
```

---

### Models and Providers

#### GET /api/models

List available LLM models.

**Query Parameters:**
- `provider` (optional): Filter by provider (openai, anthropic, ollama, azure)

**Response:**
```json
{
  "models": [
    {
      "id": "gpt-4o",
      "name": "GPT-4o",
      "provider": "openai",
      "context_window": 128000,
      "supports_vision": true,
      "supports_function_calling": true
    },
    {
      "id": "claude-3-5-sonnet-20241022",
      "name": "Claude 3.5 Sonnet",
      "provider": "anthropic",
      "context_window": 200000,
      "supports_vision": true,
      "supports_function_calling": true
    }
  ]
}
```

#### GET /api/providers

List available LLM providers.

**Response:**
```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "configured": true,
      "models_count": 5
    },
    {
      "id": "anthropic",
      "name": "Anthropic",
      "configured": true,
      "models_count": 3
    },
    {
      "id": "ollama",
      "name": "Ollama",
      "configured": false,
      "models_count": 0
    }
  ],
  "default_model": "gpt-4o"
}
```

---

### Sessions

#### GET /api/sessions

List all sessions.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "Bug Fix Session",
    "status": "active",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T01:00:00Z",
    "task_count": 3
  }
]
```

#### POST /api/sessions

Create a new session.

**Request Body:**
```json
{
  "name": "New Session",
  "working_directory": "/path/to/project"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "New Session",
  "status": "active",
  "working_directory": "/path/to/project",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### GET /api/sessions/{session_id}

Get session details.

**Response:**
```json
{
  "id": "uuid",
  "name": "Bug Fix Session",
  "status": "active",
  "working_directory": "/path/to/project",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T01:00:00Z",
  "tasks": [
    {
      "id": "task-uuid",
      "description": "Fix failing test",
      "status": "completed"
    }
  ]
}
```

#### DELETE /api/sessions/{session_id}

Delete a session.

**Response:** 204 No Content

---

### Tasks

#### GET /api/sessions/{session_id}/tasks

List tasks in a session.

**Response:**
```json
[
  {
    "id": "uuid",
    "description": "Fix the failing test in tests/test_api.py",
    "status": "completed",
    "created_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:30:00Z",
    "iterations": 5
  }
]
```

#### POST /api/sessions/{session_id}/tasks

Create and start a new task.

**Request Body:**
```json
{
  "description": "Fix the failing test in tests/test_api.py",
  "model": "gpt-4o",
  "run_mode": "offline"
}
```

**Response:**
```json
{
  "id": "uuid",
  "description": "Fix the failing test in tests/test_api.py",
  "status": "running",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### GET /api/tasks/{task_id}

Get task details.

**Response:**
```json
{
  "id": "uuid",
  "description": "Fix the failing test",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:30:00Z",
  "iterations": 5,
  "plan": {
    "steps": [
      {
        "id": "step-1",
        "description": "Read the failing test",
        "status": "completed"
      }
    ]
  },
  "result": {
    "success": true,
    "summary": "Fixed the test by updating the expected value"
  }
}
```

#### POST /api/tasks/{task_id}/stop

Stop a running task.

**Response:**
```json
{
  "id": "uuid",
  "status": "stopped"
}
```

---

### Artifacts

#### GET /api/tasks/{task_id}/artifacts

List artifacts for a task.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "plan.json",
    "type": "plan",
    "size": 1024,
    "created_at": "2024-01-01T00:00:00Z"
  },
  {
    "id": "uuid",
    "name": "diff.patch",
    "type": "diff",
    "size": 2048,
    "created_at": "2024-01-01T00:30:00Z"
  }
]
```

#### GET /api/tasks/{task_id}/artifacts/{artifact_id}

Download an artifact.

**Response:** File content with appropriate Content-Type header.

---

## WebSocket API

### Task Streaming

Connect to receive real-time updates for a task.

**Endpoint:**
```
ws://localhost:8000/ws/tasks/{task_id}
```

**Authentication:**
Include the JWT token as a query parameter:
```
ws://localhost:8000/ws/tasks/{task_id}?token=<jwt_token>
```

### Message Types

#### Server Messages

**Token Stream:**
```json
{
  "type": "token",
  "content": "Let me analyze",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Tool Call:**
```json
{
  "type": "tool_call",
  "tool": "terminal",
  "input": {
    "command": "pytest tests/test_api.py -v"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Tool Result:**
```json
{
  "type": "tool_result",
  "tool": "terminal",
  "output": "1 passed, 0 failed",
  "success": true,
  "timestamp": "2024-01-01T00:00:01Z"
}
```

**Plan Update:**
```json
{
  "type": "plan_update",
  "plan": {
    "steps": [
      {
        "id": "step-1",
        "description": "Read the failing test",
        "status": "completed"
      },
      {
        "id": "step-2",
        "description": "Fix the assertion",
        "status": "in_progress"
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:02Z"
}
```

**Status Update:**
```json
{
  "type": "status",
  "status": "completed",
  "message": "Task completed successfully",
  "timestamp": "2024-01-01T00:30:00Z"
}
```

**Error:**
```json
{
  "type": "error",
  "error": "Maximum iterations reached",
  "code": "MAX_ITERATIONS_REACHED",
  "timestamp": "2024-01-01T00:30:00Z"
}
```

#### Client Messages

**Stop Task:**
```json
{
  "type": "stop"
}
```

**Provide Input:**
```json
{
  "type": "input",
  "content": "Yes, proceed with the changes"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Invalid request body |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Authenticated requests:** 100 requests per minute
- **Unauthenticated requests:** 10 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

---

## CORS

The API allows cross-origin requests from:
- `http://localhost:5173` (development)
- `http://localhost:3000` (alternative development)

For production, configure allowed origins via the `CORS_ORIGINS` environment variable.
