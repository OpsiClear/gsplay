# GSPlay Launcher

FastAPI application for managing Gaussian Splatting viewer instances with a SolidJS dashboard.

**Version:** 0.2.0

## Features

- Launch and manage multiple GSPlay instances via REST API
- Real-time GPU/CPU monitoring
- File browser for PLY folders
- WebSocket proxy for viewer access through launcher
- Stream proxy for view-only mode
- Launch history with quick relaunch
- Orphaned process detection and cleanup

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Deno](https://deno.land/) 1.40+ (for frontend)

### Installation

```bash
cd launcher

# Install Python dependencies
uv sync

# Build frontend (required for production)
cd frontend
deno task build
cd ..
```

### Run the Launcher

```bash
# Start with file browser enabled
uv run gsplay_launcher --browse-path /path/to/ply/folders

# Custom port
uv run gsplay_launcher --port 8080

# With external URL (for reverse proxy setups)
uv run gsplay_launcher --external-url https://viewer.example.com
```

### Access the Dashboard

Open http://localhost:8000 in your browser.

## Frontend Development

The frontend uses SolidJS with Deno as the build system.

### Install Deno

```bash
# Linux/macOS
curl -fsSL https://deno.land/install.sh | sh

# Windows (PowerShell)
irm https://deno.land/install.ps1 | iex

# Or via package managers
brew install deno        # macOS
scoop install deno       # Windows
```

### Development Server

```bash
cd frontend

# Start dev server with hot reload (proxies to backend on :8000)
deno task dev

# The dev server runs on http://localhost:5173
```

### Production Build

```bash
cd frontend

# Build optimized bundle
deno task build

# Output goes to frontend/dist/
```

### Frontend Structure

```
frontend/
├── deno.json           # Deno config and tasks
├── vite.config.ts      # Vite bundler config
├── index.html          # Entry HTML
└── src/
    ├── App.tsx         # Root component
    ├── index.css       # Global styles
    ├── version.ts      # App version
    ├── api/
    │   └── client.ts   # API client
    ├── components/
    │   ├── Header.tsx
    │   ├── Instances.tsx
    │   ├── FileBrowser.tsx
    │   ├── LaunchSettings.tsx
    │   ├── LaunchHistory.tsx
    │   ├── Console.tsx
    │   └── StreamPreview.tsx
    └── stores/         # SolidJS state management
        ├── app.ts      # Orchestration + re-exports
        ├── instances.ts
        ├── gpu.ts
        ├── system.ts
        ├── browse.ts
        ├── history.ts
        ├── console.ts
        ├── stream.ts
        └── error.ts
```

## API Endpoints

### Health & System

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check with instance counts |
| GET | /api/gpu | GPU information (nvidia-smi) |
| GET | /api/system | CPU/memory stats |
| GET | /api/system/msvc | MSVC compiler status (Windows) |

### Instance Management

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/instances | List all instances |
| GET | /api/instances/{id} | Get instance by ID |
| POST | /api/instances | Create and start instance |
| POST | /api/instances/{id}/stop | Stop running instance |
| DELETE | /api/instances/{id} | Delete instance |
| GET | /api/ports/next | Get next available port |

### File Browser

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/browse/config | Get browser configuration |
| GET | /api/browse?path= | List directory contents |
| POST | /api/browse/launch | Launch instance from path |

### Logs & Control

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/instances/{id}/logs | Get instance logs |
| GET | /api/instances/{id}/logs/stream | Stream logs (SSE) |
| POST | /api/instances/{id}/control/{cmd} | Send control command |

### Cleanup

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/cleanup | Scan for GSPlay processes |
| POST | /api/cleanup/stop | Stop discovered processes |

### Proxy Routes

| Path | Description |
|------|-------------|
| /v/{viewer_id}/ | Viewer proxy (WebSocket + HTTP) |
| /s/{token}/ | Stream proxy for view-only mode |

## Configuration Options

```bash
uv run gsplay_launcher --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Launcher port |
| `--browse-path` | None | Root path for file browser |
| `--external-url` | None | External URL for proxied access |
| `--network-url` | None | Network URL for LAN access |
| `--view-only` | False | Force view-only mode globally |
| `--gsplay-port-start` | 8100 | Start of viewer port range |
| `--gsplay-port-end` | 8199 | End of viewer port range |
| `--history-limit` | 10 | Max launch history entries |

## Architecture

```
launcher/
├── gsplay_launcher/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── config.py            # LauncherConfig
│   ├── models.py            # ViewerInstance, InstanceStatus
│   ├── api/
│   │   ├── app.py           # FastAPI factory
│   │   ├── dependencies.py  # Dependency injection
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── routes/          # Modular route handlers
│   │       ├── instances.py
│   │       ├── gpu.py
│   │       ├── browse.py
│   │       ├── logs.py
│   │       ├── cleanup.py
│   │       ├── control.py
│   │       └── proxy.py
│   ├── services/
│   │   ├── instance_manager.py   # Core orchestration
│   │   ├── process_manager.py    # Subprocess management
│   │   ├── port_allocator.py     # Port allocation
│   │   ├── state_persistence.py  # JSON state file
│   │   ├── file_browser.py       # Directory browsing
│   │   ├── gpu_info.py           # GPU monitoring
│   │   ├── log_service.py        # Log reading/streaming
│   │   └── websocket_proxy.py    # WebSocket proxying
│   └── cli/
│       └── cleanup.py        # Process discovery
├── frontend/                 # SolidJS dashboard
└── data/
    └── instances.json        # Persisted state
```

## Recovery After Restart

When the launcher restarts, it:

1. Loads persisted state from `data/instances.json`
2. Checks if each "running" instance's process is still alive
3. Marks alive processes as "orphaned" (can still be stopped)
4. Marks dead processes as "stopped"

This ensures viewer instances survive launcher crashes.

## Example Usage

### Create Instance via curl

```bash
curl -X POST http://localhost:8000/api/instances \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "./my_ply_folder",
    "name": "Demo",
    "gpu": 0,
    "compact": true,
    "view_only": true
  }'
```

### Stop Instance

```bash
curl -X POST http://localhost:8000/api/instances/{instance_id}/stop
```

### Browse Files

```bash
curl "http://localhost:8000/api/browse?path=subfolder"
```

## Development

### Run Backend in Development

```bash
cd launcher
uv run gsplay_launcher --browse-path ../test_data
```

### Run Frontend in Development

```bash
cd launcher/frontend
deno task dev
# Open http://localhost:5173
```

The Vite dev server proxies `/api/*` and `/v/*` requests to the backend on port 8000.

### Code Style

- Backend: Python with type hints, follows project conventions
- Frontend: TypeScript with SolidJS, modular store pattern
