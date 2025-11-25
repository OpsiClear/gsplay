# Viewer Launcher

FastAPI application for managing Gaussian Splatting viewer instances.

## Features

- Launch viewer instances with configurations via REST API
- Stop and list running instances
- Auto-assign ports with optional user override
- Persist state to survive launcher restarts
- SolidJS dashboard with Deno build system

## Installation

```bash
cd launcher
uv sync
```

## Usage

### Start the Launcher

```bash
# Start with default settings
uv run src/main.py

# Custom port
uv run src/main.py --port 8080

# With custom viewer script path
uv run src/main.py --viewer-script ../src/viewer/core/main.py
```

### Access the Dashboard

Open http://localhost:8000 in your browser.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| GET | /api/instances | List all instances |
| GET | /api/instances/{id} | Get instance by ID |
| POST | /api/instances | Create and start instance |
| POST | /api/instances/{id}/stop | Stop running instance |
| DELETE | /api/instances/{id} | Delete instance |
| GET | /api/ports/next | Get next available port |

### Example: Create Instance via curl

```bash
curl -X POST http://localhost:8000/api/instances \
  -H "Content-Type: application/json" \
  -d '{"config_path": "./my_ply_folder", "name": "Demo"}'
```

## Frontend Development

The frontend uses SolidJS with Deno.

```bash
cd frontend

# Install Deno if not already installed
# https://deno.land/manual/getting_started/installation

# Start dev server
deno task dev

# Build for production
deno task build
```

The dev server proxies API requests to http://localhost:8000.

## Architecture

```
launcher/
|-- src/
|   |-- main.py              # CLI entry point
|   |-- config.py            # LauncherConfig
|   |-- models.py            # ViewerInstance, InstanceStatus
|   |-- api/
|   |   |-- app.py           # FastAPI factory
|   |   |-- routes.py        # CRUD endpoints
|   |   |-- schemas.py       # Pydantic models
|   |-- services/
|       |-- instance_manager.py   # Core orchestration
|       |-- process_manager.py    # Subprocess management
|       |-- port_allocator.py     # Port allocation
|       |-- state_persistence.py  # JSON state file
|-- frontend/
|   |-- src/
|       |-- App.tsx          # Main component
|       |-- components/      # UI components
|       |-- stores/          # State management
|-- data/
    |-- instances.json       # Persisted state
```

## Recovery After Launcher Restart

When the launcher restarts, it:

1. Loads the persisted state from `data/instances.json`
2. Checks if each "running" instance's process is still alive
3. Marks alive processes as "orphaned" (can still be stopped)
4. Marks dead processes as "stopped"

This ensures viewer instances survive launcher crashes.
