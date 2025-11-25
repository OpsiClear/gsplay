"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gsplay_launcher.api.routes import router, set_file_browser, set_instance_manager
from gsplay_launcher.config import LauncherConfig
from gsplay_launcher.services.file_browser import FileBrowserService
from gsplay_launcher.services.instance_manager import InstanceManager

logger = logging.getLogger(__name__)

# Global config (set before creating app)
_config: LauncherConfig | None = None


def set_config(config: LauncherConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    if _config is None:
        raise RuntimeError("Config not set before creating app")

    # Startup: initialize instance manager
    logger.info("Initializing instance manager...")
    manager = InstanceManager(_config)
    manager.initialize()
    set_instance_manager(manager)

    # Initialize file browser if configured
    if _config.browse_path and _config.browse_path.is_dir():
        logger.info("Initializing file browser with root: %s", _config.browse_path)
        browser = FileBrowserService(_config.browse_path)
        set_file_browser(browser)
    else:
        logger.info("File browser disabled (no browse_path configured)")

    logger.info("Launcher started on http://%s:%d", _config.host, _config.port)

    yield

    # Shutdown: nothing to clean up (processes survive)
    logger.info("Launcher shutting down")


# Embedded HTML dashboard (no build step required)
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GSPlay Launcher</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0a0a0a; color: #d0d0d0; min-height: 100vh; font-size: 13px;
    }
    .top-bar {
      background: #111; border-bottom: 1px solid #222; padding: 8px 16px;
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }
    .top-bar h1 { font-size: 15px; font-weight: 600; white-space: nowrap; }
    .gpu-bar {
      display: flex; gap: 8px; flex: 1; justify-content: flex-end; flex-wrap: wrap;
      align-items: center;
    }
    .gpu-chip {
      background: #1a1a1a; border: 1px solid #333; border-radius: 4px;
      padding: 4px 8px; font-size: 11px; display: inline-flex; gap: 6px; align-items: center;
      flex-shrink: 0;
    }
    .gpu-chip .idx {
      background: #333; color: #aaa; padding: 1px 5px; border-radius: 3px;
      font-weight: 600; font-size: 10px; min-width: 18px; text-align: center;
    }
    .gpu-chip .name { color: #888; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .gpu-chip .stat { display: flex; gap: 2px; align-items: center; }
    .gpu-chip .stat-label { color: #555; font-size: 9px; }
    .gpu-chip .mem { color: #6b9; }
    .gpu-chip .util { color: #9b6; }
    .gpu-chip .temp { color: #b96; }
    .gpu-chip .temp.hot { color: #f66; }
    .gpu-chip .bar {
      width: 40px; height: 6px; background: #333; border-radius: 2px; overflow: hidden;
    }
    .gpu-chip .bar-fill { height: 100%; transition: width 0.3s; }
    .gpu-chip .bar-fill.mem { background: #6b9; }
    .gpu-chip .bar-fill.util { background: #9b6; }
    .container { padding: 12px 16px; max-width: 1400px; margin: 0 auto; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
    .btn {
      padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer;
      font-size: 12px; font-weight: 500; transition: background 0.15s;
    }
    .btn-sm { padding: 4px 8px; font-size: 11px; }
    .btn-primary { background: #2563eb; color: white; }
    .btn-primary:hover { background: #1d4ed8; }
    .btn-secondary { background: #333; color: #d0d0d0; }
    .btn-secondary:hover { background: #444; }
    .btn-danger { background: #b91c1c; color: white; }
    .btn-danger:hover { background: #991b1b; }
    .btn-success { background: #15803d; color: white; }
    .btn-success:hover { background: #166534; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .error-banner {
      background: #2d0000; color: #f88; padding: 8px 12px; border-radius: 4px;
      margin-bottom: 12px; display: none; font-size: 12px;
    }
    .form-panel {
      background: #141414; border: 1px solid #222; border-radius: 6px;
      padding: 12px; margin-bottom: 12px; display: none;
    }
    .form-panel.show { display: block; }
    .form-row { display: flex; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
    .form-group { flex: 1; min-width: 140px; }
    .form-group.wide { flex: 2; min-width: 280px; }
    .form-group label { display: block; margin-bottom: 4px; font-size: 11px; color: #888; }
    .form-group input {
      width: 100%; padding: 6px 8px; background: #1a1a1a;
      border: 1px solid #333; border-radius: 3px; color: #d0d0d0; font-size: 12px;
    }
    .form-group input:focus { border-color: #2563eb; outline: none; }
    .checkbox-group { display: flex; gap: 16px; align-items: center; }
    .checkbox-group label {
      display: flex; align-items: center; gap: 4px; cursor: pointer;
      font-size: 11px; color: #999;
    }
    .form-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; background: #111; border-radius: 6px; overflow: hidden; }
    th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #1a1a1a; }
    th { background: #0d0d0d; font-size: 11px; color: #666; font-weight: 500; text-transform: uppercase; }
    td { font-size: 12px; }
    tr:hover { background: #161616; }
    .name-cell { font-weight: 500; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .path-cell { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #888; font-size: 11px; }
    .badge {
      display: inline-block; padding: 2px 6px; border-radius: 3px;
      font-size: 10px; font-weight: 600; text-transform: uppercase;
    }
    .badge-running { background: #052; color: #0f0; }
    .badge-starting, .badge-stopping { background: #330; color: #ff0; }
    .badge-stopped, .badge-pending { background: #222; color: #888; }
    .badge-failed { background: #300; color: #f66; }
    .badge-orphaned { background: #320; color: #f90; }
    .actions-cell { white-space: nowrap; }
    .actions-cell .btn { margin-left: 4px; }
    .empty { text-align: center; padding: 32px; color: #666; background: #111; border-radius: 6px; }
    .meta { color: #666; font-size: 11px; }
    .driver-info { font-size: 10px; color: #555; margin-left: auto; }
    /* Browser panel styles */
    .browser-panel {
      background: #141414; border: 1px solid #222; border-radius: 6px;
      padding: 12px; margin-bottom: 12px; display: none;
    }
    .browser-panel.show { display: block; }
    .browser-header {
      display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
      padding-bottom: 8px; border-bottom: 1px solid #222;
    }
    .browser-header h3 { font-size: 13px; font-weight: 600; }
    .breadcrumbs {
      display: flex; gap: 4px; align-items: center; flex: 1;
      overflow-x: auto; white-space: nowrap;
    }
    .breadcrumb {
      padding: 3px 8px; background: #222; border-radius: 3px;
      font-size: 11px; color: #aaa; cursor: pointer; border: none;
      transition: background 0.15s;
    }
    .breadcrumb:hover { background: #333; color: #fff; }
    .breadcrumb-sep { color: #444; font-size: 10px; }
    .browser-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 8px; max-height: 320px; overflow-y: auto;
    }
    .folder-card {
      background: #1a1a1a; border: 1px solid #333; border-radius: 4px;
      padding: 10px; cursor: pointer; transition: all 0.15s;
    }
    .folder-card:hover { border-color: #555; background: #222; }
    .folder-card.ply-folder { border-color: #2a4; }
    .folder-card.ply-folder:hover { border-color: #3c6; }
    .folder-card .folder-name {
      font-weight: 500; font-size: 12px; margin-bottom: 4px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .folder-card .folder-meta { font-size: 10px; color: #666; margin-bottom: 6px; }
    .folder-card .ply-badge {
      display: inline-block; background: #143; color: #4b8; font-size: 9px;
      padding: 2px 5px; border-radius: 2px; font-weight: 600;
    }
    .folder-card .folder-actions {
      display: flex; gap: 4px; margin-top: 8px;
    }
    .folder-card .folder-actions .btn { flex: 1; font-size: 10px; padding: 4px 6px; }
    .browser-empty {
      text-align: center; padding: 24px; color: #666; font-size: 12px;
    }
    .browser-root-info {
      font-size: 10px; color: #555; margin-left: auto;
    }
  </style>
</head>
<body>
  <div class="top-bar">
    <h1>GSPlay Launcher</h1>
    <div class="gpu-bar" id="gpuBar">
      <span class="meta">Loading GPU info...</span>
    </div>
    <div class="driver-info" id="driverInfo"></div>
  </div>

  <div class="container">
    <div class="error-banner" id="errorBanner">
      <span id="errorText"></span>
      <button onclick="hideError()" style="float:right;background:none;border:none;color:#f88;cursor:pointer;">x</button>
    </div>

    <div class="toolbar">
      <button class="btn btn-primary" id="toggleFormBtn">+ New</button>
      <span class="meta" id="instanceCount"></span>
    </div>

    <div class="form-panel" id="formPanel">
      <form id="createForm">
        <div class="form-row">
          <div class="form-group wide">
            <label>Config Path (PLY folder or JSON) *</label>
            <input type="text" id="configPath" placeholder="D:/path/to/ply/folder" required>
          </div>
          <div class="form-group">
            <label>Name</label>
            <input type="text" id="name" placeholder="My GSPlay">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Port (auto)</label>
            <input type="number" id="port" placeholder="6020" min="1024" max="65535">
          </div>
          <div class="form-group">
            <label>GPU</label>
            <input type="number" id="gpu" placeholder="0" min="0">
          </div>
          <div class="form-group">
            <label>Cache</label>
            <input type="number" id="cacheSize" value="100" min="1" max="1000">
          </div>
          <div class="form-group checkbox-group" style="align-self:flex-end;">
            <label><input type="checkbox" id="viewOnly"> View Only</label>
            <label><input type="checkbox" id="compact" checked> Compact</label>
          </div>
          <div class="form-actions" style="align-self:flex-end;">
            <button type="button" class="btn btn-secondary" onclick="toggleForm()">Cancel</button>
            <button type="submit" class="btn btn-primary" id="submitBtn">Launch</button>
          </div>
        </div>
      </form>
    </div>

    <div class="browser-panel" id="browserPanel">
      <div class="browser-header">
        <h3>File Browser</h3>
        <div class="breadcrumbs" id="breadcrumbs"></div>
        <span class="browser-root-info" id="browserRootInfo"></span>
      </div>
      <div class="browser-grid" id="browserGrid"></div>
    </div>

    <div id="instancesContainer"></div>
  </div>

  <script>
    const API = '/api';
    let instances = [];
    let gpuInfo = null;
    let browseConfig = null;
    let currentBrowsePath = '';

    // Browser functions
    async function fetchBrowseConfig() {
      try {
        const res = await fetch(`${API}/browse/config`);
        browseConfig = await res.json();
        if (browseConfig.enabled) {
          document.getElementById('browserPanel').classList.add('show');
          document.getElementById('browserRootInfo').textContent = browseConfig.root_path;
          await browsePath('');
        }
      } catch (e) {
        console.error('Failed to fetch browse config:', e);
      }
    }

    async function browsePath(path) {
      if (!browseConfig || !browseConfig.enabled) return;
      currentBrowsePath = path;
      try {
        const res = await fetch(`${API}/browse?path=${encodeURIComponent(path)}`);
        if (!res.ok) {
          const err = await res.json();
          showError(err.detail || 'Failed to browse');
          return;
        }
        const data = await res.json();
        renderBreadcrumbs(data.breadcrumbs);
        renderBrowserGrid(data.entries);
      } catch (e) {
        showError('Failed to browse: ' + e.message);
      }
    }

    function renderBreadcrumbs(breadcrumbs) {
      const container = document.getElementById('breadcrumbs');
      container.innerHTML = breadcrumbs.map((b, i) => {
        const sep = i < breadcrumbs.length - 1 ? '<span class="breadcrumb-sep">/</span>' : '';
        return `<button class="breadcrumb" onclick="browsePath('${escapeHtml(b.path)}')">${escapeHtml(b.name)}</button>${sep}`;
      }).join('');
    }

    function renderBrowserGrid(entries) {
      const grid = document.getElementById('browserGrid');
      const folders = entries.filter(e => e.is_directory);
      if (folders.length === 0) {
        grid.innerHTML = '<div class="browser-empty">No folders found</div>';
        return;
      }
      grid.innerHTML = folders.map(e => {
        const plyClass = e.is_ply_folder ? 'ply-folder' : '';
        const plyBadge = e.is_ply_folder
          ? `<span class="ply-badge">${e.ply_count} PLY (${formatSize(e.total_size_mb)})</span>`
          : '';
        const modified = e.modified_at ? formatDate(e.modified_at) : '';
        const actions = e.is_ply_folder
          ? `<div class="folder-actions">
               <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); selectPath('${escapeHtml(e.path)}')">Select</button>
               <button class="btn btn-sm btn-success" onclick="event.stopPropagation(); launchFromBrowser('${escapeHtml(e.path)}', '${escapeHtml(e.name)}')">Launch</button>
             </div>`
          : '';
        return `<div class="folder-card ${plyClass}" onclick="browsePath('${escapeHtml(e.path)}')" title="${escapeHtml(e.path)}">
          <div class="folder-name">${escapeHtml(e.name)}</div>
          <div class="folder-meta">${modified} ${plyBadge}</div>
          ${actions}
        </div>`;
      }).join('');
    }

    function formatSize(mb) {
      if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
      return mb.toFixed(1) + ' MB';
    }

    function formatDate(isoString) {
      try {
        const d = new Date(isoString);
        return d.toLocaleDateString();
      } catch (e) {
        return '';
      }
    }

    function selectPath(path) {
      // Fill the config path in the create form and show the form
      document.getElementById('configPath').value = browseConfig.root_path + '/' + path;
      if (!document.getElementById('formPanel').classList.contains('show')) {
        toggleForm();
      }
    }

    async function launchFromBrowser(path, name) {
      try {
        const data = {
          path: path,
          name: name || 'GSPlay-' + path.split('/').pop(),
          cache_size: 100,
          compact: true,
        };
        const res = await fetch(`${API}/browse/launch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Failed to launch');
        }
        await fetchInstances();
      } catch (e) {
        showError(e.message);
      }
    }

    async function fetchGpuInfo() {
      try {
        const res = await fetch(`${API}/gpu`);
        if (res.ok) {
          gpuInfo = await res.json();
          renderGpuBar();
        }
      } catch (e) {
        document.getElementById('gpuBar').innerHTML = '<span class="meta">GPU info unavailable</span>';
      }
    }

    function renderGpuBar() {
      if (!gpuInfo || !gpuInfo.gpus.length) {
        document.getElementById('gpuBar').innerHTML = '<span class="meta">No GPUs detected</span>';
        return;
      }
      const chips = gpuInfo.gpus.map(g => {
        const memPct = Math.round(g.memory_used / g.memory_total * 100);
        const tempClass = g.temperature > 80 ? 'hot' : '';
        // Compact display for multi-GPU: show index badge, bars, and key stats
        return `<div class="gpu-chip" title="GPU ${g.index}: ${g.name}">
          <span class="idx">${g.index}</span>
          <span class="name">${g.name}</span>
          <span class="stat">
            <span class="stat-label">M</span>
            <span class="bar"><span class="bar-fill mem" style="width:${memPct}%"></span></span>
            <span class="mem">${memPct}%</span>
          </span>
          <span class="stat">
            <span class="stat-label">U</span>
            <span class="bar"><span class="bar-fill util" style="width:${g.utilization}%"></span></span>
            <span class="util">${g.utilization}%</span>
          </span>
          <span class="temp ${tempClass}">${g.temperature}C</span>
        </div>`;
      }).join('');
      document.getElementById('gpuBar').innerHTML = chips;
      document.getElementById('driverInfo').textContent =
        `${gpuInfo.gpus.length} GPU${gpuInfo.gpus.length > 1 ? 's' : ''} | Driver ${gpuInfo.driver_version} | CUDA ${gpuInfo.cuda_version}`;
    }

    async function fetchInstances() {
      try {
        const res = await fetch(`${API}/instances`);
        const data = await res.json();
        instances = data.instances;
        renderInstances();
      } catch (e) {
        showError('Failed to fetch instances: ' + e.message);
      }
    }

    function renderInstances() {
      const container = document.getElementById('instancesContainer');
      const active = instances.filter(i => isActive(i.status)).length;
      document.getElementById('instanceCount').textContent = `${active} active / ${instances.length} total`;

      if (instances.length === 0) {
        container.innerHTML = '<div class="empty">No GSPlay instances. Click "+ New" to create one.</div>';
        return;
      }

      container.innerHTML = `<table>
        <thead><tr><th>Name</th><th>Status</th><th>Port</th><th>GPU</th><th>Config</th><th>PID</th><th>Actions</th></tr></thead>
        <tbody>${instances.map(i => `<tr>
          <td class="name-cell" title="${escapeHtml(i.name)}">${escapeHtml(i.name)}</td>
          <td><span class="badge badge-${i.status}">${i.status}</span></td>
          <td>${i.port}</td>
          <td>${i.gpu !== null ? i.gpu : '-'}</td>
          <td class="path-cell" title="${escapeHtml(i.config_path)}">${escapeHtml(i.config_path)}</td>
          <td class="meta">${i.pid || '-'}</td>
          <td class="actions-cell">
            ${isActive(i.status) ? `
              <button class="btn btn-sm btn-success" onclick="openGSPlay('${i.url}')">Open</button>
              <button class="btn btn-sm btn-danger" onclick="stopInstance('${i.id}')">Stop</button>
            ` : `
              <button class="btn btn-sm btn-secondary" onclick="deleteInstance('${i.id}', '${escapeHtml(i.name)}')">Delete</button>
            `}
          </td>
        </tr>`).join('')}</tbody>
      </table>`;
    }

    function isActive(status) {
      return status === 'running' || status === 'starting' || status === 'orphaned';
    }

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function toggleForm() {
      const panel = document.getElementById('formPanel');
      panel.classList.toggle('show');
      document.getElementById('toggleFormBtn').textContent = panel.classList.contains('show') ? 'Cancel' : '+ New';
    }

    document.getElementById('createForm').onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      btn.disabled = true;
      btn.textContent = 'Launching...';

      const data = {
        config_path: document.getElementById('configPath').value.trim(),
        name: document.getElementById('name').value.trim() || undefined,
        port: document.getElementById('port').value ? parseInt(document.getElementById('port').value) : undefined,
        gpu: document.getElementById('gpu').value ? parseInt(document.getElementById('gpu').value) : undefined,
        cache_size: parseInt(document.getElementById('cacheSize').value) || 100,
        view_only: document.getElementById('viewOnly').checked,
        compact: document.getElementById('compact').checked,
      };

      try {
        const res = await fetch(`${API}/instances`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Failed to create instance');
        }
        document.getElementById('createForm').reset();
        document.getElementById('cacheSize').value = '100';
        document.getElementById('compact').checked = true;
        toggleForm();
        await fetchInstances();
      } catch (e) {
        showError(e.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Launch';
      }
    };

    async function stopInstance(id) {
      try {
        const res = await fetch(`${API}/instances/${id}/stop`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to stop instance');
        await fetchInstances();
      } catch (e) {
        showError(e.message);
      }
    }

    async function deleteInstance(id, name) {
      if (!confirm(`Delete "${name}"?`)) return;
      try {
        const res = await fetch(`${API}/instances/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to delete instance');
        await fetchInstances();
      } catch (e) {
        showError(e.message);
      }
    }

    function openGSPlay(url) { window.open(url, '_blank'); }
    function showError(msg) {
      document.getElementById('errorText').textContent = msg;
      document.getElementById('errorBanner').style.display = 'block';
    }
    function hideError() { document.getElementById('errorBanner').style.display = 'none'; }

    document.getElementById('toggleFormBtn').onclick = toggleForm;

    // Initial load and polling
    fetchGpuInfo();
    fetchInstances();
    fetchBrowseConfig();
    setInterval(fetchGpuInfo, 3000);
    setInterval(fetchInstances, 5000);
  </script>
</body>
</html>
"""


def create_app(config: LauncherConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Parameters
    ----------
    config : LauncherConfig | None
        Launcher configuration. If None, uses global config.

    Returns
    -------
    FastAPI
        Configured application.
    """
    if config is not None:
        set_config(config)

    app = FastAPI(
        title="GSPlay Launcher",
        description="Launch and manage Gaussian Splatting GSPlay instances",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve embedded HTML dashboard at root
    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return DASHBOARD_HTML

    # Check for built frontend and serve static assets
    static_dir = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info("Serving static assets from %s", static_dir)
    else:
        logger.info("Serving embedded dashboard (no build)")

    return app
