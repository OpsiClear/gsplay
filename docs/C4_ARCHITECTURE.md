# C4 Architecture Model

This document describes the GSPlay system architecture using the [C4 model](https://c4model.com/).

## Level 1: System Context

Shows how GSPlay fits into the world around it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │   End User   │         │    Admin     │         │  Developer   │
    │  (Researcher │         │   (System    │         │  (Plugin     │
    │   VFX Artist)│         │  Operator)   │         │   Author)    │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           │ Views scenes           │ Manages instances      │ Extends system
           │ Adjusts params         │ Configures GPUs        │ Creates sources
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │                         GSPlay System                           │
    │                                                                 │
    │   Real-time viewer for dynamic 4D Gaussian Splatting scenes    │
    │                                                                 │
    └───────────────────────────────┬─────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │    File      │         │    NVIDIA    │         │     Web      │
    │   Storage    │         │     GPU      │         │   Browser    │
    │              │         │              │         │              │
    │ Local/S3/GCS │         │  CUDA 12.8   │         │ Chrome/Edge  │
    └──────────────┘         └──────────────┘         └──────────────┘
```

### Actors

| Actor | Description |
|-------|-------------|
| **End User** | Researchers, VFX artists viewing and manipulating 4D Gaussian scenes |
| **Admin** | System operators managing viewer instances and GPU allocation |
| **Developer** | Plugin authors extending GSPlay with custom data sources |

### External Systems

| System | Purpose |
|--------|---------|
| **File Storage** | PLY file storage (local, S3, GCS, Azure) |
| **NVIDIA GPU** | CUDA 12.8 for GPU-accelerated rendering |
| **Web Browser** | Primary UI interaction via Viser framework |

---

## Level 2: Container Diagram

Shows the high-level technology choices and how containers communicate.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONTAINER DIAGRAM                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Web Browser   │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │ HTTP                   │ WebSocket              │ HTTP
              ▼                        ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│                         │  │                     │  │                     │
│   GSPlay Launcher       │  │   GSPlay Viewer     │  │  Remote Control     │
│   [FastAPI + SolidJS]   │  │   [Viser + gsplat]  │  │  API [HTTP]         │
│                         │  │                     │  │                     │
│   Port: 8000            │  │   Port: N           │  │   Port: N+2         │
│                         │  │                     │  │                     │
│   - Instance mgmt       │  │   - 3D rendering    │  │   - Rotation ctrl   │
│   - File browser        │  │   - Camera control  │  │   - Playback ctrl   │
│   - GPU monitoring      │  │   - Color/opacity   │  │   - State queries   │
│   - WebSocket proxy     │  │   - Playback        │  │                     │
│                         │  │                     │  │                     │
└───────────┬─────────────┘  └──────────┬──────────┘  └─────────────────────┘
            │                           │
            │ subprocess                │ WebSocket
            │                           ▼
            │                ┌─────────────────────┐
            │                │                     │
            └───────────────▶│   GStream Server    │
                             │   [WebSocket]       │
                             │                     │
                             │   Port: N+1         │
                             │                     │
                             │   - JPEG streaming  │
                             │   - ~100-150ms      │
                             │                     │
                             └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │                     │
                             │   File Storage      │
                             │   [PLY Files]       │
                             │                     │
                             │   Local / Cloud     │
                             │                     │
                             └─────────────────────┘
```

### Containers

| Container | Technology | Port | Responsibility |
|-----------|------------|------|----------------|
| **GSPlay Launcher** | FastAPI + SolidJS | 8000 | Dashboard, instance management, file browser |
| **GSPlay Viewer** | Viser + gsplat + PyTorch | N | Core rendering, interactive UI |
| **GStream Server** | WebSocket + asyncio | N+1 | Low-latency binary JPEG streaming |
| **Remote Control API** | HTTP REST | N+2 | Programmatic control endpoints |

### Communication

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| Browser | Launcher | HTTP/WS | Dashboard UI, instance control |
| Browser | Viewer | WebSocket | Viser 3D interaction |
| Browser | GStream | WebSocket | Live stream preview |
| Launcher | Viewer | subprocess | Spawn/monitor instances |
| Launcher | GStream | WS Proxy | Route stream to dashboard |
| External | Control API | HTTP | Automation, scripting |

### Control API Endpoints

The Remote Control API (Port N+2) provides HTTP endpoints for automation:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rotate-cw` | POST | Rotate scene clockwise |
| `/rotate-ccw` | POST | Rotate scene counter-clockwise |
| `/center-scene` | POST | Center scene to origin |
| `/playback-state` | GET | Query playback status |
| `/play` | POST | Start playback |
| `/pause` | POST | Pause playback |
| `/state` | GET | Full viewer state |

---

## Level 3: Component Diagram

### GSPlay Viewer Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GSPlay Viewer                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  core/                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    main     │  │     app     │  │   Model     │  │   Render    │        │
│  │   (CLI)     │──▶│ (GSPlay)   │──▶│  Component  │  │  Component  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  rendering/   │    │  processing/  │    │     ui/       │
│               │    │               │    │               │
│ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│ │  Camera   │ │    │ │   Color   │ │    │ │  Layout   │ │
│ │Controller │ │    │ │Adjustments│ │    │ │           │ │
│ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │
│ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│ │ Renderer  │ │    │ │  Opacity  │ │    │ │Controller │ │
│ │ (gsplat)  │ │    │ │           │ │    │ │           │ │
│ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │
│ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│ │   JPEG    │ │    │ │  Volume   │ │    │ │  Layers   │ │
│ │  Encoder  │ │    │ │  Filter   │ │    │ │           │ │
│ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │              interaction/                    │
        │  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
        │  │  EventBus │  │ Playback  │  │Handlers │  │
        │  │           │  │Controller │  │         │  │
        │  └───────────┘  └───────────┘  └─────────┘  │
        └─────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  streaming/   │    │   control/    │    │    state/     │
│               │    │               │    │               │
│  WebSocket    │    │  HTTP API     │    │  EditManager  │
│  Server       │    │  Server       │    │  SceneBounds  │
└───────────────┘    └───────────────┘    └───────────────┘
```

### Launcher Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GSPlay Launcher                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  api/routes/                                                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │instances │ │  browse  │ │   gpu    │ │   logs   │ │  proxy   │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│  ┌─────────────────────────────────────┼───────────────────────────────┐   │
│  │  services/                          ▼                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Instance   │  │   Process    │  │    Port      │               │   │
│  │  │   Manager    │──▶│   Manager    │──▶│  Allocator   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │    File      │  │     GPU      │  │     Log      │               │   │
│  │  │   Browser    │  │     Info     │  │   Service    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │  ┌──────────────┐  ┌──────────────┐                                  │   │
│  │  │  WebSocket   │  │    State     │                                  │   │
│  │  │    Proxy     │  │ Persistence  │                                  │   │
│  │  └──────────────┘  └──────────────┘                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Frontend (SolidJS)                                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  components/                                                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│  │  │ Instances│ │  File    │ │  Launch  │ │ Console  │ │  Stream  │   │  │
│  │  │          │ │ Browser  │ │ Settings │ │          │ │ Preview  │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  stores/                                                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│  │  │instances │ │   gpu    │ │  browse  │ │  stream  │ │  history │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Domain & Infrastructure Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Domain Layer (Pure)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │   entities     │  │   interfaces   │  │     data       │                 │
│  │                │  │                │  │                │                 │
│  │  GSTensor      │  │ BaseGaussian   │  │ GaussianData   │                 │
│  │  GSData        │  │  Source        │  │  (CPU/GPU      │                 │
│  │  SceneBounds   │  │ DataSink       │  │   wrapper)     │                 │
│  │  GaussianLayer │  │  Protocol      │  │                │                 │
│  │  Composite     │  │ Interpolatable │  │ FormatInfo     │                 │
│  │   GSTensor     │  │  Source        │  │  (encoding     │                 │
│  │                │  │ HealthCheckable│  │   state)       │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │   lifecycle    │  │    time        │  │   services/    │                 │
│  │                │  │                │  │   transform    │                 │
│  │  PluginState   │  │ TimeDomain     │  │                │                 │
│  │   (enum)       │  │  - discrete    │  │ TransformSvc   │                 │
│  │  HealthStatus  │  │  - continuous  │  │ ColorAdjustSvc │                 │
│  │   (enum)       │  │  - interpolated│  │ Pure math ops  │                 │
│  │  LifecycleMixin│  │                │  │                │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                                   │
└─────────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   registry/  │  │ processing/  │  │  exporters/  │  │     io/      │    │
│  │              │  │     ply/     │  │              │  │              │    │
│  │ SourceReg   │  │  Loader      │  │  PlySink     │  │ UniversalPath│    │
│  │ SinkReg     │  │  Writer      │  │  Compressed  │  │  Discovery   │    │
│  │              │  │  Format      │  │  PlySink     │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  resources/  │  │  resilience/ │  │  validation/ │  │    health/   │    │
│  │              │  │              │  │              │  │              │    │
│  │ GPUResource  │  │  Retry       │  │  ConfigValid │  │ HealthMonitor│    │
│  │  Manager     │  │  CircuitBrk  │  │              │  │              │    │
│  │  Executor    │  │              │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Models Layer                                        │
└─────────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │         ply/                 │  │       composite/              │        │
│  │                              │  │                               │        │
│  │  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │        │
│  │  │   OptimizedPlyModel    │  │  │  │    CompositeModel      │  │        │
│  │  │                        │  │  │  │                        │  │        │
│  │  │  - PLY file discovery  │  │  │  │  - Multi-layer scenes  │  │        │
│  │  │  - Format detection    │  │  │  │  - Layer visibility    │  │        │
│  │  │  - Frame caching       │  │  │  │  - Z-order management  │  │        │
│  │  │  - Lifecycle mgmt      │  │  │  │                        │  │        │
│  │  └────────────────────────┘  │  │  └────────────────────────┘  │        │
│  │                              │  │                               │        │
│  │  ┌────────────────────────┐  │  │                               │        │
│  │  │  InterpolatedPlyModel  │  │  │                               │        │
│  │  │  - Temporal blending   │  │  │                               │        │
│  │  └────────────────────────┘  │  │                               │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Plugin System

The plugin architecture enables extensibility through custom data sources and sinks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Plugin System                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  src/plugins/                                                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  base/                                                                │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │ DataSourceBase │  │  DataSinkBase  │  │   Decorators   │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ - lifecycle    │  │ - export()     │  │ @source_plugin │         │  │
│  │  │ - discovery    │  │ - format mgmt  │  │ @sink_plugin   │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  discovery.py                                                         │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ PluginDiscovery: Auto-discovers plugins via entry_points       │  │  │
│  │  │ - Scans pyproject.toml [project.entry-points."gsplay.plugins"] │  │  │
│  │  │ - Registers sources with SourceRegistry                        │  │  │
│  │  │ - Registers sinks with SinkRegistry                            │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Registered Plugins (via entry_points)                                │  │
│  │                                                                        │  │
│  │  Sources:                          Sinks:                             │  │
│  │  ┌──────────────────────┐          ┌──────────────────────┐          │  │
│  │  │ load-ply             │          │ ply                   │          │  │
│  │  │ → OptimizedPlyModel  │          │ → PlySink             │          │  │
│  │  ├──────────────────────┤          ├──────────────────────┤          │  │
│  │  │ composite            │          │ compressed-ply        │          │  │
│  │  │ → CompositeModel     │          │ → CompressedPlySink   │          │  │
│  │  └──────────────────────┘          └──────────────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Plugin Lifecycle:
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ CREATED │───▶│  INIT   │───▶│  READY  │───▶│SUSPENDED│───▶│SHUTDOWN │
  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
       │              │              │              │              │
       │         on_init()      on_load()     on_unload()   on_shutdown()
```

---

## Level 4: Code Diagram

### Key Classes and Protocols

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Domain Protocols                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────┐
│         BaseGaussianSource             │
│              <<Protocol>>              │
├────────────────────────────────────────┤
│ + metadata() -> SourceMetadata         │
│ + can_load(path: str) -> bool          │
│ + total_frames: int                    │
│ + time_domain: TimeDomain              │
│ + get_frame_at_time(t: float)          │
│   -> GaussianData                      │
│ + get_processed_frame(t: float,        │
│   config, device) -> GSTensor          │
└────────────────────────────────────────┘
                    △
                    │ implements
        ┌───────────┴───────────┐
        │                       │
┌───────────────────┐   ┌───────────────────┐
│ OptimizedPlyModel │   │  CompositeModel   │
├───────────────────┤   ├───────────────────┤
│ - ply_files       │   │ - layers          │
│ - frame_cache     │   │ - visibility      │
│ - format_info     │   │ - z_order         │
├───────────────────┤   ├───────────────────┤
│ + discover_files()│   │ + add_layer()     │
│ + load_frame()    │   │ + remove_layer()  │
│ + get_frame_at_   │   │ + merge()         │
│   time()          │   │                   │
└───────────────────┘   └───────────────────┘


┌────────────────────────────────────────┐
│         LifecycleProtocol              │
│              <<Protocol>>              │
├────────────────────────────────────────┤
│ + state: PluginState                   │
│ + on_init()                            │
│ + on_load()                            │
│ + on_unload()                          │
│ + on_shutdown(timeout: float)          │
└────────────────────────────────────────┘


┌────────────────────────────────────────┐
│         DataSinkProtocol               │
│              <<Protocol>>              │
├────────────────────────────────────────┤
│ + metadata() -> DataSinkMetadata       │
│ + write(data: GaussianData,            │
│         path: str) -> bool             │
└────────────────────────────────────────┘
                    △
                    │ implements
        ┌───────────┴───────────┐
        │                       │
┌───────────────────┐   ┌───────────────────┐
│     PlySink       │   │CompressedPlySink  │
└───────────────────┘   └───────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Entities                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────┐
│              GSTensor                  │
│            (GPU - PyTorch)             │
├────────────────────────────────────────┤
│ + means: Tensor      [N, 3]            │
│ + scales: Tensor     [N, 3]            │
│ + quats: Tensor      [N, 4]            │
│ + opacities: Tensor  [N]               │
│ + sh0: Tensor        [N, 3]            │
│ + shN: Tensor        [N, K, 3]         │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│               GSData                   │
│            (CPU - NumPy)               │
├────────────────────────────────────────┤
│ + means: ndarray     [N, 3]            │
│ + scales: ndarray    [N, 3]            │
│ + quats: ndarray     [N, 4]            │
│ + opacities: ndarray [N]               │
│ + sh0: ndarray       [N, 3]            │
│ + shN: ndarray       [N, K, 3]         │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│            GaussianData                │
│          (Unified Wrapper)             │
├────────────────────────────────────────┤
│ + data: GSTensor | GSData              │
│ + device: str                          │
├────────────────────────────────────────┤
│ + to_gpu(device) -> GSTensor           │
│ + to_cpu() -> GSData                   │
│ + num_gaussians: int                   │
└────────────────────────────────────────┘
```

### Viewer Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Viewer Runtime Flow                                    │
└─────────────────────────────────────────────────────────────────────────────┘

     User Input                    System Events
          │                              │
          ▼                              ▼
    ┌──────────┐                  ┌──────────────┐
    │  Viser   │                  │  Playback    │
    │   UI     │                  │  Controller  │
    └────┬─────┘                  └──────┬───────┘
         │                               │
         │  UI events                    │ tick events
         ▼                               ▼
    ┌─────────────────────────────────────────┐
    │               EventBus                   │
    │                                          │
    │  EventType: COLOR, TRANSFORM, PLAYBACK  │
    └────────────────────┬────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  Camera   │   │   Edit    │   │   Model   │
   │Controller │   │  Manager  │   │ Component │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │
         │               │               │
         ▼               ▼               ▼
    ┌─────────────────────────────────────────┐
    │            RenderComponent               │
    │                                          │
    │  gsplat rasterization → JPEG encoding   │
    └────────────────────┬────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │   Viser   │   │  GStream  │   │  Control  │
   │  Browser  │   │  WebSocket│   │    API    │
   │  (N)      │   │  (N+1)    │   │   (N+2)   │
   └───────────┘   └───────────┘   └───────────┘
```

---

## Data Flow

### PLY Load to Render Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Pipeline                                │
└──────────────────────────────────────────────────────────────────────────┘

  PLY Files                    Memory                      Display
  (Storage)                    (GPU)                      (Browser)
      │                          │                            │
      ▼                          │                            │
┌───────────┐                    │                            │
│  Discover │                    │                            │
│  & Sort   │                    │                            │
│  Files    │                    │                            │
└─────┬─────┘                    │                            │
      │                          │                            │
      ▼                          │                            │
┌───────────┐                    │                            │
│   Load    │                    │                            │
│   PLY     │─────────Load──────▶│                            │
│  (gsply)  │                    │                            │
└─────┬─────┘                    │                            │
      │                          ▼                            │
      │                   ┌───────────┐                       │
      │                   │ GSTensor  │                       │
      │                   │  (raw)    │                       │
      │                   └─────┬─────┘                       │
      │                         │                             │
      │                         ▼                             │
      │                   ┌───────────┐                       │
      │                   │   Edit    │                       │
      │                   │  Manager  │                       │
      │                   │  (color,  │                       │
      │                   │  opacity, │                       │
      │                   │  filter)  │                       │
      │                   └─────┬─────┘                       │
      │                         │                             │
      │                         ▼                             │
      │                   ┌───────────┐                       │
      │                   │ GSTensor  │                       │
      │                   │ (edited)  │                       │
      │                   └─────┬─────┘                       │
      │                         │                             │
      │                         ▼                             │
      │                   ┌───────────┐                       │
      │                   │  gsplat   │                       │
      │                   │ Rasterize │                       │
      │                   └─────┬─────┘                       │
      │                         │                             │
      │                         ▼                             │
      │                   ┌───────────┐                       │
      │                   │   JPEG    │                       │
      │                   │  Encode   │──────WebSocket───────▶│
      │                   │ (nvJPEG)  │                       │
      │                   └───────────┘                       │
      │                                                       │
      │                                                       ▼
      │                                                ┌───────────┐
      │                                                │  Browser  │
      │                                                │  Display  │
      │                                                └───────────┘
```

### Edit Pipeline & Processing Modes

The Edit Manager applies transformations to raw Gaussian data before rendering.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Edit Pipeline                                     │
└──────────────────────────────────────────────────────────────────────────┘

  GSTensor (raw)
       │
       ▼
┌─────────────────┐
│  VolumeFilter   │ ── Filters by bounds, opacity threshold, scale limits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│SceneTransformer │ ── Applies rotation, translation, scale transforms
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ColorProcessor  │ ── Adjusts hue, saturation, brightness, contrast
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│OpacityAdjuster  │ ── Scales opacity values
└────────┬────────┘
         │
         ▼
  GSTensor (edited)


Processing Modes (CPU/GPU Strategy Selection):
┌──────────────────────────────────────────────────────────────────────────┐
│  Mode              │  Filter  │ Transform │  Color   │  Opacity │ Use    │
├──────────────────────────────────────────────────────────────────────────┤
│  ALL_GPU (default) │   GPU    │    GPU    │   GPU    │   GPU    │ Fast   │
│  COLOR_GPU         │   CPU    │    CPU    │   GPU    │   GPU    │ Mixed  │
│  TRANSFORM_GPU     │   CPU    │    GPU    │   CPU    │   CPU    │ Mixed  │
│  ALL_CPU           │   CPU    │    CPU    │   CPU    │   CPU    │ Low    │
│                    │          │           │          │          │ VRAM   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Event-Driven Architecture

The viewer uses an event bus for loose coupling between components.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Event System                                      │
└──────────────────────────────────────────────────────────────────────────┘

  Event Types:
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │  MODEL_LOADED  │  │ FRAME_CHANGED  │  │EXPORT_REQUESTED│
  └────────────────┘  └────────────────┘  └────────────────┘
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │  COLOR_CHANGED │  │TRANSFORM_CHANGED│ │RERENDER_NEEDED │
  └────────────────┘  └────────────────┘  └────────────────┘

  Event Flow:
  ┌──────────┐     emit()     ┌──────────┐    dispatch()    ┌──────────┐
  │ Producer │ ─────────────▶ │ EventBus │ ───────────────▶ │ Handler  │
  │ (UI/Ctrl)│                │          │                  │          │
  └──────────┘                └──────────┘                  └──────────┘
       │                           │                             │
  Viser GUI                   Centralized                  App callbacks:
  Playback                     routing                   _on_frame_changed
  Control API                                            _on_color_changed
```

---

## Deployment

### Single Viewer

```bash
uv run gsplay --config ./path/to/ply/folder
```

```
┌────────────────────────────────────────────────────────────────┐
│  Host Machine                                                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  GSPlay Viewer Process                                   │  │
│  │                                                          │  │
│  │  Port 6020: Viser UI                                    │  │
│  │  Port 6021: GStream WebSocket                           │  │
│  │  Port 6022: Control API                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   NVIDIA GPU    │                                           │
│  │   CUDA 12.8     │                                           │
│  └─────────────────┘                                           │
└────────────────────────────────────────────────────────────────┘
```

### Multi-Instance with Launcher

```bash
uv run -m gsplay_launcher --browse-path /data/scenes
```

```
┌────────────────────────────────────────────────────────────────┐
│  Host Machine                                                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  GSPlay Launcher                                       │    │
│  │  Port 8000                                             │    │
│  │                                                        │    │
│  │  ┌──────────────┐  ┌──────────────┐                   │    │
│  │  │ FastAPI      │  │ SolidJS      │                   │    │
│  │  │ Backend      │  │ Frontend     │                   │    │
│  │  └──────────────┘  └──────────────┘                   │    │
│  └───────────────────────────┬───────────────────────────┘    │
│                              │ subprocess                      │
│              ┌───────────────┼───────────────┐                │
│              ▼               ▼               ▼                │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │ Viewer #1     │ │ Viewer #2     │ │ Viewer #3     │       │
│  │ 6020-6022     │ │ 6024-6026     │ │ 6028-6030     │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐                      │
│  │   GPU 0         │ │   GPU 1         │                      │
│  └─────────────────┘ └─────────────────┘                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Technology Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | SolidJS, TypeScript | Launcher dashboard |
| **UI Framework** | Viser | 3D web viewer |
| **Backend** | FastAPI, Python 3.12 | Launcher API |
| **Rendering** | gsplat, CUDA 12.8 | GPU rasterization |
| **ML Framework** | PyTorch 2.9+ | Tensor operations |
| **Streaming** | WebSocket, asyncio | Low-latency delivery |
| **Data Format** | PLY (gsply) | Gaussian splatting data |
| **Build** | uv, Deno | Package management |

---

## Security Considerations

### Network Exposure

| Service | Binding | Authentication | Notes |
|---------|---------|----------------|-------|
| **Launcher** | 0.0.0.0:8000 | None | Use firewall in production |
| **Viewer (Viser)** | 0.0.0.0:N | None | WebSocket based |
| **GStream** | 0.0.0.0:N+1 | None | Binary JPEG stream |
| **Control API** | 0.0.0.0:N+2 | None | HTTP REST endpoints |

### Recommendations

1. **Firewall Configuration**: All services bind to all interfaces by default. Configure firewall rules to restrict access in production.

2. **TLS/HTTPS**: WebSocket streams transmit unencrypted JPEG frames. Use a reverse proxy (nginx, Caddy) with TLS for internet-facing deployments.

3. **File Access**: The launcher's file browser exposes the configured browse path. Limit `--browse-path` to intended directories only.

4. **GPU Resources**: No resource limits by default. Multi-tenant deployments should implement GPU memory quotas.

---

## References

- [C4 Model](https://c4model.com/)
- [Viser Documentation](https://viser.studio/)
- [gsplat Documentation](https://docs.gsplat.studio/)
- [GSPlay README](https://github.com/opsiclear/gsplay/blob/master/README.md)
- [Architecture Details](https://github.com/opsiclear/gsplay/blob/master/CLAUDE.md)
