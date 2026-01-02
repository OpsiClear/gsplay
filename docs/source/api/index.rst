API Reference
=============

This section provides detailed API documentation for all GSPlay modules.

The codebase follows Clean Architecture with these layers:

- **Domain**: Core business entities and interfaces (no dependencies)
- **Infrastructure**: I/O, caching, external adapters
- **Models**: Application-level model implementations
- **GSPlay**: Presentation layer (UI, rendering, viewer)
- **Plugins**: Plugin system and base classes
- **Shared**: Cross-cutting utilities
- **Launcher**: Web-based instance management

.. toctree::
   :maxdepth: 2

   domain
   infrastructure
   models
   gsplay
   plugins
   shared
   launcher
