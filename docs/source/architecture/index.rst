Architecture
============

GSPlay follows Clean Architecture principles with clear separation of concerns.

.. toctree::
   :maxdepth: 2

   c4_architecture

Overview
--------

The codebase is organized into layers:

**Domain Layer** (``src/domain/``)
   Core business entities and interfaces with no external dependencies.
   Contains ``GSTensor``, ``GSData``, protocols, and pure business logic.

**Infrastructure Layer** (``src/infrastructure/``)
   I/O operations, caching, and external adapters.
   Handles PLY file loading, cloud storage, and exporters.

**Models Layer** (``src/models/``)
   Application-level model implementations.
   ``OptimizedPlyModel``, ``CompositeModel``, ``InterpolatedPlyModel``.

**Presentation Layer** (``src/gsplay/``)
   UI, rendering, and viewer application.
   Camera control, event handling, WebSocket streaming.

**Plugin System** (``src/plugins/``)
   Extensibility through custom data sources and sinks.

Dependency Flow
---------------

::

   gsplay/ --> models/ --> domain/
      |          |           ^
      v          v           |
   infrastructure/ ---------+
      (depends on domain interfaces)

The domain layer has no dependencies. Infrastructure depends only on domain
interfaces. Models implement domain interfaces. The presentation layer
orchestrates everything.
