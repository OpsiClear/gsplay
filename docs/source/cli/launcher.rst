gsplay_launcher CLI
===================

The ``gsplay_launcher`` command starts a web-based dashboard for managing
multiple GSPlay viewer instances.

Usage
-----

.. code-block:: bash

   gsplay_launcher --browse-path /data/scenes

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--browse-path``
     - .
     - Root directory for file browser
   * - ``--port``
     - 8000
     - Launcher web server port
   * - ``--host``
     - 0.0.0.0
     - Host to bind to
   * - ``--history-limit``
     - 100
     - Maximum number of instances to keep in history
   * - ``--gpu-selection``
     - True
     - Enable GPU selection in launch dialog

Features
--------

File Browser
~~~~~~~~~~~~

Browse PLY folders and launch viewer instances directly from the web UI.

Instance Management
~~~~~~~~~~~~~~~~~~~

- View running instances
- See instance logs in real-time
- Stop instances
- Track launch history

GPU Monitoring
~~~~~~~~~~~~~~

Real-time GPU utilization and memory usage for each detected NVIDIA GPU.

Stream Preview
~~~~~~~~~~~~~~

Preview the WebSocket stream from running instances directly in the dashboard.

Examples
--------

.. code-block:: bash

   # Basic usage
   gsplay_launcher --browse-path /data/scenes

   # Custom port
   gsplay_launcher --port 8080

   # Disable GPU selection (single GPU systems)
   gsplay_launcher --gpu-selection false

Cleanup
-------

To clean up orphaned instances:

.. code-block:: bash

   clear_instances
