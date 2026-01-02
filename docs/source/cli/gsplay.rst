gsplay CLI
==========

The ``gsplay`` command launches the Gaussian Splatting viewer.

Usage
-----

.. code-block:: bash

   # Direct folder path (simplest)
   gsplay /path/to/ply/folder

   # With explicit subcommand
   gsplay view /path/to/ply/folder

   # With options
   gsplay view /path/to/ply/folder --port 6019 --host 0.0.0.0

Commands
--------

view (default)
~~~~~~~~~~~~~~

Launch the GSPlay viewer.

.. list-table:: Options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``config``
     - (required)
     - Path to model configuration (JSON file or PLY folder)
   * - ``--port``
     - 6019
     - Viser server port
   * - ``--host``
     - 0.0.0.0
     - Host to bind to
   * - ``--stream-port``
     - -1 (auto)
     - WebSocket stream port. Set to 0 to disable.
   * - ``--log-level``
     - INFO
     - Logging level: DEBUG, INFO, WARNING, ERROR
   * - ``--gpu``
     - None (auto)
     - GPU device number
   * - ``--view-only``
     - False
     - Hide input path, config save, and export options
   * - ``--compact``
     - False
     - Mobile-friendly compact UI

plugins-list
~~~~~~~~~~~~

List all discovered plugins.

.. code-block:: bash

   gsplay plugins-list
   gsplay plugins-list --verbose

plugins-info
~~~~~~~~~~~~

Show information about a specific plugin.

.. code-block:: bash

   gsplay plugins-info load-ply

plugins-test
~~~~~~~~~~~~

Test a plugin with the test harness.

.. code-block:: bash

   gsplay plugins-test load-ply --config ./config.json

Examples
--------

Basic Viewing
~~~~~~~~~~~~~

.. code-block:: bash

   # View PLY sequence
   gsplay ./my_scene/

   # Custom port
   gsplay ./my_scene/ --port 8080

   # Enable WebSocket streaming
   gsplay ./my_scene/ --stream-port 8081

Configuration File
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using JSON config
   gsplay --config scene_config.json

Remote Access
~~~~~~~~~~~~~

.. code-block:: bash

   # Bind to all interfaces for remote access
   gsplay ./scene/ --host 0.0.0.0 --port 6020
