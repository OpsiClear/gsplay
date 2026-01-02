# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime


# -- Path setup --------------------------------------------------------------
# Add source directories to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../launcher"))

# -- Project information -----------------------------------------------------
project = "GSPlay"
copyright = f"{datetime.now().year}, OpsiClear"
author = "OpsiClear"
version = "0.1.1"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # Markdown support
    "myst_parser",
    # Additional features
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Source file configuration -----------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# -- MyST-Parser configuration (Markdown support) ---------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3

# -- Napoleon configuration (NumPy/Google docstrings) -----------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Use ivar format for instance variables to reduce duplication with dataclass fields
napoleon_use_ivar = True
# Don't include attribute types in docstrings (let type hints handle it)
napoleon_attr_annotations = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__, __dict__, __module__, __dataclass_fields__, __dataclass_params__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"


# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "gsplat",
    "gsply",
    "gsmod",
    "gsmod.torch",
    "gsmod.gsdata_pro",
    "gsmod.config",
    "gsmod.config.values",
    "viser",
    "viser.transforms",
    "tyro",
    "scipy",
    "tqdm",
    "triton",
    "pyyaml",
    "yaml",
    "websockets",
    "fastapi",
    "uvicorn",
    "psutil",
    "httpx",
    "sse_starlette",
    "starlette",
    "pydantic",
    "packaging",
]

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output options -----------------------------------------------------
html_theme = "furo"
html_title = "GSPlay Documentation"
html_short_title = "GSPlay"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/opsiclear/gsplay",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0366d6",
        "color-brand-content": "#0366d6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#58a6ff",
        "color-brand-content": "#58a6ff",
    },
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- TODO extension ----------------------------------------------------------
todo_include_todos = True
