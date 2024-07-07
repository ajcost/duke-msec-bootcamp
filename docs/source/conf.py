# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../src"))


project = "Duke MSEC Code Bootcamp Assignment"
copyright = (
    f"{datetime.datetime.now().year}, by Boston Consulting Group. All rights reserved"
)
author = "Adam Costarino"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "myst_nb",
]

# Exclude build directory and Jupyter backup files:
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# Always build notebooks in the sphinx build
nbsphinx_execute = "always"

templates_path = ["_templates"]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ajcost/duke-msec-bootcamp",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    "collapse_navigation": False,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

myst_nb_execution_mode = "off"


def setup(app):
    """Overwrite rtd theme with custom features."""
    app.add_css_file("custom.css")
