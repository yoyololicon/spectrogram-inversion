# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../torch_specinv'))


# -- Project information -----------------------------------------------------

project = 'torch_specinv'
copyright = '2019, Chin Yun Yu'
author = 'Chin Yun Yu'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'pytorch_sphinx_theme'
html_theme = 'sphinx_rtd_theme'
# html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#


html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'torch': ('https://pytorch.org/docs/master/', None)
}

autodoc_mock_imports = ['torch', 'tqdm']

# prevent contents.rst error on readthedocs
master_doc = 'index'