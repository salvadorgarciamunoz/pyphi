import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'pyphi'
copyright = '2026, Salvador Garcia Munoz'
author = 'Salvador Garcia Munoz'
release = '6.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']