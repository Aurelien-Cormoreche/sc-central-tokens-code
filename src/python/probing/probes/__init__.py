"""Importing this package registers every built-in probe type (see base.py's
PROBE_FACTORY_REGISTRY) -- experiment.py only needs `import
src.python.probing.probes` for build_active_probes() to see them all."""
from src.python.probing.probes import cell_level, context  # noqa: F401 -- side-effect registration
