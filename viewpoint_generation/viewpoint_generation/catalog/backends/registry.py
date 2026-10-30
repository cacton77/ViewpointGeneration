"""Backends by name.

Each entry maps a backend's registry name to a factory that builds it from
the environment. Imports are lazy, so a backend's optional dependencies are
only needed when that backend is configured (`PLM_BACKENDS`).
"""

import importlib

# name -> 'module:factory'. The factory takes no arguments and returns a
# configured backend instance.
BACKENDS = {
    'dx3': 'viewpoint_generation.catalog.backends.dx3.backend:from_env',
    'fake': 'viewpoint_generation.catalog.backends.fake:from_env',
}


def available():
    """Registered backend names."""
    return sorted(BACKENDS)


def backend_class(name):
    """The class implementing a registered backend (for CLI hooks), or None."""
    target = BACKENDS.get(name)
    if target is None:
        return None
    module_name, _ = target.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, 'BACKEND_CLASS', None)


def create(name):
    """Build a registered backend from its environment configuration.

    Raises:
        KeyError: when no backend of that name is registered.
    """
    target = BACKENDS[name]
    module_name, factory = target.split(':')
    return getattr(importlib.import_module(module_name), factory)()


def create_all(names):
    """Build every backend in a comma-separated or list of names.

    Unknown names are reported rather than silently dropped.

    Returns:
        tuple: (dict name -> backend, list of error messages)
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',') if n.strip()]
    backends, errors = {}, []
    for name in names:
        if name in backends:
            continue
        try:
            backends[name] = create(name)
        except KeyError:
            errors.append(f"Unknown PLM backend {name!r}; registered: {', '.join(available())}.")
        except Exception as e:  # noqa: BLE001 - one broken backend must not stop the others
            errors.append(f'Could not start PLM backend {name!r}: {e}')
    return backends, errors
