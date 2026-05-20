"""
Proxy objects for lazy plugin function/class resolution with introspection support.

These proxies are transparent to IPython/Jupyter help and ``inspect.signature``:
accessing ``__doc__``, ``__wrapped__``, or other introspection attributes triggers
resolution of the underlying object and copies its metadata.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from contextlib import suppress
from typing import Any

_INTROSPECTION_ATTRS = frozenset(
    {
        "__doc__",
        "__name__",
        "__qualname__",
        "__module__",
        "__wrapped__",
        "__signature__",
        "__annotations__",
        "__code__",
        "__defaults__",
        "__kwdefaults__",
        "__closure__",
        "__globals__",
    }
)


class lazy_proxy:
    """
    A callable proxy that lazily resolves and caches the real callable.

    Parameters
    ----------
    resolve : callable
        A no-argument callable that returns the real object.

    Examples
    --------
    ::

        >>> def _resolve():
        ...     from myplugin import myfunc
        ...     return myfunc
        >>> proxy = lazy_proxy(_resolve)
        >>> proxy  # triggers resolution
        <function myfunc at ...>
        >>> proxy.__doc__  # real docstring
    """

    _MISSING = object()

    def __init__(self, resolve: Callable[[], Any]) -> None:
        self.__dict__["_resolve"] = resolve
        self.__dict__["_resolved"] = self._MISSING

    def _resolve_now(self) -> Any:
        if self.__dict__["_resolved"] is self._MISSING:
            resolved = self.__dict__["_resolve"]()
            self.__dict__["_resolved"] = resolved
            functools.update_wrapper(self, resolved)
            with suppress(ValueError, TypeError):
                self.__dict__["__signature__"] = inspect.signature(resolved)
        return self.__dict__["_resolved"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve_now()(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        if name in _INTROSPECTION_ATTRS:
            self._resolve_now()
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        resolved = self._resolve_now()
        return repr(resolved)
