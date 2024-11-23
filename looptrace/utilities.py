"""Very general-purpose utilities"""

import functools
from typing import Callable, Iterable, Mapping, Optional, ParamSpec, TypeVar
from expression import Option, Result, compose, curry_flip
from numpydoc_decorator import doc

_A = TypeVar("_A")
_K = TypeVar("_K")
_P = ParamSpec("_P")
_V = TypeVar("_V")

_Exception = TypeVar("_Exception", bound=Exception)


@curry_flip(1)
@doc(
    summary="Build a function which finds the first element from a given collection satisfying the given predicate",
    parameters=dict(
        items="The collection in which to search", 
        predicate="The criterion for selecting the first element",
    ), 
    returns="A function which finds the first element in a given collection which satisfies the given predicate",
)
def find_first_option(items: Iterable[_A], predicate: Callable[[_A], bool]) -> Option[_A]:
    for a in items:
        if predicate(a):
            return Option.Some(a)
    return Option.Nothing()


@curry_flip(1)
@doc(
    summary="Lookup keys but with expression.Result API",
    parameters=dict(
        k="The key for which to fetch the value", 
        m="The mapping in which to do the lookup",
    ), 
    returns="Either the value at the given key, or an error message",
)
def get_either(k: _K, m: Mapping[_K, _V]) -> Result[_V, str]:
    return get_option(m)(k).to_result(f"Missing key: {k}")


@curry_flip(1)
@doc(
    summary="Lookup keys but with expression.Option API",
    parameters=dict(
        k="The key for which to fetch the value", 
        m="The mapping in which to do the lookup",
    ), 
    returns="Either the value at the given key, or the empty value",
)
def get_option(k: _K, m: Mapping[_K, _V]) -> Option[_V]:
    return Option.of_obj(m.get(k))


# Courtest of @Hugovdberg in Issues discussion on dbratti/Expression repo
@curry_flip(1)
def wrap_exception(
    fun: Callable[_P, _A],
    exc: type[_Exception] | tuple[type[_Exception], ...] = Exception,
) -> Callable[_P, Result[_A, _Exception]]:
    """Wrap a function that might raise an Exception in a Result monad

    Args:
        fun (Callable[P, a]):
            The function to be wrapped.
        exc (Union[Tuple[Type[Exception], ...], Type[Exception]], optional):
            The Exception types to be wrapped into the monad. Defaults to Exception.

    Returns:
        Callable[P, Result[a, Exception]]: 
            The decorated function.

    Examples:
        >>> @wrap_exception(ZeroDivisionError)
        ... def inverse(x: int) -> float:
        ...     return 1 / x
        >>> t: Result[float, ZeroDivisionError] = inverse(0)
    """

    @functools.wraps(fun)
    def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Result[_A, _Exception]:
        try:
            return Result[_A, _Exception].Ok(fun(*args, **kwargs))
        except exc as e:
            return Result[_A, _Exception].Error(e)

    return _wrapper


@curry_flip(1)
def wrap_error_message(
    fun: Callable[_P, Result[_A, _Exception]],
    context: Optional[str] = None,
) -> Callable[_P, Result[_A, str]]:
    write_error: Callable[[_Exception], str] = \
        str if context is None else (lambda e: f"{context}: {e}")
    def transform(either: Result[_A, _Exception]) -> Result[_A, str]:
        return either.map_error(write_error)
    return compose(fun, transform)
