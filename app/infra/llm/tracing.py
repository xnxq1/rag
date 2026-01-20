from abc import ABC, abstractmethod
from typing import Callable

from langsmith import traceable

from app.infra.config import settings


class TracerInterface(ABC):
    @abstractmethod
    def trace(self, name: str, run_type: str = "chain") -> Callable:
        pass


class LangSmithTracer(TracerInterface):
    def trace(self, name: str, run_type: str = "chain") -> Callable:
        return traceable(name=name, run_type=run_type)


class NoOpTracer(TracerInterface):
    def trace(self, name: str, run_type: str = "chain") -> Callable:
        def decorator(func):
            return func  # Ничего не делает

        return decorator


def get_tracer() -> TracerInterface:
    if settings.langsmith_tracing:
        return LangSmithTracer()
    return NoOpTracer()


tracer = get_tracer()
