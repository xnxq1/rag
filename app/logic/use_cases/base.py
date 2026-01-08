import abc
from typing import Any


class UseCaseInterface(abc.ABC):
    @abc.abstractmethod
    async def handle(self, *args, **kwargs) -> Any: ...
