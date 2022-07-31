from typing import Dict, Optional, Iterable, Tuple, Iterator
from tabulate import tabulate  # 优雅的创建表格的库


class Registry(Iterable[Tuple[str, object]]):

    def __init__(self, name: str):
        # name 是该注册器的名称
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:

        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )

        self._obj_map[name] = obj

    def register(self, obj: object = None) -> Optional[object]:
        # 如果传入了obj则执行注册，否则返回一个注册器。
        if obj is None:
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        # 从obj_map中获取到注册的对象
        ret = self._obj_map.get(name)

        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, object]]:
        return iter(self._obj_map.items())

    __str__ = __repr__  # TODO
