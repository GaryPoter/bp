# -*- coding:utf-8 -*-
# author: 何伟
import math
from array import array
import reprlib
import numbers


class Vector:
    typecode = 'd'

    def __init__(self, args):
        self._components = array(self.typecode, args)

    def __iter__(self):
        return iter(self._components)

    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['): -1]
        return 'Vector({})'.format(components)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return ((bytes[ord(self.typecode)]) +
                bytes(array(self._components)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.sqrt(sum(x * x for x in self))

    def __bool__(self):
        return bool(abs(self))

    @classmethod
    def frombytes(cls, octets):
        typecode = octets[0]
        memv = memoryview(octets[: -1]).cast(typecode)
        return cls(memv)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, item):
        # return self._components[item]
        cls = type(self)
        if isinstance(item, slice):
            return cls(self._components[item])
        elif isinstance(item, numbers.Integral):
            return self._components[item]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))


v1 = Vector([3, 4, 5])
v7 = Vector(range(7))
print(v7[1: 4])
# print(format(abs(v1), '.2f'))
