# -*- coding: utf-8 -*-
"""
Author -- Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 04.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Example solutions for tasks in the provided tasks file.
"""

from collections.abc import Sequence
from typing import Optional

# Given the "LinkedList" implementation below, your tasks will be to extend the
# functionality of this class. For training purposes, do NOT use any existing
# data structures in your implementations.


class Node:
    
    def __init__(self, val):
        self.val = val
        self.next: Optional[Node] = None


class LinkedList:
    
    def __init__(self):
        self._head: Optional[Node] = None
        self._size: int = 0
    
    def __len__(self):
        return self._size
    
    def __iter__(self):
        class LinkedListIterator:
            
            def __init__(self, start: Node):
                self.curr = start
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.curr is None:
                    raise StopIteration
                item = self.curr.val
                self.curr = self.curr.next
                return item
        
        return LinkedListIterator(self._head)
    
    #
    # Task 1
    #
    
    # Extend the "LinkedList" class by providing the method "append(self, val)"
    # that inserts "val" at the end of the list. Also make sure to adjust the
    # "_size" attribute accordingly or the provided "__len__" method does not
    # work correctly.
    def append(self, val):
        new_node = Node(val)
        # Special case: start of list.
        if self._head is None:
            self._head = new_node
        else:
            curr = self._head
            while curr.next is not None:
                curr = curr.next
            curr.next = new_node
        self._size += 1
    
    #
    # Task 2
    #
    
    # Extend the "LinkedList" class by providing the method "remove(self, val)"
    # that removes the first occurrence of "val" and returns True if the list
    # contains "val". If "val" is not in the list, the list remains unchanged
    # and False is returned instead. Also make sure to adjust the "_size"
    # attribute accordingly or the provided "__len__" method does not work
    # correctly.
    def remove(self, val):
        prev = None
        curr = self._head
        while curr is not None and curr.val != val:
            prev = curr
            curr = curr.next
        found = curr is not None
        if found:
            # Special case: start of list.
            if prev is None:  # Equivalent to "curr is self._head".
                self._head = curr.next
            else:
                prev.next = curr.next
            self._size -= 1
        return found
    
    #
    # Task 3
    #
    
    # Extend the "LinkedList" class by providing the special method "__str__" to
    # get a human-friendly string representation of a list, which should look
    # like: empty: "[]", one item: "[1]", n items: "[1, 7, 4]". Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__str__
    def __str__(self):
        s = "["
        for i, elem in enumerate(self):
            if i == self._size - 1:
                s += str(elem)
            else:
                s += f"{elem}, "
        return s + "]"
    
    #
    # Task 4
    #
    
    # Extend the "LinkedList" class by providing the special method "__eq__" to
    # enable equality checks on "LinkedList" objects. A "LinkedList" should be
    # considered equal to another "LinkedList" if they have the same length and
    # if all elements at the same index positions are equal. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__eq__
    def __eq__(self, other):
        # Could implement this shortcut to avoid potentially time-consuming
        # equality comparisons (return as soon as possible) if the list object
        # is compared with itself. The question is if this scenario is likely
        # regarding our lists, since otherwise, we just perform an additional
        # identity check and have to check element-wise afterwards anyway. If
        # the equality comparison is (potentially) time-consuming and self
        # comparisons are not unlikely to occur, then such a shortcut is a good
        # idea. While the first is true for our lists (element-wise comparison
        # is increasingly slow for larger lists), the second arguably is not, as
        # comparing a list object with itself is rather unlikely.
        #
        # if self is other:
        #     return True
        if not isinstance(other, LinkedList):
            return NotImplemented
        if self._size != other._size:
            return False
        for elem1, elem2 in zip(self, other):
            if elem1 != elem2:
                return False
        return True
    
    #
    # Task 5
    #
    
    # Extend the "LinkedList" class by providing the special method
    # "__getitem__" to enable Python's indexing support. Valid inputs are
    # integers and "slice" objects
    # (https://docs.python.org/3/library/functions.html#slice). For an integer,
    # the corresponding item at this index position should be returned. For a
    # "slice" object, a new "LinkedList" object with the items specified by the
    # index range of this "slice" object should be returned. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
    def __getitem__(self, key):
        if isinstance(key, int):
            # Could check beforehand if "key" is actually a valid index and
            # raise an "IndexError" to avoid unnecessarily iterating through the
            # list (see the solution implementation for "__setitem__" with int
            # keys). Iteration is slow, but this is just how linked lists work.
            for i, elem in enumerate(self):
                # Negative index support, e.g.: 0, 1, 2 -> -3, -2, -1 = i - size
                if i == key or i - self._size == key:
                    return elem
            raise IndexError("LinkedList index out of range")
        if isinstance(key, slice):
            new_ll = LinkedList()
            # Extract valid start/stop indices to avoid running out-of-bounds.
            start, stop, step = key.indices(self._size)
            # Simply calling "__getitem__" for all slice indices is actually
            # incredibly slow, since we repeatedly iterate over our list
            # (ideally, we would only iterate once and collect all elements in
            # this single iteration). However, for convenience's sake, this
            # suffices for now.
            for i in range(start, stop, step):
                new_ll.append(self[i])
            return new_ll
        raise TypeError(f"LinkedList indices must be integers or slices, not {type(key).__name__}")
    
    #
    # Task 6
    #
    
    # Extend the "LinkedList" class by providing the special method
    # "__setitem__" to enable Python's indexing support, now also for setting
    # values. Valid inputs for the index/indices are integers and "slice"
    # objects. Valid inputs for the value/values that should be set are single
    # objects for integers and "Sequence" objects
    # (https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)
    # for "slice" objects (the number of elements in the sequence must match the
    # number of elements specified by the index range of the "slice" object).
    # Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__setitem__
    def __setitem__(self, key, value):
        if isinstance(key, int):
            # Transform negative indices, and then, also check for valid index
            # ranges, which makes life easier for us.
            if key < 0:
                key = key + self._size
            if key < 0 or key >= self._size:
                raise IndexError("LinkedList assignment index out of range")
            # Now, it is guaranteed that "key" is a positive and valid index,
            # so we definitely will find the corresponding list node.
            curr = self._head
            for i in range(key):
                curr = curr.next
            curr.val = value
        elif isinstance(key, slice):
            if not isinstance(value, Sequence):
                raise TypeError("can only assign a sequence")
            # Extract valid start/stop indices to avoid running out-of-bounds.
            start, stop, step = key.indices(self._size)
            # Trick: Wrap the slice into a "range" object, which supports "len".
            if len(range(start, stop, step)) != len(value):
                raise ValueError("sequence must have equally many elements as there are slice indices")
            # Same performance consideration as mentioned in "__getitem__".
            for i, j in enumerate(range(start, stop, step)):
                self[j] = value[i]
        else:
            raise TypeError(f"LinkedList indices must be integers or slices, not {type(key).__name__}")


# Some testing code including output and assertions
print("Starting with an empty LinkedList")
ll = LinkedList()
assert str(ll) == "[]"
print(ll)
print("Appending values from [0..10)")
for v in range(10):
    ll.append(v)
assert len(ll) == 10
assert str(ll) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
print(ll)

removed = ll.remove(5)
print("Removing 5 successful?", removed)
assert removed and len(ll) == 9
print(ll)
removed = ll.remove(123)
print("Removing 123 successful?", removed)
assert not removed and len(ll) == 9
print(ll)

print()
print("Getting value at index 1")
assert ll[1] == 1
print(ll[1])
print("Getting value at index -1")
assert ll[-1] == 9
print(ll[-1])
print("Getting value at incorrect index 123")
try:
    ll[123]
except IndexError as ex:
    print(ex)
print("Getting value with incorrect index type '1'")
try:
    ll["1"]
except TypeError as ex:
    print(ex)
print("Getting values with slice [1:6:2]")
expected = LinkedList()
expected.append(1)
expected.append(3)
expected.append(6)
assert ll[1:6:2] == expected
print(ll[1:6:2])
print("Getting values with slice [::-1]")
expected = LinkedList()
for v in [9, 8, 7, 6, 4, 3, 2, 1, 0]:
    expected.append(v)
assert ll[::-1] == expected
print(ll[::-1])

print()
print("Setting value at index 1 to 90")
ll[1] = 90
assert ll[1] == 90
print(ll)
print("Setting value at index -1 to 123")
ll[-1] = 123
assert ll[-1] == 123
print(ll)
print("Setting value at incorrect index 123")
try:
    ll[123] = "will fail anyway"
except IndexError as ex:
    print(ex)
print("Setting value with incorrect index type '1'")
try:
    ll["1"] = "will fail anyway"
except TypeError as ex:
    print(ex)
print("Setting values with slice [1:6:2] to [11, 33, 55]")
ll[1:6:2] = [11, 33, 55]
expected = LinkedList()
expected.append(11)
expected.append(33)
expected.append(55)
assert ll[1:6:2] == expected
print(ll)
print("Setting values with slice [::-3] to [-1, -2, -3]")
ll[::-3] = [-1, -2, -3]
expected = LinkedList()
expected.append(-1)
expected.append(-2)
expected.append(-3)
assert ll[::-3] == expected
print(ll)
print("Setting incorrect value type 1 with slice")
try:
    ll[:] = 1
except TypeError as ex:
    print(ex)
print("Setting incorrectly sized value [1, 2, 3] with slice [1:3] of different size")
try:
    ll[1:3] = [1, 2, 3]  # len(slice indices) == 2 != len(values) == 3
except ValueError as ex:
    print(ex)

print()
ll1 = LinkedList()
ll1.append(1)
assert str(ll1) == "[1]"
ll2 = LinkedList()
ll2.append(1)
print("ll1 =", ll1)
print("ll2 =", ll2)
print("ll1 == ll2?", ll1 == ll2)
assert ll1 == ll2
print("Appending value 2 to ll2")
ll2.append(2)
print("ll2 =", ll2)
print("ll1 == ll2?", ll1 == ll2)
assert ll1 != ll2
