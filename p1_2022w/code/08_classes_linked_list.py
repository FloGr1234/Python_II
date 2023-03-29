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

In this file, we will learn some more about classes and special methods by
implementing our own, custom data structure: a linked list.
"""

################################################################################
# Linked list
################################################################################

# Previously, we saw that we can extend the functionality of our classes by
# implementing various special methods. In this code here, we will introduce
# even more such methods, and to show their functionality, we use a new class
# called "LinkedList", which is a custom data structure that stores its elements
# in node objects rather than in a contiguous array (which, for instance, the
# Python list implementation does internally), and these node objects are linked
# together (hence the name). You can think of it as follows:
#
# The node class contains one element and a reference to the next node, e.g.,
#     some_node
#     +++++++++++++++
#     + val = 1     +
#     + next = None +
#     +++++++++++++++
# is a node with the integer object 1 as its element, and it currently does not
# reference any following node.
#
# A linked list is simply a collection of such nodes that reference each other.
# For instance, here is a linked list with three linked nodes:
#     node1 = head    node2           node3
#     +++++++++++     +++++++++++     +++++++++++++++
#     + val = 1 +     + val = 7 +     + val = 4     +
#     + next = -----> + next = -----> + next = None +
#     +++++++++++     +++++++++++     +++++++++++++++
# For accessing these linked nodes as a list, we simply start at "node1", i.e.,
# the "head" of our list, and then we follow the "next" references until there
# are no more nodes ("next" is None). Side note: This is a singly-linked list.
#
# This is exactly what is given below (feel free to inspect the code). For
# simplicity's sake, only a "prepend" method is implemented, which is enough
# to show the following special methods.

from typing import Optional  # Just for convenience (autocompletion, etc.)


# Utility class that models node objects in our linked list.
class Node:
    
    def __init__(self, val):
        self.val = val
        self.next: Optional[Node] = None


# We could extend a fitting class from the built-in module "collections.abc"
# (e.g., "Sequence"), but for training and demonstrating purposes, we will
# implement this data structure from scratch.
class LinkedList:
    
    def __init__(self):
        # Private head attribute (first node of our list and main access point).
        self._head: Optional[Node] = None
    
    def prepend(self, val):
        """Inserts the specified value at the start of the list."""
        new_node = Node(val)
        new_node.next = self._head
        self._head = new_node


# Create a new linked list and add some elements (the list will be [1, 7, 4]).
ll = LinkedList()
ll.prepend(4)
ll.prepend(7)
ll.prepend(1)


# One nice thing would be to get the size of our list with the built-in function
# "len". We can achieve this by implementing the special method "__len__".
class LinkedList:
    
    def __init__(self):
        # Private head attribute (first node of our list and main access point).
        self._head: Optional[Node] = None
    
    def prepend(self, val):
        """Inserts the specified value at the start of the list."""
        new_node = Node(val)
        new_node.next = self._head
        self._head = new_node
    
    # We just run through our linked list (node by node, starting at our head
    # node) and increment a counter. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__len__
    #
    # Note: Typically, we would have a separate attribute that stores the size
    # and is updated when the list is modified (this way, we do not have to
    # iterate through the entire list every time this method is called).
    # However, this is again just for demonstrating purposes how to iterate
    # through a linked list.
    def __len__(self):
        size = 0
        curr = self._head
        while curr is not None:
            curr = curr.next
            size += 1
        return size


# Create a new linked list, add some elements (the list will be [1, 7, 4]) and
# retrieve the size of this list.
ll = LinkedList()
ll.prepend(4)
ll.prepend(7)
ll.prepend(1)
print(len(ll))  # Will invoke "LinkedList.__len__(self=ll)".


# Now, we might be wondering whether we could also use Python's iteration
# mechanism with our list, so we could write something like
#     for elem in ll:
#         # Do something with "elem"
# since this is a typical list operation. And yes, we can. All we have to do is
# to provide an implementation for the method "__iter__", which will make our
# list iterable (thereby enabling Python's for-each loop above).
class LinkedList:
    
    def __init__(self):
        # Private head attribute (first node of our list and main access point).
        self._head: Optional[Node] = None
    
    def prepend(self, val):
        """Inserts the specified value at the start of the list."""
        new_node = Node(val)
        new_node.next = self._head
        self._head = new_node
    
    def __len__(self):
        size = 0
        curr = self._head
        while curr is not None:
            curr = curr.next
            size += 1
        return size
    
    # As per the specification, the "__iter__" method must return a new iterator
    # object, which is then used to iterate through our list. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__iter__
    def __iter__(self):
        # This means that we have to create our own, custom iterator here. We do
        # this by creating a new "LinkedListIterator" class that implements all
        # the methods that are required by Python in order for it to be an
        # iterator. These methods are "__iter__" and "__next__". More details:
        # https://docs.python.org/3/glossary.html#term-iterator
        # https://docs.python.org/3/library/stdtypes.html#iterator-types
        
        # Local utility class that models an iterator object of our linked list.
        # We only need this class in this "__iter__" method, so it makes sense
        # to restrict its usage by modeling it as a local class.
        class LinkedListIterator:
            
            # Specify where our iterator should start the iteration.
            def __init__(self, start: Node):
                self.curr = start
            
            # Here, we simply return the iterator itself. Specification:
            # https://docs.python.org/3/library/stdtypes.html#iterator.__iter__
            def __iter__(self):
                return self
            
            # This is the actually interesting method, which always returns the
            # next element (or raises a "StopIteration" if there are no more
            # elements left). Specification:
            # https://docs.python.org/3/library/stdtypes.html#iterator.__next__
            #
            # Note that we do not return the list "Node" objects (they are not
            # interesting to the user, as they are just part of our list
            # implementation) but just the stored value.
            def __next__(self):
                if self.curr is None:
                    raise StopIteration
                val = self.curr.val
                self.curr = self.curr.next
                return val
        
        # As per the specification, always create a new iterator object, so
        # repeated calls to "__iter__" always return an iterator that starts
        # from the beginning.
        return LinkedListIterator(self._head)


# With the above provided implementation, we can now utilize Python's iteration:
ll = LinkedList()
ll.prepend(4)
ll.prepend(7)
ll.prepend(1)
# The for loop will invoke "it = LinkedList.__iter__(self=ll)" and then make
# repeated calls to "it.__next()__" until "StopIteration" exception is raised.
for elem in ll:
    print(elem)

# Another nice thing of our "LinkedList" being iterable is the fact that we can
# now also use the "in" operation out of the box to check for membership,
# because Python can now simply iterate through the list and check if it
# contains the specified value.
print(7 in ll)
# Therefore, we do not need to implement the special method "__contains__",
# which would have otherwise been required, since we already have an "__iter__"
# implementation that is automatically used instead. However, we could still
# implement "__contains__" as well, e.g., for performance reasons if we had a
# sorted list and thus could potentially cancel the search early. Specification:
# https://docs.python.org/3/reference/datamodel.html#object.__contains__

# Another note: Iteration in Python is also possible when implementing the
# method "__getitem__", as long as the implementation appropriately raises an
# "IndexError" in case of out-of-bounds indices (see the specification). Python
# searches for the "__iter__" method, and if it is not found, it tries to fall
# back to the "__getitem__" method. For more details, see
# https://docs.python.org/3/glossary.html#term-iterable
# Analogous to "__contains__", this may have performance implications. For
# instance, if we had a "__getitem__" method in our linked list example, we
# would definitely not want the iteration to be done with this method (for every
# requested item, we would need to repeatedly traverse the list up to this item)
# but rather with a dedicated iterator object (like we did with our
# "LinkedListIterator") that implements a fast iteration behavior.
