# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 20.07.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn about syntax, comments and docstrings in Python,
common Python variables, variable operations and data type conversions, and
about printing output to the console as well as reading input from it.

For more details on Python functionality in general, see the official
documentation and tutorial:
https://docs.python.org/3/
https://docs.python.org/3/tutorial/index.html
https://docs.python.org/3/tutorial/introduction.html
"""

################################################################################
# Docstrings
################################################################################

# Python offers the option for a description at the beginning of each file,
# function or class which is called "docstring" and is enclosed in three double
# quotes. What you can see above is such a docstring. Docstrings are for
# documentation purposes and are not executed as Python code, i.e., they don't
# do anything.

# The content in docstrings can range from short notes to extensive
# documentation with input/output arguments etc. It will automatically be parsed
# (if possible) and displayed in the help for the specific function or class. In
# PyCharm, you can toggle this help by left-clicking on a function or class and
# pressing Ctrl+q or selecting "Quick Documentation" in the "View" menu bar.

# There are different conventions/styles on how to write docstrings. If you are
# working on an existing project, make sure to use the same style. If you set up
# your own project, choose any style you want and then stay consistent. The
# general conventions for docstrings can be found here:
# https://www.python.org/dev/peps/pep-0257/


################################################################################
# Comments
################################################################################

# Comments are part of a Python file that are not executed, i.e., they don't do
# anything. Comments in Python start with a hashtag (#). You can also place
# comments after non-comment statements using a hashtag. Since this line starts
# with a character, it is a comment.

# There are conventions and advices regarding code style in Python, such as
# https://www.python.org/dev/peps/pep-0008/. It will help you a lot to keep your
# code readable and clean, however, Python does not explicitly force you to
# follow it. We will stick to this style as good as possible in this lecture.
# PyCharm tries to highlight code that goes against PEP standards.


################################################################################
# General
################################################################################

# Python program code is stored in text files. The standard filename suffix
# indicating a Python file is ".py", e.g., "my_file.py".

# A Python code file is executed line by line (from top to bottom of the file).
# Python evaluates expressions from left to right, but when evaluating an
# assignment, the right-hand side is evaluated before the left-hand side.
# https://docs.python.org/3/reference/expressions.html#evaluation-order
# https://docs.python.org/3/reference/expressions.html#operator-precedence


################################################################################
# Variables
################################################################################

#
# General
#

# Variables in Python are just references to objects stored and generated
# automatically in the background, i.e., they are essentially only names or
# identifiers that reference objects - they themselves do not store any
# information. Variable names must start with characters that are not digits and
# not operators. Python variables are case-sensitive. Use all lowercase names
# with underscores for variables.

# In the next line, a variable "var" is created that points to a string object
# with content "hello, world!", which is created and stored somewhere in the
# background. This is done via the assignment operator "=", where the variable
# name is on the left and the content to assign is on the right side of the
# equal sign.
var = "hello, world!"

# We can change the type and content of a variable at any time, since the
# variable is just a reference to some object stored in the background. We can,
# e.g., let it point to the integer object 5 now:
var = 5

# You can remove a variable with the del statement, but the storage/memory it
# points to will only be freed some time after no more variables are pointing
# there. This happens automatically via the "garbage collector" that removes
# objects from memory that are no no longer referenced by anything (garbage).
del var

#
# Error messages
#

# Variable names must not start with digits. Try to run this line in your Python
# console without the hashtag:
# 1var = 5
# You will see that you receive an error message (SyntaxError). Such error
# messages can help you a lot to find out what went wrong. Try to get familiar
# with reading and using them.

#
# Memory usage
#

# A variable pointing to an object takes up 16+x bytes (in native Python) for a
# 64bit Python3 installation. This results from:
# (type pointer (8 bytes) + refcount (8 bytes) + object bytes (x bytes))
# = 16+x bytes
# So we pay for convenient code by inefficient memory usage because we have a
# 16 bytes overhead for every (CPython) object, in addition to the x bytes it
# takes to store the object itself. However, you can write memory efficient code
# in Python if you use one of the many Python packages we will see later.


################################################################################
# Data types (1)
################################################################################

#
# Booleans ("bool")
#

# Booleans can take the values True or False. They are used for binary
# information, like checking whether a condition is fulfilled or not.

# Here, we create a boolean variable with content True
true_variable = True
# Here, we create another boolean variable with content False
false_variable = False

# The "not" keyword can be used to negate a boolean
another_true_variable = not false_variable

# Booleans in native Python3 are stored like integers (see below), where
# True has the value 1 and False has the value 0.

#
# Integer numbers ("int")
#

# Integer numbers can be created using digits.

# Here, we create a variable that points to an integer object 5.
an_integer = 5

# Integers in native Python3 are variable-length objects. This means that you do
# not have to worry about overflow problems with integers with large values. In
# practice, a base integer using 4 bytes (=32bits) is stored with an additional
# 8 bytes counter. One integer can use multiple base integer memories and the
# counter counts how many base integer memories are used. For an integer, Python
# would use 12 bytes and allocate more memory as needed. Combined with the
# 16 bytes overhead, this results in 28 bytes. The integer 0 is an exception
# that only uses the 8 bytes counter and no 4 bytes base memory.
# Note: This allocation scheme will not be important for us. All that matters is
# that we do not have to worry about overflow problems and that we are aware
# that our memory usage might be inefficient.

#
# Floating point numbers ("float")
#

# Floating point numbers can be created using digits combined with a dot.

# Here, we create a variable that points to a float object with value 3.563
a_float = 3.563

# Alternatively, we can use the character "e" between digits to indicate decimal
# powers for float numbers. Syntax: multiplier e decimal_power
another_float = 2e1  # Corresponds to 2 * 10^1 = 20.0
yet_another_float = 1e-3  # Corresponds to 1 * 10^-3 = 0.001

# Floating point numbers in native Python3 are stored as 8 bytes float (=64bits
# float). For storing a CPython float, Python3 would use 8 bytes. Combined with
# the 16 bytes overhead, this results in 24 bytes in total. Keep in mind that
# floating point values are stored with a fixed number of bits and therefore can
# lose precision.

#
# Making numbers more readable
#

# One can use underscore characters in numbers for readability:
large_int = 123_456_789
large_float = 123_456_789.123_456_789


################################################################################
# Mathematical operations
################################################################################

# Python supports all sorts of mathematical operations using the operators
# + for addition
# - for subtraction
# * for multiplication
# / for division
# ** for power of (syntax: a ** b corresponds to "a" to the power of "b")
# % modulo operation (get the remainder of a division)
# // get the floor value of a division (integer division)

# Addition of ints produces an int
sum_integer = an_integer + an_integer

# Addition of floats produces a float
sum_float = a_float + a_float

# Combining ints and floats produces a float
combination = an_integer + a_float

# Mathematical operations can be combined arbitrarily (common rules for order of
# mathematical operations, including parentheses, apply)
combination = ((an_integer + a_float * an_integer) / a_float) ** 5

# You can use the syntax shortcuts +=, -=, etc. if you want to modify a variable
combination += 4  # Identical to: combination = combination + 4
combination /= 2.5
combination **= 2

# If you combine boolean and other numbers, True will be 1 and False will be 0
combination -= True

# You can break long statements into multiple lines with a backslash "\"
combination += 4 + 6 - \
    3 + 5
# Or you can just put them into parentheses and have linebreaks
combination -= (1 + 2 +
                3)


################################################################################
# Data types (2)
################################################################################

#
# Strings ("str")
#

# A string is an immutable object consisting of characters. "Immutable" means
# that if you modify a string, you actually create a completely new one and will
# use up memory for both strings.
a_string = "I am a string"

# Strings can be created using double quotes, single quotes, three double
# quotes, or three single quotes. The outer quote is always the relevant one.
a_string = "I am a string"
a_string = 'I am also a string'
a_string = 'You can use the other quotes " inside a string'
a_string = "... and the other way around: ' "
a_string = "Line one.\nLine 2.\nLine 3."  # "\n" is the newline character

# 3 quotes may include linebreaks (adds newline character "\n" automatically)
a_string = """I am a string...
and I include a linebreak
as I span multiple lines."""

# You can escape effects of quotes by using the escape character backslash "\"
a_string = 'I want to use " and \' in a string without ending it.'

# If you want to use backslashes in strings, you have to escape their own escape
# function, i.e., use a double backslash:
windows_path = "C:\\Windows\\System32"

# Alternatively, you can use a "raw" string by putting an "r" in front of the
# first quote to ignore the special meaning of characters:
windows_path = r"C:\Windows\System32"
# Note: If a backslash "\" is used as last character of a raw string, it has an
# additional special meaning and will escape the last quote character that
# would end the string. r"C:\Windows\System32\"" would be equivalent to
# r'C:\Windows\System32"'. If you want to use a "\" as last character, you will
# have to escape the special meaning, even in raw strings: r"C:\Windows\\".

#
# There are multiple convenience operations on strings:
#

# Concatenation
conc_strings = "I am " + "3 concatenated" + ' strings!'
conc_strings += " Plus one more!"
automatic_concatenation = "This will result in a single string as " \
                          "if it were " "manually concatenated"

# Case transformation (here, we actually used an advanced concept called
# methods, which we will learn later)
uppercase_string = conc_strings.upper()
lowercase_string = "SILENCE!".lower()

# Repetition
repeated_string = "bla" * 3

# This will not work but try it and try to understand the error message:
# not_possible = "bla" / 3

# Length (the used concept here is a function, which we will learn later)
string_length = len(conc_strings)

# Getting the i-th character of a string via "some_string[i]", where the index
# "i" must be in the range [0, len(some_string)). We will learn more about
# indexing later.
some_string = "hello"
char = some_string[0]  # Character at index 0 = first character = "h"
# However, since strings are immutable, assignment is not possible:
# some_string[0] = "a"

# Counting substrings
number_of_bla = repeated_string.count("bla")

# Splitting or joining strings (from and to so-called lists, which we will learn
# later, again in combination with methods)
separated_elements = "my-string-elements".split("-")
joined_elements = ";".join(separated_elements)

# Replacing and finding substrings (keep in mind that this creates a
# completely new string in the background! again: methods are used here)
replacement = "This is a placeholder".replace("placeholder", "string")
check_for_substring = "string" in conc_strings
locate_substring = replacement.find("is a")

# More common string methods:
# http://docs.python.org/3/library/stdtypes.html#string-methods

# You can include all kinds of objects in strings with f"{variable_name}" using
# formatted strings. Formatted strings start with an "f" character
a = 1
b = 2.2
c = a + b
string = f"We calculated c = a + b = {a} + {b} = {c}"

# There are many types of formatting options available, like
long_number = 100 / 3
formatted_string = "You can format floating point values as :width.precision " \
                   f"{long_number:10.5f}"
# You can even use variables, or expressions in general, in the options:
n_decimals = 3
another_formatted_string = f"{long_number:.{n_decimals}f}"

# For more information on string formatting and the available options, see
# https://docs.python.org/3/library/string.html#formatstrings

# Strings can be indexed and sliced like lists, which we learn later.

#
# None
#

# None is the sole value of the data type "NoneType" in Python. It typically
# represents the absence of a value. This will play a role later, when, e.g., a
# function does not return anything.
a = None


################################################################################
# Data type conversions
################################################################################

# If you want to convert an object to another data type, be careful since you
# might lose information! You can convert to data types like this:
val = 2.9
# Conversion of "val" to float (it actually already was a float here):
as_float = float(val)
# Conversion of "val" to int. Note how we lose the information after the
# floating point (it is not rounded to the nearest integer!):
as_int = int(val)
# Conversion of "val" to bool. 0 translates to False, everything else to True:
as_bool = bool(val)
# Conversion of "val" to str. Python will do its best to convert meaningfully:
as_string = str(val)
# NoneType cannot be converted to int or float, but can be "converted" to str:
# not_possible = int(None)
none_string = str(None)


################################################################################
# Details on evaluation order
################################################################################

# Multiple assignments in one line are possible. Assignments are performed from
# right to left. The following line will create "var3" with content 8, then
# "var2" with content of "var3" and then "var1" with content of "var2":
var1 = var2 = var3 = 2 * 4


################################################################################
# Console input and output
################################################################################

# The console is the primary and default input and output device. To create an
# output, we make use of the built-in function "print", which will print the
# specified object in its string representation to the console. By default, a
# newline character "\n" is automatically added at the end.
print("Hello, world!")  # This is already a string
print(123.456)  # Converts the float via "str" (see data type conversion above)

# Multiple objects can also be passed to "print", which will then be printed in
# sequential order, separated by a whitespace:
print("this", "will", "be", "separated", "by", 1, "whitespace")

# User input from the console can be retrieved with the function "input", which
# will read the input as string, i.e., appropriate data type conversion must be
# done manually afterwards.
my_input = input("Please enter something: ")  # "my_input" will be a string
# Here, we again use a Python built-in called "type" that returns the data type
# of the object we passed (you can ignore the "__name__")
print(f"data type of 'my_input': {type(my_input).__name__}")
