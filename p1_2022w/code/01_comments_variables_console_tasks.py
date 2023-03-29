# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Van Quoc Phuong Huynh, Andreas Schörgenhumer
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

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

#
# Task 1
#

# Create a variable "a" with the content 1 and a variable "b" with the content
# False.

# Your code here #


#
# Task 2
#

# Convert the following variable "a" into an integer. Take a look at the
# variable in the debugger. What will happen to the floating point number if it
# is converted to an integer?
a = 5.356

# Your code here #


#
# Task 3
#

# Create a variable "a" with content "test" and a variable "b" that points to
# the same content as "a". Afterwards let "a" point to 2 and delete "b". What
# will happen to the string "test" after deleting variable "b"?

# Your code here #


#
# Task 4
#

# Create a variable "a" with content 123 and create a variable "b" pointing to a
# string that consists of the sentence "this is a test " followed by the
# contents of variable "a". Finally, create a variable "c" that repeats this
# sentence 3 times.

# Your code here #


#
# Task 5
#

# Take a look at the following lines of code. Why does variable "a" have a
# different value than variable "b"? Which of the two version should be used if
# you need the precise value?
a = 123456789123456789
b = int(123456789123456789.0)
print(f"value a: {a}")
print(f"value b: {b}")

# No code required for the solution of this task, only an explanation #


#
# Task 6
#

# What are the data types of values referenced by variables "a" and "b"? What is
# the role of "_" among the digits? Examine the syntax of the formatting string
# for the content of "a" and "b".
a = 123_456_789.123_456_789
b = 123_456_789
print(f"a: {a}, formatted a: {a:,.4f}")
print(f"b: {b}, formatted b: {b:,d}")

# No code required for the solution of this task, only an explanation #


#
# Task 7
#

# You are given the time period of 1234567.789 total seconds. Print this value
# in the following format:
# "Time period: <int> Days, <int> Hours, <int> Minutes, <float with 2 decimal places> Seconds"
# Hint: You can solve this task by looking at remainders of divisions.
total_seconds = 1_234_567.789

# Your code here #


#
# Task 8
#

# Read a number from the console (you can assume correct user inputs), convert
# it to float and print the result with 2 decimal places.

# Your code here #
