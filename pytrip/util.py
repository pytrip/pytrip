#
#    Copyright (C) 2010-2016 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Module with auxilliary functions (mostly internal use).
"""


def get_class_name(item):
    """
    :returns: name of class of 'item' object.
    """
    return item.__class__.__name__


def evaluator(funct, name='funct'):
    """ Wrapper for evaluating a function.

    :params str funct: string which will be parsed
    :params str name: name which will be assigned to created function.

    :returns: function f build from 'funct' input.
    """
    code = compile(funct, name, 'eval')

    def f(x):
        return eval(code, locals())

    f.__name__ = name
    return f
