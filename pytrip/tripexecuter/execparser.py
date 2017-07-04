#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
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
Parser for TRiP files.
"""

import os
# import uuid
import logging

from pytrip.tripexecuter import Field


logger = logging.getLogger(__name__)


class ExecParser(object):
    """
    """

    # map trip commands onto method names
    _trip_commands = {"ct": "_parse_ct",
                      "field": "_parse_field",
                      "plan": "_parse_plan",
                      "opt": "_parse_opt",
                      "optimize": "_parse_opt",
                      "scancap": "_parse_scancap"}

    # scancap arguments. {<trip_parameter> : (<handler_method>, <format_specifier>)}
    _scancap_args = {"offh2o": ("_update_obj", "f"),
                     "bolus": ("_na", "s"),
                     "focus2stepsizefactor": ("_na", "s"),
                     "calbibration": ("_na", "s"),
                     "path": ("_na", "s"),
                     "rifi": ("_na", "s"),
                     "couchangle": ("_na", "s"),
                     "gantryangle": ("_na", "s")}

    # generic plan arguments. {<trip_parameter> : (<handler_method>, <format_specifier>)}
    _plan_args = {"dose": ("_na", "s"),
                  "targettissue": ("_na", "s"),
                  "residualtissue": ("_na", "s"),
                  "partialbiodose": ("_na", "s"),
                  "incube": ("_na", "s"),
                  "outcube": ("_na", "s"),
                  "debug": ("_na", "s")}

    # optimization arguments. {<trip_parameter> : (<handler_method>, <format_specifier>)}
    _opt_args = {"iter": ("_na", "s"),
                 "graceiter": ("_na", "s"),
                 "bio": ("_na", "s"),
                 "phys": ("_na", "s"),
                 "H2Obased": ("_na", "s"),
                 "CTbased": ("_na", "s"),
                 "singly": ("_na", "s"),
                 "matchonly": ("_na", "s"),
                 "dosealgorithm": ("_na", "s"),
                 "bioalgorithm": ("_na", "s"),
                 "optalgorithm": ("_na", "s"),
                 "events": ("_na", "s"),
                 "eps": ("_na", "s"),
                 "geps": ("_na", "s"),
                 "myfac": ("_na", "s"),
                 "doseweightfactor": ("_na", "s"),
                 "field": ("_na", "s"),
                 "debug": ("_na", "s")}

    # field specific arguments. {<trip_parameter> : (<handler_method>, <format_specifier>)}
    _field_args = {"file": ("_na", "s"),
                   "import": ("_na", "s"),
                   "export": ("_na", "s"),
                   "read": ("_na", "s"),
                   "write": ("_na", "s"),
                   "list": ("_na", "s"),
                   "delete": ("_na", "s"),
                   "display": ("_na", "s"),
                   "inspect": ("_na", "s"),
                   "reset": ("_na", "s"),
                   "new": ("_na", "s"),
                   "reverseorder": ("_na", "s"),
                   "target": ("_na", "s"),
                   "gantry": ("_na", "s"),
                   "couch": ("_na", "s"),
                   "chair": ("_na", "s"),
                   "stereotacticcoordinates": ("_na", "s"),
                   "fwhm": ("_na", "s"),
                   "rastersteps": ("_na", "s"),
                   "raster": ("_na", "s"),  # abbreviated of the above # TODO: can we implement some pointer/alias?
                   "zsteps": ("_na", "s"),
                   "beam": ("_na", "s"),
                   "weight": ("_na", "s"),
                   "contourextension": ("_na", "s"),
                   "contourext": ("_na", "s"),  # abbreviated of the above
                   "doseextension": ("_na", "s"),
                   "doseext": ("_na", "s"),  # abbreviated of the above
                   "projectile": ("_na", "s"),
                   "proj": ("_na", "s"),  # abbreviated of the above
                   "bev": ("_na", "s"),
                   "nolateral": ("_na", "s"),
                   "raw": ("_na", "s"),
                   "dosemeanlet": ("_na", "s"),
                   "algorithm": ("_na", "s"),
                   "bioalgorithm": ("_na", "s"),
                   "debug": ("_na", "s")}

    def __init__(self, plan):
        self.plan = plan

    def _parse_exec(self, path):
        """ Parse an .exec file and store it into self.plan

        :params str path: path to .exec file including file extension
        """
        self.folder = os.path.dirname(path)
        with open(path, "r") as fp:
            data = fp.read()

        data = data.split("\n")

        for line in data:
            _key = line.split(" ")[0]  # get first word of line
            if _key in self._trip_commands:
                # lookup and call corresponding method
                logger.debug("Calling self {:s}".format(self._trip_commands[_key]))
                getattr(self, self._trip_commands[_key])(line)

    @staticmethod
    def _unpack_arg(_arg):
        """ Returns the interior of the paranthesis of an argument.
        Any quotes will be stripped from the subarg.
        Any whitespaces will be stripped from the subarg or arg.

        example:
        _arg = "bolus(2.00)"

        will return

        "bolus", "2.00"

        """
        _subarg = _arg[_arg.find("(")+1:_arg.find(")")].strip("\"").strip("\'").strip()
        _arg = _arg[0:_arg.find("(")].strip()
        return _arg, _subarg

    def _parse_extra_args(self, _line, _dict, _obj):
        """
        Parse all args following the "/" in a given line and a given parsing dictionary.
        The parsing dictionary maps the TRiP parameter ( == argument) with a proper method.

        :params _obj: may either be self.plan or self.plan.fields[i]

        """
        items = _line.split("/")
        if len(items) > 1:
            _args = items[1].split()
            for _arg in _args:
                _par, _val = self._unpack_arg(_arg)  # returns always string
                # _par holds the argument/parameter name
                # _val holds the value for the argument/parameter

                # now lookup a proper method from the dict
                if _par in _dict:
                    _method = _dict[_par][0]
                    _format = _dict[_par][1]
                    # _method is the name of the method to be called (key in the dicts above)
                    # _obj is the target object (the base plan or a given field)
                    # _format is the format specifier how the _val should be stored in _par for the _obj class
                    getattr(self, _method)(_obj, _par, _val, _format)

    def _parse_ct(self, line):
        """ Parses the 'ct' command arguments
        """
        items = line.split("/")
        if len(items) > 1:
            path = items[0].split()[1]
            args = items[1].split()
            if "read" in args:
                self.plan.basename = path

    def _parse_plan(self, line):
        """
        Method for parsing plan data
        """
        self._parse_extra_args(line, self._plan_args, self.plan)

    def _parse_field(self, line):
        """
        Method for parsing field data
        """
        if "new" in line:
            field = Field()
            self.plan.fields.append(field)
            self._parse_extra_args(line, self._field_args, self.plan.fields[-1])

    def _parse_scancap(self, line):
        """
        Generic parser
        """
        self._parse_extra_args(line, self._scancap_args, self.plan)

    def _parse_opt(self, line):
        """
        Not implemented.
        """
        pass

    @staticmethod
    def _update_obj(_obj, _par, _val, _format):
        """
        :params _obj: object whose paramters will be updated
        :params _par: attribute to be set in obj
        :params _val: variable which this attribute will be set to
        """
        # Add all needed type conversions here:
        if _format == 'f':
            value = float(_val)
        elif _format == 'i':
            value = int(_val)
        else:
            value = _val
        setattr(_obj, _par, value)

    @staticmethod
    def _na(_obj, _arg1, _arg2, _format):
        """ None Applicable.
        This method simply prints a N/A warning and exits.
        """
        logger.warning("Not implmented: '{:s}={:s} format={:s}'".format(_arg1, _arg2, _format))
