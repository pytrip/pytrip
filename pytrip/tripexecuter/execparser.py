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

import logging

from pytrip.tripexecuter import Field


logger = logging.getLogger(__name__)


class ExecParser(object):
    """
    """

    # map trip commands onto method names
    _trip_commands = {"ct": "_parse_ct",
                      # TODO: "voi": "_parse_voi",
                      "field": "_parse_field",
                      "plan": "_parse_plan",
                      "opt": "_parse_opt",
                      "optimize": "_parse_opt",
                      "scancap": "_parse_scancap"}

    # Here follows a several trip parameters/arguments which can be modified.
    # All parameters which are not supported are mapped to the _na() method,
    # which does nothing but displays a warning.

    # scancap arguments. {<trip_parameter> : (<handler_method>, <format_specifier>)}
    _scancap_args = {"offh2o": ("_update_obj", "f"),
                     "bolus": ("_update_obj", "f"),
                     "focus2stepsizefactor": ("_na", "s"),            # not implemented
                     "calbibration": ("_na", "s"),                    # not implemented
                     "path": ("_update_obj", "s", "scanpath"),
                     "rifi": ("_update_obj", "f"),
                     "couchangle": ("_na", "s"),                      # not implemented
                     "gantryangle": ("_na", "s"),                     # not implemented
                     "minparticles": ("_update_obj", "i")}

    # <attribute_name> must only be added if it is different from the name of <trip_parameter>
    # generic plan arguments. {<trip_parameter> : (<handler_method>, <format_specifier>, [<attribute_name>])}
    _plan_args = {"dose": ("_update_obj", "f", "target_dose"),
                  "targettissue": ("_update_obj", "s", "target_tissue_type"),
                  "residualtissue": ("_update_obj", "s", "res_tissue_type"),
                  "partialbiodose": ("_na", "s"),                     # not implemented
                  "incube": ("_update_obj", "s", "incube_basename"),  # TODO: may need special handler for suffix
                  "outcube": ("_na", "s"),                            # not implemented
                  "debug": ("_na", "s")}                              # not implemented

    # optimization arguments. {<trip_parameter> : (<handler_method>, <format_specifier>, [<attribute_name>])}
    _opt_args = {"iter": ("_update_obj", "i", "iterations"),
                 "graceiter": ("_na", "s"),                           # not implemented
                 "bio": ("_update_obj", "s", "opt_method"),
                 "phys": ("_update_obj", "s", "opt_method"),
                 "H2Obased": ("_update_obj", "s", "opt_principle"),
                 "CTbased": ("_update_obj", "s", "opt_principle"),
                 "singly": ("_na", "s"),                              # not implemented
                 "matchonly": ("_na", "s"),                           # not implemented
                 "dosealgorithm": ("_update_obj", "s", "dose_alg"),
                 "dosealg": ("_update_obj", "s", "dose_alg"),
                 "bioalgorithm": ("_update_obj", "s", "bio_alg"),
                 "bioalg": ("_update_obj", "s", "bio_alg"),
                 "optalgorithm": ("_update_obj", "s", "opt_alg"),
                 "optalg": ("_update_obj", "s", "opt_alg"),
                 "events": ("_na", "s"),                              # not implemented
                 "eps": ("_update_obj", "f"),
                 "geps": ("_update_obj", "f"),
                 "myfac": ("_na", "s"),                               # not implemented
                 "doseweightfactor": ("_update_obj", "f", "target_dose_percent"),  # TODO: may check 10 % or 0.1
                 "field": ("_na", "s"),                               # not implemented
                 "debug": ("_na", "s")}                               # not implemented

    # field specific arguments. {<trip_parameter> : (<handler_method>, <format_specifier>, [<attribute_name>])}
    _field_args = {"file": ("_update_obj", "s", "use_raster_file"),
                   "import": ("_na", "s"),                            # not implemented
                   "export": ("_na", "s"),                            # not implemented
                   "read": ("_na", "s"),                              # not implemented
                   "write": ("_na", "s"),                             # not implemented
                   "list": ("_na", "s"),                              # not implemented
                   "delete": ("_na", "s"),                            # not implemented
                   "display": ("_na", "s"),                           # not implemented
                   "inspect": ("_na", "s"),                           # not implemented
                   "reset": ("_na", "s"),                             # not implemented
                   "new": ("_na", "s"),                               # special case
                   "reverseorder": ("_na", "s"),                      # not implemented
                   "target": ("_na", "s"),                            # not implemented
                   "gantry": ("_update_obj", "f"),
                   "couch": ("_update_obj", "f"),
                   "chair": ("_update_obj", "f"),
                   "stereotacticcoordinates": ("_na", "s"),           # not implemented
                   "fwhm": ("_update_obj", "f"),                      # not implemented
                   "rastersteps": ("_update_obj", "[f,f]", "raster_step"),
                   "raster": ("_update_obj", "[f,f]", "raster_step"),
                   "zsteps": ("_update_obj", "f"),                    # either "zsteps" (as in TRiP98 manual)..
                   "zstep": ("_update_obj", "f", "zsteps"),           # .. or "zstep" (which is used)
                   "beam": ("_na", "s"),                              # not implemented
                   "weight": ("_na", "s"),                            # not implemented
                   "contourextension": ("_update_obj", "f", "contour_extension"),
                   "contourext": ("_update_obj", "f", "contour_extension"),
                   "doseextension": ("_update_obj", "f", "dose_extension"),
                   "doseext": ("_update_obj", "f", "dose_extension"),
                   "projectile": ("_update_obj", "s"),                # TODO: needs special handling
                   "proj": ("_na", "s"),                              # abbreviated of the above
                   "bev": ("_na", "s"),                               # not implemented
                   "nolateral": ("_na", "s"),                         # not implemented
                   "raw": ("_na", "s"),                               # not implemented
                   "dosemeanlet": ("_na", "s"),                       # not implemented
                   "algorithm": ("_na", "s"),                         # not implemented (used for .bev files only)
                   "bioalgorithm": ("_na", "s"),                      # not implemented (used for .bev files only)
                   "debug": ("_na", "s")}                             # not implemented

    def __init__(self, plan):
        self.plan = plan

    def parse_exec(self, path):
        """ Parse an .exec file and store it into self.plan

        :params str path: path to .exec file including file extension
        """
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
        """ Returns the interior of the parenthesis of an argument.

        'arg(value)'     -> 'arg', 'value'
        'arg("value")'   -> 'arg', 'value'
        'arg('value')'   -> 'arg', 'value'
        'arg'            -> 'arg', ''
        '"arg"'          -> 'arg', ''
        ''arg''          -> 'arg', ''

        Any quotes will be stripped from the value.
        Any whitespaces will be stripped from the value or arg.

        If there is no value given, simply the _arg is returned and an empty string for the value.

        example:
        >>> ExecParser._unpack_arg("bolus(2.00)")
        ('bolus', '2.00')

        """
        if "(" in _arg:
            _val = _arg[_arg.find("(") + 1:_arg.find(")")].strip("\"").strip("\'").strip()
            _arg = _arg[0:_arg.find("(")].strip()
        else:
            _val = ""
            _arg = _arg.strip("\"").strip("\'").strip()
        return _arg, _val

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

                # there are two kinds of arguments:
                # Case 1) : foobar1(value)
                # Case 2) : foobar2

                _par, _val = self._unpack_arg(_arg)  # returns always string
                # _par holds the TRiP argument/parameter name
                # _val holds the value for the argument/parameter

                if not _val:
                    _val = _par

                # now lookup a proper method from the dict
                if _par in _dict:
                    _method = _dict[_par][0]
                    _format = _dict[_par][1]
                    # sometimes, the TRiP parameter names is not equal to the name of the attribute in _obj
                    # in these cases, a third string is in the dict, which holds the attribute name.
                    if len(_dict[_par]) > 2:
                        _objattr = _dict[_par][2]
                    else:
                        _objattr = _par

                    # _method is the name of the method to be called (key in the dicts above)
                    # _obj is the target object (the base plan or a given field)
                    # _format is the format specifier how the _val should be stored in _par for the _obj class
                    getattr(self, _method)(_obj, _objattr, _val, _format)

    def _parse_ct(self, line):
        """
        Parses the 'ct' command arguments
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
        items = line.split("/")
        number = int(items[0].split()[1])
        if len(items) > 1 and "new" in items[1]:
            logger.debug("Parse a new field number {:d}".format(number))
            field = Field()
            field.number = number
            self.plan.fields.append(field)
            self._parse_extra_args(line, self._field_args, self.plan.fields[-1])

    def _parse_scancap(self, line):
        """
        Parser for the scancap command.
        """
        self._parse_extra_args(line, self._scancap_args, self.plan)

    def _parse_opt(self, line):
        """
        Parser for the optimization command.
        """
        self._parse_extra_args(line, self._opt_args, self.plan)

    @staticmethod
    def _update_obj(_obj, _objattr, _val, _format):
        """
        :params _obj: object whose parameters will be updated
        :params _par: attribute to be set in obj
        :params _val: variable which this attribute will be set to
        """
        # Add all needed type conversions here:
        if _format == 'f':  # single float
            value = float(_val)

        elif _format == 'i':  # single integer
            value = int(_val)

        # handle tuples of arbitrary length
        # "2,2" -> (2,2) or (2.0,2.0) depending on _format = "(i,i)" or "(f,f)"
        elif "(" in _format:
            if "f" in _format:  # tuple will be float
                value = tuple(map(float, _val.split(",")))
            elif "i" in _format:  # tuple will be int
                value = tuple(map(int, _val.split(",")))

        # handle list of arbitrary length
        # "2,2" -> [2,2] or [2.0,2.0] depending on _format = "[i,i]" or "[f,f]"
        elif "[" in _format:
            if "f" in _format:  # tuple will be float
                value = list(map(float, _val.split(",")))
            elif "i" in _format:  # tuple will be int
                value = list(map(int, _val.split(",")))

        # single string (catch all)
        else:
            value = _val
        logger.debug("Setting obj.{:s}={:s}".format(_objattr, str(value)))
        setattr(_obj, _objattr, value)

    @staticmethod
    def _na(_obj, _arg1, _arg2, _format):
        """
        Not Applicable - used by not implemented TRiP98 arguments/parameters

        This method simply prints a N/A warning and exits.
        """
        logger.warning("Not implemented: '{:s}={:s} format={:s}'".format(_arg1, _arg2, _format))
