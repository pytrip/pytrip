#
#    Copyright (C) 2010-2021 PyTRiP98 Developers.
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
"""This file contains the ExecutorLogger classes which are used to log execution logs"""

import logging

logger = logging.getLogger(__name__)


class ExecutorLogger:
    def info(self, text):
        pass  # interface method

    def log(self, text):
        pass  # interface method

    def error(self, text):
        pass  # interface method

    def end(self):
        pass  # interface method


class ConsoleExecutorLogger(ExecutorLogger):
    def info(self, text):
        print(text)

    def log(self, text):
        print(text)

    def error(self, text):
        print(text)


class FileExecutorLogger(ExecutorLogger):
    def __init__(self, stdout_path, stderr_path):
        logger.debug("Write stdout to {:s}".format(stdout_path))
        logger.debug("Write stderr to {:s}".format(stderr_path))
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.stdout = ""
        self.stderr = ""

    def info(self, text):
        self.stdout += text + "\n"

    def log(self, text):
        self.stdout += text + "\n"

    def error(self, text):
        if self.stdout_path != self.stderr_path:
            self.stderr += text + "\n"
        else:
            self.stdout += text + "\n"

    def end(self):
        with open(self.stdout_path, "w") as file:
            file.write(self.stdout)

        if self.stdout_path != self.stderr_path:
            with open(self.stderr_path, "w") as file:
                file.write(self.stderr)
