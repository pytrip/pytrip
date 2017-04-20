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
TODO: documentation here.
"""
import os
import shutil
import tarfile
import copy

import numpy as np
from subprocess import Popen, PIPE

try:
    import paramiko
except:
    pass

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.vdx import VdxCube
##import pytriplib
import uuid
import logging

logger = logging.getLogger(__name__)


class Execute():
    def __init__(self, ctx, vdx):
        """ Initialize the Execute class.
        :params CtxCube() ctx: the CT images as a regular pytrip.CtxCube() object.
        If the CtxCube() obejcts are constant, this object can be used to execute multiple
        plans in parallel.
        """

        logger.debug("Initializing TripExecuter()")
        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this execute object
        self.ctx = ctx
        self.vdx = vdx
        ##self.rbe = rbe  # maybe move this to plan level?
        self.listeners = []
        self.trip_bin_path = "TRiP98"  # where TRiP98 is installed, if not accessible in /usr/local/bin or similar
        self.logfile_stdout = "trip98.stdout"
        self.logfile_stderr = "trip98.stderr"
        self.rsakey_local_path = "~/.ssh/id_rsa"
        self._runtrip = True  # set this to False for a dry run without TRiP98 (for testing purposes)

        # remote directory where a new temporary directory will be created, the package extracted and executed.
        self.remote_base_dir = "./"

        ##def delete_workdir(self):
        ##shutil.rmtree(self.path)

    def execute(self, plan, _callback=None):
        """
        Executes the Plan() object using TRiP98.
        """
        logger.debug("Execute TRiP98...")
        self._callback = _callback  # TODO: check if GUI really needs this.
        self._pre_execute(plan)  # prepare directory where all will be run, put files in it.
        plan.save_exec(plan._exec_path)  # add the .exec as well
        self._run_trip(plan) # run TRiP
        
        ##self._finish(plan)

    def _pre_execute(self, plan):
        """
        Prepare a temporary working directory where TRiP will be executed.
        Sets:
        _temp_dir where all will be executed
        _exec_path generates full path to the file name.exec file will be stored
        
        """
        # local plan name will not contain any spaces, and consists of a base name only, with no suffix.
        plan._basename = plan.basename.replace(" ", "_")

        # do not save working dir to self, this we may run multiple plans in the same Execute object.
        # instead we will save it to the plan.
        plan._working_dir = os.path.expandvars(plan.working_dir)

        # we will not run in working dir, but in an isolated sub directory which will be
        # uniquely created for this purpose, located after the working dir
        # the directory where the package is prepare (and TRiP98 will be executed if locally)
        # is stored in plan._temp_dir

        if not hasattr(plan, "_temp_dir"):
            import tempfile
            plan._temp_dir = tempfile.mkdtemp(prefix='trip98_', dir=plan._working_dir)

        plan._temp_dir = os.path.join(plan._working_dir, plan._temp_dir)
        plan._exec_path = os.path.join(plan._temp_dir, plan.basename + ".exec")

        logger.debug("Created temporary working directory {:s}".format(plan._temp_dir))

        _flist = []
        
        if plan.incube_basename:
            _flist.append(os.path.join(plan._working_dir, plan.incube_basename + ".dos"))
            _flist.append(os.path.join(plan._working_dir, plan.incube_basename + ".hed"))
            
        for _field in plan.fields:
            if _field.use_raster_file:
                _flist.append(os.path.join(plan._working_dir, _field.basename + ".rst"))

        for _fn in _flist:
            logger.debug("Copy {:s} to {:s}".format(_fn, plan._temp_dir))
            shutil.copy(_fn, plan._temp_dir)

        self.ctx.write(os.path.join(plan._temp_dir, self.ctx.basename + ".ctx"))  # will also make the hed
        self.vdx.write(os.path.join(plan._temp_dir, self.vdx.basename + ".vdx"))

    def add_log_listener(self, listener):
        """ A listener is something which has a .write(txt) method.
        """
        self.listeners.append(listener)

    def log(self, txt):
        """
        """
        txt = txt.replace("\n", "")
        for l in self.listeners:
            l.write(txt)

    def _run_trip(self, plan, _dir = ""):
        """ Method for executing the attached exec.
        :params str dir: overrides dir where the package is assumed to be
        """
        if not _dir:
            _dir = plan._temp_dir
        if plan.remote:
            self._run_trip_remote(plan, _dir)
        else:
            self._run_trip_local(plan, _dir)

    def _run_trip_local(self, plan,_run_dir):
        """
        Runs TRiP98 on local computer.
        """
        logger.info("Run TRiP98 in LOCAL mode.")

        _outfile = os.path.join(_run_dir, "out.txt")
        logger.debug("Write stdout and stderr to {:s}".format(_outfile))
        fp_stdout = open((_outfile), "w")
        fp_stderr = open((_outfile), "w")

        os.chdir(_run_dir)

        if not self._runtrip:  # for testing
            norun = "echo "
        else:
            norun = ""

        # start local process running TRiP98
        p = Popen([norun + self.trip_bin_path], stdout=PIPE, stdin=PIPE)

        # fill standard input with configuration file content
        # wait until process is finished and get standard output and error streams
        stdout, stderr = p.communicate(plan._trip_exec.encode("ascii"))

        if stdout is not None:
            logger.debug("Local answer stdout:" + stdout.decode("ascii"))
            fp_stdout.write(stdout.decode("ascii"))
            self.log(stdout.decode("ascii"))

        if stderr is not None:
            logger.debug("Local answer stderr:" + stderr.decode("ascii"))
            fp_stderr.write(stderr.decode("ascii"))

        os.chdir('..')

        fp_stdout.close()
        fp_stderr.close()
            
    def _run_trip_remote(self, plan, basedir = "."):
        """ Method for executing the attached plan remotely.
        :params str basedir: place where PyTRiP will mess around. Do not keep things here which should not be deleted.
        """
        logger.info("Run TRiP98 in REMOTE mode.")

        self._compress_files(plan._temp_dir)  # make a tarball out of the TRiP98 package
        self._copy_files_to_server()

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # If no password is supplied, try to look for a private key
        if self.plan.get_password() is "" or None:
            rsa_keypath = os.path.expanduser(self.rsakey_local_path)
            if not os.path.isfile(rsa_keypath):
                # login with provided username + empty password
                try:
                    ssh.connect(self.plan.server, username=self.plan.username, password="")
                except:
                    logger.error("Cannot connect to " + self.plan.server)
                    logger.error("Check username, password or key in " + self.rsakey_local_path)
                    raise
            else:
                # login with provided username + private key
                rsa_key = paramiko.RSAKey.from_private_key_file(rsa_keypath)
                try:
                    ssh.connect(self.plan.server, username=self.plan.username, pkey=rsa_key)
                except:
                    logger.error("Cannot connect to " + self.plan.server)
                    logger.error("Check username and your key in " + self.rsakey_local_path)
                    raise
        else:
            # login with provided username + password
            ssh.connect(self.plan.server, username=self.plan.username, password=self.plan.password)

        if not self._runtrip:
            norun = "echo "
        else:
            norun = ""

        ##TODO: some remote temp-dir creation mechanism is needed here.

        tgz_filename = "{:s}.tar.gz".format(self.plan.basename)
        remote_tgz_path = os.path.join(basedir, tgz_filename)  # remote place where temp.tar.gz is stored
        remote_run_dir = os.path.join(basedir, self.plan.basename)  # remote run dir

        commands = ["cd " + basedir + ";" + "tar -zxvf " + remote_tgz_path,
                    "cd " + remote_run_dir + ";" + norun + "bash -lc '" + self.trip_bin_path + " < plan.exec '",
                    "cd " + basedir + ";" + "tar -zcvf " + remote_tgz_path + " " + remote_run_dir,
                    "cd " + basedir + ";" + "rm -r " + remote_run_dir]

        # local dirs where stdout/err will be written to
        ## TODO: pass _outfile in by argument.
        _outfile = self.plan._temp_dir
        logger.debug("Write stdout and stderr to {:s}".format(_outfile))
        fp_stdout = open((_outfile), "w")
        fp_stderr = open((_outfile), "w")

        for cmd in commands:
            logger.debug("Execute on remote server: {:s}".format(cmd))
            self.log(cmd)
            stdin, stdout, stderr = ssh.exec_command(cmd)
            answer_stdout = stdout.read()
            answer_stderr = stderr.read()
            logger.debug("Remote answer stdout: {:s}".format(answer_stdout))
            logger.debug("Remote answer stderr: {:s}".format(answer_stderr))
            fp_stdout.write(answer_stdout)
            fp_stderr.write(answer_stderr)
            self.log(answer_stdout)
        ssh.close()
        fp_stdout.close()
        fp_stderr.close()

        self._copy_back_from_server()
        self._extract_tarball()

    def _finish(self):
        # return requested results, copy them back in to plan.working_dir
        
        pass

    def _compress_files(_source_dir, _target_path = None):
        """
        Builds the tar.gz from what is found in _source_dir.
        Resulting file will be stored in "source_dir/.." if not specified otherwise

        :params str _source_dir: path to dir with the files to be compressed.
        :params str _target_file: path to file, if None, then the resulting tarball will be <_source_dir>.tar.gz

        """

        if _target_path is None:
            _dir = os.path.dirname(_source_dir)
            # _basedir, _basename = os.path.split(_dir)
            _target_path = _dir + ".tar.gz"

        logger.debug("Compressing files in {:s} to {:s}".format(_source_dir, _target_path))

        with tarfile.open(os.path.join(_target_path), "w:gz") as tar:
            tar.add(_source_dir, arcname=_target_path)

    def set_plan(self, plan):
        self.plan = plan
        self.plan_name = self.plan.get_name().replace(" ", "_")

    def save_exec(self, path):
        """
        Writes the .exec script to disk.
        """
        with open(path, "wb+") as fp:
            fp.write(self.trip_exec)

    def save_data(self, path):
        """ Saves the attached CTX and VDX files to proper place
        :params path: full path to filenames, but without suffix.
        ##TODO: we need a term for a full path without suffix.
        """
        out_path = path
        ctx = self.images
        ctx.patient_name = self.plan_name
        ctx.write(os.path.join(out_path + ".ctx"))
        structures = VdxCube(ctx)
        structures.version = "2.0"
        for voi in self.plan.get_vois():
            voxelplan_voi = voi.get_voi().get_voi_data()
            structures.add_voi(voxelplan_voi)
            if voi.is_target():
                voxelplan_voi.type = '1'
            else:
                voxelplan_voi.type = '0'
        structures.write_trip(out_path + ".vdx")

    def get_transport(self):
        transport = paramiko.Transport((self.plan.get_server(), 22))
        transport.connect(username=self.plan.get_username(), password=self.plan.get_password())
        return transport

    def _copy_files_to_server(self):
        """
        Copies the generated tar.gz file to the remote server.
        """
        logger.debug("Copy tar.gz to server:" + self.path + ".tar.gz -> " +
                     os.path.join(self.remote_dir, "temp.tar.gz"))
        transport = self.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(localpath=self.path + ".tar.gz",
                 remotepath=os.path.join(self.remote_dir, 'temp.tar.gz'))
        sftp.close()
        transport.close()

    ##def _run_ssh_command(self, cmd):
    ##    ssh = paramiko.SSHClient()
    ##    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ##    ssh.connect(self.plan.get_server(), username=self.plan.get_username(), password=self.plan.get_password())
    ##    self.parent.write_to_log("Run Trip\n")
    ##    stdin, stdout, stderr = ssh.exec_command(cmd)
    ##    ssh.close()

    def _copy_back_from_server(self):
        transport = self.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(os.path.join(self.remote_dir, 'temp.tar.gz'), self.path + ".tar.gz")
        sftp.close()
        transport.close()

    def _extract_tarball():
        """ Extracts a tarball with the name self.path + ".tar.gz"
        into self.working_dir
        """
        output_folder = self.path
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        with tarfile.open(self.path + ".tar.gz", "r:gz") as tar:
            tar.extractall(self.working_dir)

    def _clean_up(self):
        """ Remove tarball and the extracted directory
        """
        f = "%s" % (self.path + ".tar.gz")
        if os.path.exists(f):
            os.remove(f)
        shutil.rmtree(self.path)
