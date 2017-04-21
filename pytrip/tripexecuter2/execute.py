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

from subprocess import Popen, PIPE

try:
    import paramiko
except:
    pass

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.vdx import VdxCube
from pytrip.ctx import CtxCube
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
        self._norun = False  # set this to False for a dry run without TRiP98 (for testing purposes)

        # remote planning
        # remote directory where a new temporary directory will be created, the package extracted and executed.
        self.remote = False  # remote or local execution
        self.servername = ""
        self.username = ""
        self.password = ""
        self.remote_base_dir = "./"

    def __str__(self):
        return self._print()
        
    def _print(self):
        """ Pretty print current attributes.
        """
        out = ""
        out += "|\n"
        out += "| Remote access\n"
        out += "|   Remote execution            : {:s}\n".format(str(self.remote))
        out += "|   Server                      : '{:s}'\n".format(self.servername)
        out += "|   Username                    : '{:s}'\n".format(self.username)
        out += "|   Password                    : '{:s}'\n".format("*" * len(self.password))

        return out

    def execute(self, plan, _callback=None):
        """
        Executes the Plan() object using TRiP98.
        """
        logger.debug("Execute TRiP98...")
        self._callback = _callback  # TODO: check if GUI really needs this.
        self._pre_execute(plan)  # prepare directory where all will be run, put files in it.
        plan.save_exec(plan._exec_path)  # add the .exec as well
        self._run_trip(plan)  # run TRiP
        self._finish(plan)

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

    def _run_trip(self, plan, _dir=""):
        """ Method for executing the attached exec.
        :params str dir: overrides dir where the TRiP package is assumed to be
        """
        if not _dir:
            _dir = plan._temp_dir
        if self.remote:
            self._run_trip_remote(plan, _dir)
        else:
            self._run_trip_local(plan, _dir)

    def _run_trip_local(self, plan, _run_dir):
        """
        Runs TRiP98 on local computer.
        """
        logger.info("Run TRiP98 in LOCAL mode.")

        _outfile = os.path.join(_run_dir, "out.txt")
        logger.debug("Write stdout and stderr to {:s}".format(_outfile))
        fp_stdout = open((_outfile), "w")
        fp_stderr = open((_outfile), "w")

        os.chdir(_run_dir)

        if self._norun:  # for testing
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

    def _run_trip_remote(self, plan, _run_dir=None):
        """ Method for executing the attached plan remotely.
        :params Plan plan: plan object
        :params str run_dir:  
        """
        logger.info("Run TRiP98 in REMOTE mode.")

        print("temp_dir: {:s}".format(plan._temp_dir))

        tar_path = self._compress_files(plan._temp_dir)  # make a tarball out of the TRiP98 package
        self._copy_files_to_server(tar_path)

        ssh = self._get_ssh_client()

        if self._norun:
            norun = "echo "
        else:
            norun = ""

        ##TODO: some remote temp-dir creation mechanism is needed here.

        tgz_filename = "{:s}.tar.gz".format(plan.basename)
        remote_tgz_path = os.path.join(basedir, tgz_filename)  # remote place where temp.tar.gz is stored
        remote_run_dir = os.path.join(basedir, plan.basename)  # remote run dir

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

    def _finish(self, plan):
        """ return requested results, copy them back in to plan.working_dir
        """

        for _fn in plan._out_files:
            _path = os.path.join(plan._temp_dir, _fn)
            logger.debug("copy {:s} to {:s}".format(_path, plan.working_dir))
            shutil.copy(_path, plan.working_dir)

        for _fn in plan._out_files:
            _path = os.path.join(plan._temp_dir, _fn)
            if ".phys.dos" in _fn:
                _d = DosCube()
                _d.read(_path)
                plan.dosecubes.append(_d)

            if ".bio.dos" in _fn:
                _d = CtxCube()
                _d.read(_path)
                plan.dosecubes.append(_d)

            if ".dosemlet.dos" in _fn:
                _l = LETCube()
                _l.read(_path)
                plan.letcubes.append(_l)

            if ".rst" in _fn:
                print("NB:", _path)
                # os.path.basename(_fn.=

    @staticmethod
    def _compress_files(_source_dir):
        """
        Builds the tar.gz from what is found in _source_dir.
        Resulting file will be stored in "source_dir/.." if not specified otherwise

        :params str _source_dir: path to dir with the files to be compressed.
        :returns: full path to tar.gz file.
        """

        # _source_dir may be of either "/foo/bar" or "/foo/bar/". We only need "bar",
        # but dirname() will return foo in the first case and bar in the second,
        # fix this by adding an extra seperator, in that case all will be stripped.
        # add an extra trailing slash, if there are multiple 
        _dir = os.path.dirname(_source_dir + os.sep)
        _pardir, _basedir = os.path.split(_dir)
        _target_filename = _basedir + ".tar.gz"

        _cwd = os.getcwd()
        os.chdir(_pardir)

        # _basedir, _basename = os.path.split(_dir)

        logger.debug("Compressing files in {:s} to {:s}".format(_source_dir, _target_filename))

        with tarfile.open(_target_filename, "w:gz") as tar:
            tar.add(_source_dir, arcname=_basedir)

        os.chdir(_cwd)

        return os.path.join(_pardir, _target_filename)

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

    def _copy_files_to_server(self, path):
        """
        Copies the generated tar.gz file to the remote server.
        """
        _to_uri = self.servername + ":" + self.remote_base_dir

        _dir, _filename = os.path.split(path)
        
        logger.debug("Copy {:s} to {:s}".format(path, _to_uri))

        sftp, transport = self._get_sftp_client()
        sftp.put(localpath=path, remotepath=os.path.join(self.remote_base_dir, _filename))
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

    def _extract_tarball(self, plan):
        """ Extracts a tarball with the name self.path + ".tar.gz"
        into self.working_dir
        """
        output_folder = plan.path
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        with tarfile.open(self.path + ".tar.gz", "r:gz") as tar:
            tar.extractall(plan.working_dir)

    def _clean_up(self):
        """ Remove tarball and the extracted directory
        """
        f = "%s" % (self.path + ".tar.gz")
        if os.path.exists(f):
            os.remove(f)
        shutil.rmtree(self.path)

        
    def _get_ssh_client(self):
        """ returns an open ssh client
        """
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # If no password is supplied, try to look for a private key
        if self.password is "" or None:
            rsa_keypath = os.path.expanduser(self.rsakey_local_path)
            if not os.path.isfile(rsa_keypath):
                # login with provided username + empty password
                try:
                    ssh.connect(self.servername, username=self.username, password="")
                except:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username, password or key in " + self.rsakey_local_path)
                    raise
            else:
                # login with provided username + private key
                rsa_key = paramiko.RSAKey.from_private_key_file(rsa_keypath)
                try:
                    ssh.connect(self.servername, username=self.username, pkey=rsa_key)
                except:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username and your key in " + self.rsakey_local_path)
                    raise
        else:
            # login with provided username + password
            ssh.connect(self.servername, username=self.username, password=self.password)

        return ssh

    def _get_sftp_client(self):
        """ returns a sftp client object and the corresponding transport socket.
        Both must be closed after use.
        """
        transport = paramiko.Transport((self.servername, 22))
        # transport.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # If no password is supplied, try to look for a private key
        if self.password is "" or None:
            rsa_keypath = os.path.expanduser(self.rsakey_local_path)
            if not os.path.isfile(rsa_keypath):
                # login with provided username + empty password
                try:
                    transport.connect(username=self.username, password="")
                except:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username, password or key in " + self.rsakey_local_path)
                    raise
            else:
                # login with provided username + private key
                rsa_key = paramiko.RSAKey.from_private_key_file(rsa_keypath)
                print(dir(rsa_key))
                print(type(rsa_key))
                try:
                    transport.connect(username=self.username, pkey=rsa_key)
                except:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username and your key in " + self.rsakey_local_path)
                    raise
        else:
            # login with provided username + password
            transport.connect(username=self.username, password=self.password)

        sftp = paramiko.SFTPClient.from_transport(transport)

        return sftp, transport
