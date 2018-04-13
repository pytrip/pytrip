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
This file contains the Execute() class which can execute a Plan() object locally or on a remote server
with a complete TRiP98 installations.
"""
import os
import uuid
import shutil
import tarfile
import logging

from subprocess import Popen, PIPE

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.ctx import CtxCube

logger = logging.getLogger(__name__)

try:
    import paramiko
except ImportError:
    logger.warning("Paramiko package not installed, only local TRiP access possible")


class Execute(object):
    """ Execute class for running trip using attached Ctx, Vdx and Plan objects.
    """
    def __init__(self, ctx, vdx, ctx_path="", vdx_path=""):
        """ Initialize the Execute class.
        :params CtxCube() ctx: the CT images as a regular pytrip.CtxCube() object.
        :params VdxCube() ctx: the contours as a regular pytrip.VdxCube() object.

        If the Ctx and Vdx obejcts are constant, this object can be used to execute multiple
        plans in parallel.

        pytrip will write the Ctx and Vdx cube to a temporary directory where all will
        be dealt with. However, as it is right now, PyTRiP98 has sometimes trouble
        to convert to something which is readable TRiP98.
        For this case, we provide these optional hooks, where paths to Ctx and Vdx files can be
        specified, these will then be copied directly without any internal conversion.
        :params str ctx_path: optional source of .ctx cube, will override ctx.write()
        :params str vdx_path: optional source of .vdx cube, will override vdx.write()
        """

        logger.debug("Initializing TripExecuter()")

        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this execute object

        self.ctx = ctx
        self._ctx_path = ctx_path

        self.vdx = vdx
        self._vdx_path = vdx_path

        self.listeners = []

        # TODO: this should be only the TRiP98 command, and not the full path
        # however some issue when running remote, so lets leave this for now
        self.trip_bin_path = "TRiP98"  # where TRiP98 is installed, if not accessible in /usr/local/bin or similar

        self.logfile_prefix_stdout = "trip98.stdout."  # will be suffixed by plan.basename
        self.logfile_prefix_stderr = "trip98.stderr."  # will be suffixed by plan.basename

        self._norun = False  # set this to False for a dry run without TRiP98 (for testing purposes)
        self._cleanup = True  # delete any temporary dirs created (currently this is only the package dir)

        # remote planning
        # remote directory where a new temporary directory will be created, the package extracted and executed.
        self.remote = False  # remote or local execution
        self.servername = ""
        self.username = ""
        self.password = ""
        self.remote_base_dir = "./"
        self.rsakey_local_path = "~/.ssh/id_rsa"

    def __str__(self):
        """ str output handler
        """
        return self._print()

    def _print(self):
        """ Pretty print current attributes.
        """
        out = "\n"
        out += "   Executer\n"
        out += "----------------------------------------------------------------------------\n"
        out += "| General configuration\n"
        out += "|   UUID                        : {:s}\n".format(str(self.__uuid__))
        out += "|   CtxCube.basename            : '{:s}'\n".format(self.ctx.basename)
        out += "|   VdxCube.basename            : '{:s}'\n".format(self.vdx.basename)
        out += "|   STDOUT prefix               : '{:s}'\n".format(self.logfile_prefix_stdout)
        out += "|   STDERR prefix               : '{:s}'\n".format(self.logfile_prefix_stderr)
        out += "|   TRiP98 command              : '{:s}'\n".format(self.trip_bin_path)
        out += "|   Cleanup                     : {:s}\n".format(str(self._cleanup))

        out += "|\n"
        out += "| Remote access\n"
        out += "|   Remote execution            : {:s}\n".format(str(self.remote))
        out += "|   Server                      : '{:s}'\n".format(self.servername)
        out += "|   Username                    : '{:s}'\n".format(self.username)
        out += "|   Password                    : '{:s}'\n".format("*" * len(self.password))

        return out

    def execute(self, plan, run=True, _callback=None):
        """
        Executes the Plan() object using TRiP98.

        :returns int: return code from trip execution.
        """

        if run:
            logger.debug("Execute TRiP98...")
            self._norun = False
        else:
            logger.debug("Execute TRiP98 (dry-run)...")
            self._norun = True

        self._callback = _callback  # TODO: check if GUI really needs this.
        self._pre_execute(plan)  # prepare directory where all will be run, put files in it.
        plan.save_exec(plan._exec_path)  # add the .exec as well
        rc = self._run_trip(plan)  # run TRiP
        self._finish(plan)
        return rc

    def _pre_execute(self, plan):
        """
        Prepare a temporary working directory where TRiP will be executed.
        Sets:
        _temp_dir where all will be executed
        _exec_path generates full path to the file name.exec file will be stored
        """

        # attach working dir to plan, but expanded for environment variables
        if not plan.working_dir:
            logger.warning("No working directory was specified for plan. Setting it to ./")
            plan.working_dir = "./"
        plan._working_dir = os.path.expandvars(plan.working_dir)

        # We will not run TRiP98 in working dir, but in an isolated subdirectory which will be
        # uniquely created for this purpose, located after the working dir.
        # This contents of this subdir will be termed "the package", as it is a clean environment.
        # The name of the subdir will be stored in plan._temp_dir
        # It may be safely deleted afterwards, once the resulting files are copied back to the plan.working_dir

        if not hasattr(plan, "_temp_dir"):
            import tempfile
            plan._temp_dir = tempfile.mkdtemp(prefix='trip98_', dir=plan._working_dir)

        plan._temp_dir = os.path.join(plan._working_dir,
                                      plan._temp_dir)

        plan._exec_path = os.path.join(plan._temp_dir,
                                       plan.basename + ".exec")

        logger.debug("Created temporary working directory {:s}".format(plan._temp_dir))

        _flist = []  # list of files which must be copied to the package.

        if plan.incube_basename:
            _flist.append(os.path.join(plan._working_dir, plan.incube_basename + ".dos"))
            _flist.append(os.path.join(plan._working_dir, plan.incube_basename + ".hed"))

        for _field in plan.fields:
            if _field.use_raster_file:
                _flist.append(os.path.join(plan._working_dir, _field.basename + ".rst"))

        # once the file list _flist is complete, copy it to the package location
        for _fn in _flist:
            logger.debug("Copy {:s} to {:s}".format(_fn, plan._temp_dir))
            shutil.copy(_fn, plan._temp_dir)

        # Ctx and Vdx files are not copied, but written from the objects passed to Execute() during __init__.
        # This gives the user better control over what Ctx and Vdx should be based for the planning.
        # Copying can though be forced by specifying .ctx and .vdx path during self.__init__
        # This is necessary as long as PyTRiP has trouble to produce TRiP98 readable files reliably.
        if self._ctx_path:
            _ctx_base, _ = os.path.splitext(self._ctx_path)
            logger.info("Copying {:s} to tmp dir, instead of writing from Ctx object.".format(self._ctx_path))
            shutil.copy(self._ctx_path, plan._temp_dir)  # copy .ctx
            shutil.copy(_ctx_base + ".hed", plan._temp_dir)  # copy.hed
        else:
            self.ctx.write(os.path.join(plan._temp_dir, self.ctx.basename + ".ctx"))  # will also make the hed

        if self._vdx_path:
            logger.info("Copying {:s} to tmp dir, instead of writing from Vdx object.".format(self._vdx_path))
            shutil.copy(self._vdx_path, plan._temp_dir)
        else:
            self.vdx.write(os.path.join(plan._temp_dir, self.vdx.basename + ".vdx"))

    def add_log_listener(self, listener):
        """ A listener is something which has a .write(txt) method.
        """
        self.listeners.append(listener)

    def log(self, txt):
        """ Writes txt to all listeners, stripping any newlines.
        """
        txt = txt.replace("\n", "")
        for l in self.listeners:
            l.write(txt)

    def _run_trip(self, plan, _dir=""):
        """ Method for executing the attached exec.
        :params str dir: overrides dir where the TRiP package is assumed to be, otherwise
        the package location is taken from plan._temp_dir

        :returns int: return code from trip execution.
        """
        if not _dir:
            _dir = plan._temp_dir
        if self.remote:
            rc = self._run_trip_remote(plan, _dir)
        else:
            rc = self._run_trip_local(plan, _dir)
        return rc

    def _run_trip_local(self, plan, _run_dir):
        """
        Runs TRiP98 on local computer.
        :params Plan plan : plan object to be executed
        :params str _run_dir: directory where a clean trip package is located, i.e. where TRiP98 will be run.

        :returns int: return code of TRiP98 execution. (0, even if optimiztion fails)
        """
        logger.info("Run TRiP98 in LOCAL mode.")

        trip, ver = self.test_local_trip()
        if trip is None:
            logger.error("Could not find TRiP98 using path \"{:s}\"".format(self.trip_bin_path))
            raise EnvironmentError
        else:
            logger.info("Found {:s} version {:s}".format(trip, ver))

        # stdout and stderr are always written locally
        _stdout_path = os.path.join(plan._working_dir, self.logfile_prefix_stdout + plan.basename)
        _stderr_path = os.path.join(plan._working_dir, self.logfile_prefix_stderr + plan.basename)
        logger.debug("Write stdout to {:s}".format(_stdout_path))
        logger.debug("Write stderr to {:s}".format(_stderr_path))

        fp_stdout = open(_stdout_path, "w")
        fp_stderr = open(_stderr_path, "w")

        _pwd = os.getcwd()
        os.chdir(_run_dir)

        # start local process running TRiP98
        if self._norun:  # for testing, just echo the command which would be executed
            p = Popen(["echo", self.trip_bin_path], stdout=PIPE, stdin=PIPE)
        else:
            p = Popen([self.trip_bin_path], stdout=PIPE, stdin=PIPE)

        # fill standard input with configuration file content
        # wait until process is finished and get standard output and error streams
        stdout, stderr = p.communicate(plan._trip_exec.encode("ascii"))
        rc = p.returncode
        if rc != 0:
            logger.error("TRiP98 error: return code {:d}".format(rc))
        else:
            logger.debug("TRiP98 exited with status: {:d}".format(rc))

        if stdout is not None:
            logger.debug("Local answer stdout:" + stdout.decode("ascii"))
            fp_stdout.write(stdout.decode("ascii"))
            self.log(stdout.decode("ascii"))

        if stderr is not None:
            logger.debug("Local answer stderr:" + stderr.decode("ascii"))
            fp_stderr.write(stderr.decode("ascii"))

        os.chdir(_pwd)

        fp_stdout.close()
        fp_stderr.close()

        return rc

    def _run_trip_remote(self, plan, _run_dir=None):
        """ Method for executing the attached plan remotely.
        :params Plan plan: plan object
        :params str _run_dir: TODO: not used

        :returns: integer exist staus of TRiP98 execution. (0, even if optimiztion fails)
        """
        logger.info("Run TRiP98 in REMOTE mode.")

        tar_path = self._compress_files(plan._temp_dir)  # make a tarball out of the TRiP98 package

        # prepare relevant dirs and paths
        _, tgz_filename = os.path.split(tar_path)
        remote_tgz_path = os.path.join(self.remote_base_dir, tgz_filename)
        local_tgz_path = os.path.join(plan._working_dir, tgz_filename)
        remote_run_dir = os.path.join(self.remote_base_dir, tgz_filename.rstrip(".tar.gz"))  # TODO: os.path.splitext
        remote_exec_fn = plan.basename + ".exec"
        remote_rel_run_dir = tgz_filename.rstrip(".tar.gz")  # TODO: os.path.splitext

        # stdout and stderr are always written locally
        _stdout_path = os.path.join(plan._working_dir, self.logfile_prefix_stdout + plan.basename)
        _stderr_path = os.path.join(plan._working_dir, self.logfile_prefix_stderr + plan.basename)
        logger.debug("Write stdout to {:s}".format(_stdout_path))
        logger.debug("Write stderr to {:s}".format(_stderr_path))

        # cp tarball to server
        self._copy_file_to_server(tar_path, remote_tgz_path)

        if self._norun:
            norun = "echo "
        else:
            norun = ""

        # The TRiP98 command must be encapsulated in a bash -l -c "TRiP98 < fff.exec"
        # then .bashrc_profile is checked. (However, not .bashrc)
        _tripcmd = "bash -l -c \"" + self.trip_bin_path + " < " + remote_exec_fn + "\""

        commands = ["cd " + self.remote_base_dir + ";" + "tar -zxvf " + remote_tgz_path,  # unpack tarball
                    "cd " + remote_run_dir + ";" + norun + _tripcmd,
                    "cd " + self.remote_base_dir + ";" + "tar -zcvf " + remote_tgz_path + " " + remote_rel_run_dir,
                    "cd " + self.remote_base_dir + ";" + "rm -r " + remote_run_dir]

        # test if TRiP is installed
        logger.debug("Test if TRiP98 can be reached remotely...")
        trip, ver = self.test_remote_trip()
        if trip is None:
            logger.error("Could not find TRiP98 on {:s} using path \"{:s}\"".format(self.servername,
                                                                                    self.trip_bin_path))
            raise EnvironmentError
        else:
            logger.info("Found {:s} version {:s} on {:s}".format(trip, ver, self.servername))

        fp_stdout = open(_stdout_path, "w")
        fp_stderr = open(_stderr_path, "w")

        # open ssh channel and run commands
        ssh = self._get_ssh_client()
        for _cmd in commands:
            logger.debug("Execute on remote server: {:s}".format(_cmd))
            self.log(_cmd)
            stdin, stdout, stderr = ssh.exec_command(_cmd)
            rc = int(stdout.channel.recv_exit_status())  # recv_exit_status() returns an string type
            if rc != 0:
                logger.error("TRiP98 error: return code {:d}".format(rc))
            else:
                logger.debug("TRiP98 exited with status: {:d}".format(rc))

            answer_stdout = stdout.read().decode('utf-8')
            answer_stderr = stderr.read().decode('utf-8')
            logger.info("Remote answer stdout:\n{:s}".format(answer_stdout))
            logger.info("Remote answer stderr:\n{:s}".format(answer_stderr))
            fp_stdout.write(answer_stdout)
            fp_stderr.write(answer_stderr)
            self.log(answer_stdout)

        fp_stdout.close()
        fp_stderr.close()
        ssh.close()

        self._move_file_from_server(remote_tgz_path, local_tgz_path)
        self._extract_tarball(remote_tgz_path, plan._working_dir)
        logger.debug("Locally remove {:s}".format(local_tgz_path))
        os.remove(local_tgz_path)

        return rc

    def _finish(self, plan):
        """ return requested results, copy them back in to plan._working_dir
        """

        for _file_name in plan._out_files:
            _path = os.path.join(plan._temp_dir, _file_name)

            # only copy files back, if we actually have been running TRiP
            if self._norun:
                logger.info("dummy run: would now copy {:s} to {:s}".format(_path, plan._working_dir))
            else:
                logger.info("copy {:s} to {:s}".format(_path, plan._working_dir))
                shutil.copy(_path, plan._working_dir)

        for _file_name in plan._out_files:
            _path = os.path.join(plan._temp_dir, _file_name)
            if ".phys.dos" in _file_name:
                _ctx_cube = DosCube()
                if not self._norun:
                    _ctx_cube.read(_path)
                    plan.dosecubes.append(_ctx_cube)

            if ".bio.dos" in _file_name:
                _ctx_cube = CtxCube()
                if not self._norun:
                    _ctx_cube.read(_path)
                    plan.dosecubes.append(_ctx_cube)

            if ".dosemlet.dos" in _file_name:
                _let_cube = LETCube()
                if not self._norun:
                    _let_cube.read(_path)
                    plan.letcubes.append(_let_cube)

            if ".rst" in _file_name:
                logger.warning("attaching fields to class not implemented yet {:s}".format(_path))
                # TODO
                # need to access the RstClass here for each rst file. This will then need to be attached
                # to the proper field in the list of fields.

        if self._cleanup:
            logger.debug("Delete {:s}".format(plan._temp_dir))
            shutil.rmtree(plan._temp_dir)

    @staticmethod
    def _compress_files(_source_dir):
        """
        Builds the tar.gz from what is found in _source_dir.
        Resulting file will be stored in "source_dir/.." if not specified otherwise

        :params str _source_dir: path to dir with the files to be compressed.
        :returns: full path to tar.gz file.

        :example:
        _compress_files("/home/bassler/test/foobar")
        will compress the contents of /foobar into foobar.tar.gz, including the foobar root dir
        and returns "/home/bassler/test/foobar.tar.gz"
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

        logger.debug("Compressing files in {:s} to {:s}".format(_source_dir, _target_filename))

        with tarfile.open(_target_filename, "w:gz") as tar:
            tar.add(_basedir, arcname=_basedir)

        os.chdir(_cwd)

        return os.path.join(_pardir, _target_filename)

    @staticmethod
    def _extract_tarball(tgz_path, basedir):
        """ Extracts a tarball at path into self.working_dir
        :params tgz_path: full path to tar.gz file
        :params basedir: where extracted files (including the tgz root dir) will be stored
        :returns: a string where the files were extracted including the tgz root dir

        i.e. _extract_tarball("foobar.tar.gz", "/home/bassler/test")
        will return "/home/bassler/test/foobar"

        """

        logger.debug("Locally extract {:s} in {:s}".format(tgz_path, basedir))

        _basedir, _tarfile = os.path.split(tgz_path)
        _outdirname = _tarfile.rstrip(".tar.gz")  # TODO: os.path.splitext()
        _outdir = os.path.join(_basedir, _outdirname)

        if os.path.exists(_outdir):
            shutil.rmtree(_outdir)

        _cwd = os.getcwd()
        os.chdir(basedir)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall("./")

        os.chdir(_cwd)
        return _outdir

    def _copy_file_to_server(self, _from, _to):
        """
        Copies the generated tar.gz file to the remote server.
        :params _from: full path to file
        :params _to: full path to location on server
        """

        _to_uri = self.servername + ":" + _to
        logger.debug("Copy {:s} to {:s}".format(_from, _to_uri))

        sftp, transport = self._get_sftp_client()
        sftp.put(localpath=_from, remotepath=_to)
        sftp.close()
        transport.close()

    def _copy_file_from_server(self, _from, _to):
        """ Copies a single file from a server
        :param _from: full path on remote server
        :param _to: full path on local computer
        """
        _from_uri = self.servername + ":" + _from
        logger.debug("Copy {:s} to {:s}".format(_from_uri, _to))

        sftp, transport = self._get_sftp_client()
        sftp.get(_from, _to)
        sftp.close()
        transport.close()

    def _move_file_from_server(self, _from, _to):
        """ Copies a removes a single file from a server
        :param _from: full path on remote server
        :param _to: full path on local computer
        """
        _from_uri = self.servername + ":" + _from
        logger.debug("Move {:s} to {:s}".format(_from_uri, _to))

        sftp, transport = self._get_sftp_client()
        sftp.get(_from, _to)
        sftp.remove(_from)
        sftp.close()
        transport.close()

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
                except Exception:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username, password or key in " + self.rsakey_local_path)
                    raise
            else:
                # login with provided username + private key
                rsa_key = paramiko.RSAKey.from_private_key_file(rsa_keypath)
                try:
                    ssh.connect(self.servername, username=self.username, pkey=rsa_key)
                except Exception:
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
                except Exception:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username, password or key in " + self.rsakey_local_path)
                    raise
            else:
                # login with provided username + private key
                rsa_key = paramiko.RSAKey.from_private_key_file(rsa_keypath)
                try:
                    transport.connect(username=self.username, pkey=rsa_key)
                except Exception:
                    logger.error("Cannot connect to " + self.servername)
                    logger.error("Check username and your key in " + self.rsakey_local_path)
                    raise
        else:
            # login with provided username + password
            transport.connect(username=self.username, password=self.password)

        sftp = paramiko.SFTPClient.from_transport(transport)

        return sftp, transport

    def test_local_trip(self):
        """ Test if TRiP98 can be reached locally.
        :returns tripname, tripver: Name of TRiP98 installation and its version.
        :returns: None, None if not installed
        """

        p = Popen([self.trip_bin_path], stdout=PIPE, stdin=PIPE)
        stdout, stderr = p.communicate("exit".encode('ascii'))

        _out = stdout.decode('utf-8')

        if "This is TRiP98" in _out:
            tripname = _out.split(" ")[3][:-1]
            tripver = _out.split(" ")[5].split("(")[0]
            return tripname, tripver
        else:
            return None, None

    def test_remote_trip(self):
        """ Test if TRiP98 can be reached remotely.
        :returns tripname, tripver: Name of TRiP98 installation and its version.
        :returns: None, None if not installed
        """

        # execute the list of commands on the remote server
        _tripcmd_test = "bash -l -c \"" + "cat exit | " + self.trip_bin_path + "\""

        # test if TRiP98 can be reached

        ssh = self._get_ssh_client()
        stdin, stdout, stderr = ssh.exec_command(_tripcmd_test)
        _out = stdout.read().decode('utf-8')
        ssh.close()

        if "This is TRiP98" in _out:
            tripname = _out.split(" ")[3][:-1]
            tripver = _out.split(" ")[5].split("(")[0]
            return tripname, tripver
        else:
            return None, None
