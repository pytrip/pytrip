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
import time

from subprocess import Popen, PIPE, STDOUT

try:
    from shlex import quote  # Python >= 3.3
except ImportError:
    from pipes import quote

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.tripexecuter.executor_logger import FileExecutorLogger
from pytrip.util import human_readable_size, get_size, TRiP98FilePath

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

        self.executor_loggers = []  # elements of type ExecutorLogger

        # TODO: this should be only the TRiP98 command, and not the full path
        # however some issue when running remote, so lets leave this for now
        self.trip_bin_path = "TRiP98"  # where TRiP98 is installed, if not accessible in /usr/local/bin or similar

        self.logfile_prefix_stdout = "trip98.stdout."
        self.logfile_prefix_stderr = "trip98.stderr."

        self._norun = False  # set this to False for a dry run without TRiP98 (for testing purposes)
        self._cleanup = True  # delete any temporary dirs created (currently this is only the package dir)

        # remote planning
        # remote directory where a new temporary directory will be created, the package extracted and executed.
        self.remote = False  # remote or local execution
        self.servername = ""
        self.username = ""
        self.password = ""
        self.remote_base_dir = "./"
        self.pkey_path = ""

        self._working_dir = ""  # dir with results will be stored in this dir
        self._use_default_logger = None
        self._file_logger = None
        self._start_time = None

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

    def execute(self, plan, run=True, use_default_logger=True):
        """
        Executes the Plan() object using TRiP98.

        :returns int: return code from trip execution.
        """

        self._norun = not run
        if run:
            logger.debug("Execute TRiP98...")
        else:
            logger.debug("Execute TRiP98 (dry-run)...")

        self._use_default_logger = use_default_logger
        self._file_logger = None

        self._pre_execute(plan)  # prepare directory where all will be run, put files in it.
        rc = self._run_trip(plan)  # run TRiP
        self._finish(plan)

        return rc

    def _pre_execute(self, plan):
        """
        Prepare a temporary working directory where TRiP will be executed.
        Sets:
        temp_dir where all will be executed
        """

        # attach working dir to plan, but expanded for environment variables
        if not plan.working_dir:
            logger.warning("No working directory was specified for plan. Setting it to ./")
            plan.working_dir = "./"
        self._working_dir = os.path.expandvars(plan.working_dir)

        # create default logger that print output to files
        if self._use_default_logger:
            # stdout and stderr are always written locally
            stdout_path = os.path.join(self._working_dir, self.logfile_prefix_stdout + plan.basename)
            stderr_path = os.path.join(self._working_dir, self.logfile_prefix_stderr + plan.basename)
            self._file_logger = FileExecutorLogger(stdout_path, stderr_path)

        # We will not run TRiP98 in working dir, but in an isolated subdirectory which will be
        # uniquely created for this purpose, located after the working dir.
        # This contents of this subdir will be termed "the package", as it is a clean environment.
        # The name of the subdir will be stored in plan.temp_dir
        # It may be safely deleted afterwards, once the resulting files are copied back to the plan.working_dir

        if not plan.temp_dir:
            import tempfile
            from datetime import datetime
            now = datetime.now()
            prefix = 'trip98_{:%Y%m%d_%H%M%S}_'.format(now)
            plan.temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self._working_dir)

        plan.temp_dir = os.path.join(self._working_dir, plan.temp_dir)
        exec_path = os.path.join(plan.temp_dir, plan.basename + ".exec")
        plan.save_exec(exec_path)  # add the .exec as well

        logger.debug("Created temporary working directory {:s}".format(plan.temp_dir))

        flist = []  # list of files which must be copied to the package.

        if plan.incube_basename:
            flist.append(os.path.join(self._working_dir, plan.incube_basename + ".dos"))
            flist.append(os.path.join(self._working_dir, plan.incube_basename + ".hed"))

        for field in plan.fields:
            if field.use_raster_file:
                flist.append(os.path.join(self._working_dir, field.basename + ".rst"))

        # once the file list flist is complete, copy it to the package location
        for fn in flist:
            logger.debug("Copy {:s} to {:s}".format(fn, plan.temp_dir))
            shutil.copy(fn, plan.temp_dir)

        # Ctx and Vdx files are not copied, but written from the objects passed to Execute() during __init__.
        # This gives the user better control over what Ctx and Vdx should be based for the planning.
        # Copying can though be forced by specifying .ctx and .vdx path during self.__init__
        # This is necessary as long as PyTRiP has trouble to produce TRiP98 readable files reliably.
        if self._ctx_path:
            ctx_base, _ = os.path.splitext(self._ctx_path)
            logger.info("Copying {:s} to tmp dir, instead of writing from Ctx object.".format(self._ctx_path))
            shutil.copy(self._ctx_path, plan.temp_dir)  # copy .ctx
            shutil.copy(ctx_base + ".hed", plan.temp_dir)  # copy.hed
        else:
            self.ctx.write(os.path.join(plan.temp_dir, self.ctx.basename + ".ctx"))  # will also make the hed

        if self._vdx_path:
            logger.info("Copying {:s} to tmp dir, instead of writing from Vdx object.".format(self._vdx_path))
            shutil.copy(self._vdx_path, plan.temp_dir)
        else:
            self.vdx.write(os.path.join(plan.temp_dir, self.vdx.basename + ".vdx"))

    def _run_trip(self, plan, run_dir=""):
        """ Method for executing the attached exec.
        :params str run_dir: overrides dir where the TRiP package is assumed to be, otherwise
        the package location is taken from plan.temp_dir

        :returns int: return code from trip execution.
        """
        if not run_dir:
            run_dir = plan.temp_dir
        if self.remote:
            rc = self._run_trip_remote(plan, run_dir)
        else:
            rc = self._run_trip_local(plan, run_dir)
        return rc

    def _run_trip_local(self, plan, run_dir):
        """
        Runs TRiP98 on local computer.
        :params Plan plan : plan object to be executed
        :params str run_dir: directory where a clean trip package is located, i.e. where TRiP98 will be run.

        :returns int: return code of TRiP98 execution. (0, even if optimiztion fails)
        """
        logger.info("Run TRiP98 in LOCAL mode.")

        trip, ver = self.test_local_trip()
        if trip is None:
            logger.error("Could not find TRiP98 using path \"{:s}\"".format(self.trip_bin_path))
            raise EnvironmentError

        logger.info("Found {:s} version {:s}".format(trip, ver))

        # start local process running TRiP98
        self._info("\nRunning TRiP98")
        if self._norun:  # for testing, just echo the command which would be executed
            p = Popen(["/bin/echo", self.trip_bin_path], stdout=PIPE, stderr=STDOUT, stdin=PIPE, cwd=run_dir)
        else:
            p = Popen([self.trip_bin_path], stdout=PIPE, stderr=STDOUT, stdin=PIPE, cwd=run_dir)

        # fill standard input with configuration file content
        p.stdin.write(plan.get_exec().encode("ascii"))
        p.stdin.flush()

        with p.stdout:
            for line in p.stdout:
                text = line.decode("ascii")
                self._log(text.rstrip())

        rc = p.wait()
        if rc != 0:
            self._error("Return code: {:d}".format(rc))
            logger.error("TRiP98 error: return code {:d}".format(rc))
        else:
            logger.debug("TRiP98 exited with status: {:d}".format(rc))
        self._log("")

        return rc

    def _run_trip_remote(self, plan, _run_dir=None):
        """ Method for executing the attached plan remotely.
        :params Plan plan: plan object
        :params str run_dir: TODO: not used

        :returns: integer exist status of TRiP98 execution. (0, even if optimization fails)
        """
        logger.info("Run TRiP98 in REMOTE mode.")

        tar_path = self._compress_files(plan.temp_dir)  # make a tarball out of the TRiP98 package

        # prepare relevant dirs and paths
        _, tgz_filename = os.path.split(tar_path)
        local_tgz_path = os.path.join(self._working_dir, tgz_filename)
        remote_tgz_path = os.path.join(self.remote_base_dir, tgz_filename)

        # extract filename by removing two extensions .tar.gz
        remote_rel_run_dir = os.path.splitext(os.path.splitext(tgz_filename)[0])[0]

        remote_run_dir = os.path.join(self.remote_base_dir, remote_rel_run_dir)
        remote_exec_fn = plan.basename + ".exec"

        # cp tarball to server
        self._copy_file_to_server(tar_path, remote_tgz_path)

        if self._norun:
            norun = "echo "
        else:
            norun = ""

        # The TRiP98 command must be encapsulated in a bash -l -c "TRiP98 < fff.exec"
        # then .bashrc_profile is checked. (However, not .bashrc)
        tripcmd = "bash -l -c \"" + quote(self.trip_bin_path) + " < " + quote(remote_exec_fn) + "\""

        # log status every 10 MB (record-size * checkpoint)
        tar_log_parameters = "--record-size=10K --checkpoint=1024 --totals"

        commands = [
            # extract tarball
            "cd " + quote(self.remote_base_dir) + ";" +
            "echo Size to extract: " +
            "$(file " + quote(remote_tgz_path) + " | rev | cut -d' ' -f1 | rev | numfmt --to=iec-i --suffix=B);" +
            "tar -zx " + tar_log_parameters + " --checkpoint-action=echo=\"Extracted bytes %{}T\" " +
            "-f " + quote(remote_tgz_path),

            # run TRiP98
            "cd " + quote(remote_run_dir) + ";" +
            norun + tripcmd,

            # compress to tarball
            "cd " + quote(remote_run_dir) + ";" +
            "rm " + plan.basename + ".hed " + plan.basename + ".ctx " + plan.basename + ".vdx;" +
            "cd " + quote(self.remote_base_dir) + ";" +
            "echo Size to compress: $(du -sh " + quote(remote_rel_run_dir) + " | cut -f1)iB;" +
            "tar -zc " + tar_log_parameters + " --checkpoint-action=echo=\"Compressed bytes %{}T\" " +
            "-f " + quote(remote_tgz_path) + " " + quote(remote_rel_run_dir) + ";" +
            "rm -r " + quote(remote_run_dir)
        ]

        # test if TRiP is installed
        logger.debug("Test if TRiP98 can be reached remotely...")
        trip, ver = self.test_remote_trip()
        if trip is None:
            logger.error("Could not find TRiP98 on {:s} using path \"{:s}\"".format(self.servername,
                                                                                    self.trip_bin_path))
            self._error("Could not find TRiP98 on {:s} using path \"{:s}\"".format(self.servername, self.trip_bin_path))
            raise EnvironmentError

        logger.info("Found {:s} version {:s} on {:s}".format(trip, ver, self.servername))

        # open ssh channel and run commands
        ssh = self._get_ssh_client()
        for cmd in commands:
            logger.debug("Execute on remote server: {:s}".format(cmd))
            self._info("Executing commands")
            self._log(cmd.replace(";", "\n"))
            _, stdout, stderr = ssh.exec_command(cmd, get_pty=True)  # skipcq: BAN-B601

            self._info("Result")
            for line in stdout:
                self._log(line.rstrip())
            for line in stderr:
                self._error(line.rstrip())

            rc = int(stdout.channel.recv_exit_status())  # recv_exit_status() returns an string type
            if rc != 0:
                self._error("Return code: {:d}".format(rc))
                logger.error("TRiP98 error: return code {:d}".format(rc))
            else:
                logger.debug("TRiP98 exited with status: {:d}".format(rc))

            self._log("")

        ssh.close()

        self._move_file_from_server(remote_tgz_path, local_tgz_path)
        self._extract_tarball(local_tgz_path, self._working_dir)
        logger.debug("Locally remove {:s}".format(local_tgz_path))
        os.remove(local_tgz_path)

        return rc

    def _finish(self, plan):
        """ return requested results, copy them back in to self._working_dir
        """

        self._info("Reading results")
        for file_name in plan.out_files:
            path = os.path.join(plan.temp_dir, file_name)

            # only copy files back, if we actually have been running TRiP
            if self._norun:
                logger.info("dummy run: would now copy {:s} to {:s}".format(path, self._working_dir))
            else:
                logger.info("copy {:s} to {:s}".format(path, self._working_dir))
                try:
                    shutil.copy(path, self._working_dir)
                except IOError as e:
                    logger.debug("No file {:s}".format(file_name))
                    self._error(str(e))
                    self._error("Simulation failed")
                    self._end()
                    raise IOError

        for file_name in plan.out_files:
            path = os.path.join(plan.temp_dir, file_name)

            if not self._norun:
                # check if file is dose file e.g. "phys.dos", "bio.dos"
                if TRiP98FilePath(file_name, DosCube).suffix in DosCube.allowed_suffix and \
                        TRiP98FilePath(file_name, DosCube).is_valid_datafile_path():
                    self._log("Reading {:s} file".format(file_name))
                    dose_cube = DosCube()
                    dose_cube.read(path)
                    plan.dosecubes.append(dose_cube)

                # check if file is let file e.g. "dosemlet.dos", "mlet.dos"
                elif TRiP98FilePath(file_name, LETCube).suffix in LETCube.allowed_suffix and \
                        TRiP98FilePath(file_name, LETCube).is_valid_datafile_path():
                    self._log("Reading {:s} file".format(file_name))
                    let_cube = LETCube()
                    let_cube.read(path)
                    plan.letcubes.append(let_cube)

                elif ".rst" in file_name:
                    logger.warning("attaching fields to class not implemented yet {:s}".format(path))
                    # TODO
                    # need to access the RstClass here for each rst file. This will then need to be attached
                    # to the proper field in the list of fields.

        self._info("Done")

        self._end()

        if self._cleanup:
            logger.debug("Delete {:s}".format(plan.temp_dir))
            shutil.rmtree(plan.temp_dir)

    def _compress_files(self, source_dir):
        """
        Builds the tar.gz from what is found in source_dir.
        Resulting file will be stored in "source_dir/.." if not specified otherwise

        :params str source_dir: path to dir with the files to be compressed.
        :returns: full path to tar.gz file.

        :example:
        _compress_files("/home/bassler/test/foobar")
        will compress the contents of /foobar into foobar.tar.gz, including the foobar root dir
        and returns "/home/bassler/test/foobar.tar.gz"
        """

        # source_dir may be of either "/foo/bar" or "/foo/bar/". We only need "bar",
        # but dirname() will return foo in the first case and bar in the second,
        # fix this by adding an extra separator, in that case all will be stripped.
        # add an extra trailing slash, if there are multiple
        dirpath = os.path.dirname(source_dir + os.sep)
        _, dirname = os.path.split(dirpath)
        target_path = source_dir + ".tar.gz"

        logger.debug("Compressing files in {:s} to {:s}".format(source_dir, target_path))
        self._info("Compressing files in {:s} to {:s}".format(source_dir, target_path))

        # trick for no nonlocal in python 2.7
        track_progress_info = {
            "total_size": get_size(source_dir),
            "sum_size": 0
        }

        def track_progress(tarinfo):
            if tarinfo.isfile():
                track_progress_info["sum_size"] += tarinfo.size
                percentage = int(track_progress_info["sum_size"] / track_progress_info["total_size"] * 100)
                self._log("Compressing file {} with size {} ({}%)".format(tarinfo.name,
                                                                          human_readable_size(tarinfo.size),
                                                                          percentage))
            return tarinfo

        with tarfile.open(target_path, "w:gz") as tar:
            self._log("Size to compress: {:s}".format(human_readable_size(track_progress_info["total_size"])))
            tar.add(source_dir, arcname=dirname, filter=track_progress)
        self._log("Compressing done\n")

        return target_path

    def _extract_tarball(self, tgz_path, target_dirpath):
        """ Extracts a tarball at path into self.working_dir
        :params tgz_path: full path to tar.gz file
        :params target_dirpath: where extracted files (including the tgz root dir) will be stored
        :returns: a string where the files were extracted including the tgz root dir

        i.e. _extract_tarball("foobar.tar.gz", "/home/bassler/test")
        will return "/home/bassler/test/foobar"
        """

        logger.debug("Locally extract {:s} in {:s}".format(tgz_path, target_dirpath))
        self._info("Extracting {:s} in {:s}".format(tgz_path, target_dirpath))

        _, tgz_file = os.path.split(tgz_path)

        # extract filename by removing two extensions .tar.gz
        out_dirname = os.path.splitext(os.path.splitext(tgz_file)[0])[0]

        out_dirpath = os.path.join(target_dirpath, out_dirname)

        if os.path.exists(out_dirpath):
            shutil.rmtree(out_dirpath)

        # trick for no nonlocal in python 2.7
        track_progress_info = {
            "sum_size": 0
        }

        def track_progress(members, files_total_size):
            for file in members:
                yield file
                if file.isfile():
                    track_progress_info["sum_size"] += file.size
                    percentage = int(track_progress_info["sum_size"] / files_total_size * 100)
                    self._log("Extracting file {} with size {} ({}%)".format(file.name,
                                                                             human_readable_size(file.size),
                                                                             percentage))

        with tarfile.open(tgz_path, "r:gz") as tar:
            total_size = sum(file.size for file in tar)
            self._log("Size to extract: {:s}".format(human_readable_size(total_size)))
            tar.extractall(target_dirpath, members=track_progress(tar, total_size))

        self._log("Extracting done\n")

        return out_dirpath

    def _copy_file_to_server(self, source, destination):
        """
        Copies the generated tar.gz file to the remote server.
        :params source: full path to file
        :params destination: full path to location on server
        """
        destination_uri = self.servername + ":" + destination
        logger.debug("Copy {:s} to {:s}".format(source, destination_uri))

        self._info("Transferring files to remote")
        sftp = self._get_sftp_client()
        self._start_time = time.time() - 3
        sftp.put(localpath=source, remotepath=destination, callback=self._log_sftp_progress)
        self._log("Transferring done\n")
        sftp.close()

    def _move_file_from_server(self, source, destination):
        """ Copies and removes a single file from a server
        :param source: full path on remote server
        :param destination: full path on local computer
        """
        source_uri = self.servername + ":" + source
        logger.debug("Move {:s} to {:s}".format(source_uri, destination))

        self._info("Transferring files from remote")
        sftp = self._get_sftp_client()
        self._start_time = time.time() - 3
        sftp.get(remotepath=source, localpath=destination, callback=self._log_sftp_progress)
        self._log("Transferring done\n")
        sftp.remove(source)
        sftp.close()

    def _log_sftp_progress(self, transferred, to_be_transferred):
        end_time = time.time()
        # log every 3 seconds or when done
        if end_time - self._start_time >= 3 or transferred == to_be_transferred:
            self._start_time = end_time
            percentage = int(transferred / to_be_transferred * 100)
            self._log("Transferred: {0}/{1} ({2}%)".format(human_readable_size(transferred),
                                                           human_readable_size(to_be_transferred),
                                                           percentage))

    def _get_ssh_client(self):
        """ returns an open ssh client
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if not self.pkey_path:
                self.pkey_path = None
            ssh.connect(self.servername, username=self.username, password=self.password, key_filename=self.pkey_path)
        except Exception as e:
            logger.error("Cannot connect to " + self.servername)
            self._error(str(e))
            self._end()
            raise

        return ssh

    def _get_sftp_client(self):
        """ returns a sftp client object and the corresponding transport socket.
        """
        sftp = self._get_ssh_client().open_sftp()

        return sftp

    def test_local_trip(self):
        """ Test if TRiP98 can be reached locally.
        :returns tripname, tripver: Name of TRiP98 installation and its version.
        :returns: None, None if not installed
        """

        p = Popen([self.trip_bin_path], stdout=PIPE, stderr=STDOUT, stdin=PIPE)
        stdout, _ = p.communicate("exit".encode('ascii'))

        out = stdout.decode('utf-8')

        if "This is TRiP98" in out:
            tripname = out.split(" ")[3][:-1]
            tripver = out.split(" ")[5].split("(")[0]
            return tripname, tripver

        return None, None

    def test_remote_trip(self):
        """ Test if TRiP98 can be reached remotely.
        :returns tripname, tripver: Name of TRiP98 installation and its version.
        :returns: None, None if not installed
        """

        # execute the list of commands on the remote server
        tripcmd_test = "bash -l -c \"" + "cat exit | " + quote(self.trip_bin_path) + "\""

        # test if TRiP98 can be reached
        ssh = self._get_ssh_client()
        _, stdout, _ = ssh.exec_command(tripcmd_test)  # skipcq: BAN-B601
        out = stdout.read().decode('utf-8')
        ssh.close()

        if "This is TRiP98" in out:
            tripname = out.split(" ")[3][:-1]
            tripver = out.split(" ")[5].split("(")[0]
            return tripname, tripver

        return None, None

    def add_executor_logger(self, executor_logger):
        """ An executor_logger is of type ExecutorLogger.
        """
        self.executor_loggers.append(executor_logger)

    def _info(self, text):
        self._loggers("info", text)

    def _error(self, text):
        self._loggers("error", text)

    def _log(self, text):
        self._loggers("log", text)

    def _end(self):
        """ Ends all executor loggers
        """
        if self._use_default_logger:
            self._file_logger.end()
        for executor_logger in self.executor_loggers:
            executor_logger.end()

    def _loggers(self, func_name, text):
        """ Sends text to all executor loggers
        """
        if self._use_default_logger:
            getattr(self._file_logger, func_name)(text)
        for executor_logger in self.executor_loggers:
            getattr(executor_logger, func_name)(text)
