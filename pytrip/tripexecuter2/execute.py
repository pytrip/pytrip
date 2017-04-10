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


class Execute(object):
    def __init__(self, ctx):
        """ Initialize the Execute class. 
        :params CtxCube() ctx: the CT images as a regular pytrip.CtxCube() object. 
        """

        logger.debug("Initializing TripExecuter()")
        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this execute object
        self.ctx = ctx
        ##self.rbe = rbe
        self.listeners = []
        self.trip_bin_path = "TRiP98"  # where TRiP98 is installed, if not accessible in /usr/local/bin or similar
        self.logfile_stdout = "trip98.stdout"
        self.logfile_stderr = "trip98.stderr"
        self.rsakey_local_path = "~/.ssh/id_rsa"
        self._runtrip = True  # set this to False for a dry run without TRiP98 (for testing purposes)
        ##self.run_dir = "" 
        self.remote_dir = "/tmp/."  # remote directory where all will be run
        
        ##def delete_workdir(self):
        ##shutil.rmtree(self.path)

    def execute(self, plan, callback=None):
        """
        Executes the Plan() object using TRiP98.
        """
        logger.debug("Execute TRiP98...")
        self.plan = plan
        self.callback = callback
        self._pre_execute()
        self._make_trip_exec()
        self._run_trip()
        self._finish()

    def _pre_execute(self):
        """
        Prepare a temporary working directory where TRiP will be executed.
        """
        self.split_plan(self.plan)
        # local plan name will not contain any spaces, and consists of a base name only, with no suffix.
        self._plan_name = self.plan.name.replace(" ", "_")
        self.working_dir = os.path.expandvars(self.plan.working_dir)

        # we will not run in working dir, but in an isolated sub directory which will be
        # uniquely created for this purpose.
        # this is then stored in self.path.
        if not hasattr(self, "_temp_dir"):
            import tempfile
            self._temp_dir = tempfile.mkdtemp(prefix='trip98_', dir=self.plan.working_dir)
        self.temp_dir = os.path.join(self.working_dir, self._temp_dir)

        ## TODO: think about how to handle directoried.
        # 1) We need a dir, where the run package is created, zipped, and then sent for execution (if remote)
        # 2) When results are coming back, these should be copied back somehow, but to where?
        #
        #self.filepath = self.path + self.plan_name
        #if os.path.exists(self.path):
        #    pass
        #    # shutil.rmtree(self.path)
        #else:
        #    os.makedirs(self.path)


    def make_trip_exec(self, no_output=False):
        """
        Generates the .exec script and stores it into self.trip_exec.
        """
        logger.info("Generating the trip_exec script...")
        oar_list = self.oar_list
        targets = self.targets
        fields = self.plan.get_fields()

        projectile = fields[0].get_projectile()
        output = []
        output.extend(self._make_exec_header())
        output.extend(self._make_exec_load_data_files(projectile))
        output.extend(self._make_exec_field(fields))
        output.extend(self._make_exec_oar(oar_list))

        ## Multiple target support.
        ## TODO: check what is going on here and if it is really needed.
        dosecube = None
        if len(targets) > 1:
            dosecube = DosCube(self.ctx)
            for i, voi in enumerate(targets):
                temp = DosCube(self.ctx)
                dose_level = int(voi.get_dose() / self.target_dose * 1000)
                if dose_level == 0:
                    dose_level = -1
                temp.load_from_structure(voi.get_voi().get_voi_data(), dose_level)
                if i == 0:
                    dosecube = temp * 1
                else:
                    dosecube.merge_zero(temp)
            dosecube.cube[dosecube.cube == -1] = int(0)

        # for incube optimization
        if self.plan.target_dose_cube():
            dosecube = self.plan.target_dose_cube

        if dosecube is not None:
            output.extend(self._make_exec_plan(incube="target_dose"))
        else:
            output.extend(self._make_exec_plan())
        if self.plan.optimize:
            output.extend(self._make_exec_opt())

        # attach the wanted output:
        # _plan_name has no suffix and is derived from plan.name with spaces converted to underscores.
        output.extend(self._make_exec_output(self._plan_name, fields))
        
        out = "\n".join(output) + "\nexit\n"
        self.trip_exec = out

    def _make_exec_output(self, basename, fields):
        """
        Generate TRiP98 exec commands for producing various output.
        :params str basename" basename for output files (i.e. no suffix) and no trailing dot
        :params last
        """
        output = []
        window = self.plan.window
        window_str = ""
        if len(window) is 6:
            window_str = " window({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f} ".format(window[0], window[1],  # Xmin/max
                                                                                     window[2], window[3],  # Ymin/max 
                                                                                     window[4], window[5])  # Zmin/max

        if self.plan.want_phys_dose:
            line = 'dose "{:s}."'.format(basename)
            line += ' /calculate  alg({:s})'.format(self.plan.dose_alg)
            line += window_str
            line += ' field(*) write'
            output.append(line)
            
        if self.plan.want_bio_dose:
            line = 'dose "{:s}."'.format(basename)
            line += ' /calculate  bioalgorithm({:s})'.format(self.plan.bio_alg)
            line += window_str
            line += ' biological norbe field(*) write'
            output.append(line)
            
        if self.plan.want_dlet:
            line = 'dose "{:s}."'.format(basename)
            line += ' /calculate  alg({:s})'.format(self.plan.dose_alg)
            line += window_str
            line += ' field(*) dosemeanlet write'
            output.append(line)
            
        if self.plan.want_rst and self.plan.optimize:
            for i, field in enumerate(fields):
                    output.append('field {:d} / write file({%s}.rst) reverseorder '.format(i + 1, field.name))
                    field.rasterfile_path = os.path.join(self.path, field.name)  # but without suffix? TODO: check
        return output

    def _make_exec_plan(self, incube=""):
        output = []
        plan = "plan / dose({:.2f}) ".format(self.plan.target_dose)
        if self.plan.target_tissue_type:
            plan += "target({:s}) ".format(self.plan.target_tissue_type)
        if self.plan.red_tissue_type:
            plan += "residual({:s}) ".format(self.plan.res_tissue_type)
        if not incube == "":
            plan += "incube({:s})".format(self.plan.incube)
        output.append(plan)
        return output

    def _make_exec_opt(self):
        """ Generates the optimizer command for TRiP based on Plan() loaded into self.
        :returns: an array of lines, but only holding a single line containing the entire command.
        """
        # opt / field(*) ctbased bio dosealg(ap) optalg(cg) bioalg(ld) geps(1E-4) eps(1e-3) iter(500)
        opt = "opt / field(*)"

        if not self.plan._opt_principles.has_key(self.plan.opt_principle)
            logger.error("Unknown optimization principle {:s}".format(self.plan.opt_principle))
        opt += " {:s}".format(self.plan.opt_principle)  # ctbased or H2Obased

        if not self.plan._opt_methods.has_key(self.plan.opt_method):
            logger.error("Unknown optimization method {:s}".format(self.plan.opt_method))
        opt += " {:s}".format(self.plan.opt_method)  # "phys" or "bio"
        
        if not self.plan._dose_algs.has_key(self.plan.dose_alg):
            logger.error("Unknown optimization dose algorithm{:s}".format(self.plan.dose_alg))
        opt += " dosealg({:s})".format(self.plan.dose_alg)  # "ap",..
        
        if not self.plan._opt_algs.has_key(self.plan.opt_alg):
            logger.error("Unknown optimization method {:s}".format(self.plan.opt_alg))
        opt += " optalg({:s})".format(self.plan.opt_alg)  # "cl"...

        # TODO: sanity check numbers
        opt += " geps({:f})".format(self.plan.geps)
        opt += " eps({:f})".format(self.plan.eps)
        return [opt]

    def _make_exec_header(self):
        """ Prepare sis, hlut, ddd, dedx, and scancap commands. 
        # TODO: scancap could go out into a new method.
        # TODO: check input params
        :returns: an array of lines, one line per command.
        """
        output = []
        output.append("time / on")
        output.append("sis  * /delete")
        output.append("hlut * /delete")
        output.append("ddd  * /delete")
        output.append("dedx * /delete")
        output.append('dedx {:s} / read'.format(self.plan.dedx_path))
        output.append('hlut {:s} / read'.format(self.plan.hlut_path))

        # ddd, spc, and sis:
        output.append('ddd {:s} / read'.format(self.plan.ddd_dir))

        if self.plan.spc_dir: # False for None and empty string.
            # We can only check if dir exists, if this is supposed to run locally.
            output.append('spc {:s} / read'.format(self.plan.spc_dir))

        if self.plan.sis_path:
            output.append('sis {:s} / read'.format(self.plan.sis_path))
            
        # Scancap:
        opt = "scancap / offh2o({:.3f})".format(self.plan.offh2o)
        opt += " rifi({:.3f})".format(self.plan.rifi)
        opt += " bolus({:.3f})".format(self.plan.bolus)
        opt += " minparticles({:d})".format(self.plan.minparticles)
        opt += " path({:s})".format(self.plan.scanpath)
        output.append(opt)
        
        output.append("random {:d}".format(self.plan.random_seed1))
        return output

    def _make_exec_inp_data(self, projectile = None):
        """ Prepare CTX, VOI loading and RBE base data
      
        :returns: an array of lines, one line per command.
        """
        output = []

        output.append("ct {:s} / read".format(self._plan_name))  # loads CTX
        output.append("voi {:s} / read".format(self._plan_name))  # loads VDX
        output.append("voi * /list")  # display all loaded VOIs. Helps for debugging.

        ## TODO: consider moving RBE file to DDD/SPC/SIS/RBE set
        ## TODO: check rbe.get_rbe_by_name method
        if self.plan.target_tissue_type:  # if not None or empty string:
            rbe = self.rbe.get_rbe_by_name(self.plan.target_tissue_type)
            output.append("rbe '{:s}' / read".format(rbe.path))

        if self.plan.res_tissue_type:
            rbe = self.rbe.get_rbe_by_name(self.plan.target_tissue_type)
            output.append("rbe '{:s}' / read".format(rbe.path))

        return output
    
    def _make_exec_fields(self, fields):
        """ Generate .exec command string for one or more fields.
        if Plan().optimize is False, then it is expected there will
        be a path to a rasterscan file in field.rst_path

        :returns: an array of lines, one line per field in fields.
        """
        output = []
        for i, _field in enumerate(fields):
            if self.plan.optimize:
                # this is a new optimization:
                line = "field {:d} / new".format(i + 1)
            else:
                # or, there is a precalculated raster scan file which will be used instead:
                field = "field {:d} / read file({:s})".format(i + 1, _field.rasterfile_path)
                
            line += " fwhm {:.3f}".format(_field.fwhm)
                
            line += " raster({%.2f},{%.2f}) ".format(_field.rasterstep[0],
                                                     _field.rasterstep[1])
            ##TODO: convert if Dicom angles were given
            ##gantry, couch = angles_to_trip(_field.gantry(), _field.couch())
            line += " couch({.1f})".format(_field.couch)
            line += " gantry({.1f})".format(_ganry.couch)

            # set isocenter if specified in field
            _tar = _field.target
            if len(_tar) is not 0:
                line += " target({%.1f},{%.1f},{%.1f}) ".format(_tar[0], _tar[1], _tar[2])

            # set dose extention:
            # TODO: check number of decimals which make sense
            # TODO: zeros allowed?
            line += " doseext({.4f})".format(_field.doseextension)
            line += " contourext({.2f})".format(_field.contourextension)
            line += " zsteps({.3f})".format(_field.zsteps)
            line += ' proj({:s})'.format(_field.projectile)
            output.append(line)
         return output

    def _make_exec_oar(self, oar_list):
        """ Generates the list of TRiP commands specifying the organs at risk.
        :params [str] oar_list: list of VOIs which are organs at risk (OARs)
        """
        output = []
        for oar in oar_list:
            _out = "voi " + oar.name.replace(" ", "_")
            _out += " / maxdosefraction({.3f}) oarset".format(oar.max_dose_fraction)
            output.append(_out)
        return output

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

     def _run_trip(self):
        """ Method for executing the attached exec.
        """
        if self.plan.remote:
            self._run_trip_remote()
        else:
            self._run_trip_local()

    def _run_trip_remote(self):
        """ Method for executing the attached plan remotely.
        """
        logger.info("Run TRiP98 in REMOTE mode.")
        # self.create_remote_run_file()
        self._compress_files()

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

        path_tgz = os.path.join(self.remote_dir, "temp.tar.gz")  # remote place where temp.tar.gz is stored
        rrun_dir = os.path.join(self.remote_dir, self.folder_name)  # remote run dir

        commands = ["cd " + self.remote_dir + ";" + "tar -zxvf " + path_tgz,
                    "cd " + rrun_dir + ";" + norun + "bash -lc '" + self.trip_bin_path + " < plan.exec '",
                    "cd " + self.remote_dir + ";" + "tar -zcvf " + path_tgz + " " + rrun_dir,
                    "cd " + self.remote_dir + ";" + "rm -r " + rrun_dir]

        logger.debug("Write stdout and stderr to " + os.path.join(self.working_dir, self.folder_name))
        fp_stdout = open(os.path.join(self.working_dir, self.folder_name, self.logfile_stdout), "w")
        fp_stderr = open(os.path.join(self.working_dir, self.folder_name, self.logfile_stderr), "w")

        for cmd in commands:
            logger.debug("Execute on remote: " + cmd)
            self.log(cmd)
            stdin, stdout, stderr = ssh.exec_command(cmd)
            answer_stdout = stdout.read()
            answer_stderr = stderr.read()
            logger.debug("Remote answer stdout:" + answer_stdout)
            logger.debug("Remote answer stderr:" + answer_stderr)
            fp_stdout.write(answer_stdout)
            fp_stderr.write(answer_stderr)
            self.log(answer_stdout)
        ssh.close()
        fp_stdout.close()
        fp_stderr.close()

        self._copy_back_from_server()
        self._extract_tarball()

    def _run_trip_local(self):
        """
        Runs TRiP98 on local computer.
        """
        logger.info("Run TRiP98 in LOCAL mode.")
        logger.debug("Write stdout and stderr to " + os.path.join(self.working_dir, self.folder_name))
        fp_stdout = open(os.path.join(self.working_dir, self.folder_name, self.logfile_stdout), "w")
        fp_stderr = open(os.path.join(self.working_dir, self.folder_name, self.logfile_stderr), "w")

        os.chdir("{:s}".format(self.path))

        if not self._runtrip:  # for testing
            norun = "echo "
        else:
            norun = ""

        # start local process running TRiP98
        p = Popen([norun + self.trip_bin_path], stdout=PIPE, stdin=PIPE)

        # fill standard input with configuration file conent
        # wait until process is finished and get standard output and error streams
        stdout, stderr = p.communicate(self.trip_exec.encode("ascii"))

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

    def _finish(self):
        pass

    def _compress_files(self):
        """
        Builds the tar.gz from what is found in self.path.
        """

        logger.debug("Compressing files in " + self.path)
        self.save_exec(os.path.join(self.working_dir, self.folder_name, "plan.exec"))
        with tarfile.open(os.path.join(self.working_dir, self.folder_name + ".tar.gz"), "w:gz") as tar:
            tar.add(self.path, arcname=self.folder_name)

    def set_plan(self, plan):
        self.plan = plan
        self.plan_name = self.plan.get_name().replace(" ", "_")

    def save_exec(self, path):
        """
        Writes the .exec script to disk.
        """
        self.split_plan()
        if self.mult_proj:
            path1 = path.replace(".exec", "")
            for projectile in self.projectiles:
                self.create_trip_exec(projectile, True)
                with open(path1 + "_" + projectile + ".exec", "wb+") as fp:
                    fp.write(self.trip_exec)
        else:
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
        shutil.rmtree("%s" % self.path)

