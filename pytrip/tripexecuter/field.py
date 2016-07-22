from pytrip.error import InputError
from pytrip.tripexecuter.pytripobj import pytripObj


class Field(pytripObj):
    def __init__(self, name):
        self.save_fields = ["name", "gantry", "couch", "fwhm", "rasterstep", "doseextension", "contourextension",
                            "zsteps", "projectile", "target"]
        self.name = name
        self.gantry = 0.0
        self.couch = 0.0
        self.fwhm = 5.0
        self.rasterstep = [2, 2]
        self.doseextension = 1.2
        self.contourextension = 0.6
        self.rasterfile = None

        self.zsteps = 1.0
        self.projectile = 'C'
        self.target = []
        self.selected = False

    def get_name(self):
        return self.name

    def set_rasterfile(self, path):
        self.rasterfile = path

    def get_rasterfile(self):
        return self.rasterfile

    def set_name(self, name):
        self.name = name

    def is_selected(self):
        return self.selected

    def toogle_selected(self):
        self.selected = not self.selected

    def get_gantry(self):
        return self.gantry

    def set_gantry(self, angle):
        if type(angle) is str and not len(angle):
            angle = 0
        try:
            angle = (float(angle) + 360) % 360
            if angle < 0 or angle > 360:
                raise Exception()
            self.gantry = angle

        except Exception:
            raise InputError("Gantry angle shoud be a " "number between 0 and 360")

    def get_couch(self):
        return self.couch

    def get_rasterstep(self):
        return self.rasterstep

    def set_rasterstep(self, a, b):
        try:
            a = float(a)
            b = float(b)
            if a < 0 or b < 0:
                raise Exception()
            self.rasterstep = [a, b]
        except Exception:
            raise InputError("Rastersteps should be " "larger than 0 and numbers")

    def set_couch(self, angle):
        if type(angle) is str and not len(angle):
            angle = 0
        try:
            angle = (float(angle) + 360) % 360
            if angle < 0 or angle > 360:
                raise Exception()
            self.couch = angle

        except Exception:
            raise InputError("Couch angle shoud be " "a number between 0 and 360")

    def set_doseextension(self, doseextension):
        try:
            doseextension = float(doseextension)
            if doseextension < 0:
                raise Exception()
            self.doseextension = doseextension
        except Exception:
            raise InputError("Doseextension should be larger 0")

    def get_doseextension(self):
        return self.doseextension

    def set_contourextension(self, contourextension):
        try:
            contourextension = float(contourextension)
            if contourextension < 0:
                raise Exception()
            self.contourextension = contourextension
        except Exception:
            raise InputError("Contourextension should be larger 0")

    def get_contourextension(self):
        return self.contourextension

    def set_zsteps(self, zsteps):
        try:
            zsteps = float(zsteps)
            if zsteps < 0:
                raise Exception()
            self.zsteps = zsteps
        except Exception:
            raise InputError("ZSteps should be larger 0")

    def get_zsteps(self):
        return self.zsteps

    def get_fwhm(self):
        return self.fwhm

    def set_fwhm(self, fwhm):
        try:
            fwhm = float(fwhm)
            if fwhm <= 0:
                raise Exception()
            self.fwhm = fwhm

        except Exception:
            raise InputError("Fwhm shoud be a number and larger than 0")

    def get_projectile(self):
        return self.projectile

    def set_projectile(self, projectile):

        if projectile not in ['H', 'C', 'O', 'Ne']:
            raise InputError("Projectile not allowed")
        self.projectile = projectile

    def get_target(self):
        return self.target

    def set_target(self, target):
        if len(target) is 0:
            self.target = []
            return
        target = target.split(",")
        if len(target) is 3:
            try:
                self.target = [float(target[0]), float(target[1]), float(target[2])]
                return
            except Exception:
                # TODO fix that !
                pass
        raise InputError("Target should be empty " "or in the format x,y,z")
