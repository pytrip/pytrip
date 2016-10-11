import os

from pytrip import VdxCube
from pytrip.ctimage import CTImages
from pytrip.ctx import CtxCube
from pytrip.tripexecuter.field import Field
from pytrip.tripexecuter.rbehandler import RBEHandler

from pytrip.tripexecuter.tripplan import TripPlan
from pytrip.tripexecuter.tripexecuter import TripExecuter


from pytrip.tripexecuter.tripplancollection import TripPlanCollection
from pytrip.tripexecuter.tripvoi import TripVoi
from pytrip.tripexecuter.voi import Voi
from pytrip.tripexecuter.voicollection import VoiCollection


clean_path = "/home/grzanka/Desktop/DHV test/TST000/TST000000"

# load CT cube
c = CtxCube()
c.read(clean_path + ".ctx")
ct_images = CTImages(c)

# load VOIs
structures = VoiCollection(None)
if os.path.exists(clean_path + ".vdx"):
     structures = VdxCube("", c)
     structures.read(clean_path + ".vdx")
print(structures.get_voi_names())

# empty plan
plan = TripPlan()

# add target VOI
v = structures.get_voi_by_name('GTV')
target_voi = Voi("GTV_VOI", v)
plan.add_voi(target_voi)
plan.get_vois()[0].target = True

# add default field
field = Field("Field 1")
field.set_projectile("C")
plan.add_field(field)

# start trip
executer = TripExecuter(ct_images.get_modified_images(plan), RBEHandler())
executer.execute(plan)