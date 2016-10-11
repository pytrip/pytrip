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

plans = TripPlanCollection()
rbe = RBEHandler()
patient_name = ""

clean_path = "/home/grzanka/Desktop/DHV test/TST000/TST000000"

c = CtxCube()
c.read(clean_path + ".ctx")
ct_images = CTImages(c)

structures = VoiCollection(None)
if os.path.exists(clean_path + ".vdx"):
     structures = VdxCube("", c)
     structures.read(clean_path + ".vdx")

print(structures.get_voi_names())
v = structures.get_voi_by_name('GTV')

target_voi = Voi("GTV_VOI", v)
print("type", type(target_voi), target_voi.get_voi_data())

plan = TripPlan()

target_voi.target = True
plan.add_voi(target_voi)

plan.get_vois()[0].target = True

vv = plan.get_vois()[0]

plan.add_field(Field("Field 1"))

executer = TripExecuter(ct_images.get_modified_images(plan), rbe)
executer.execute(plan)