import sys,os
from pytrip.cube import Cube
from pytrip.dos2 import DosCube
from pytrip.ctx2 import CtxCube
from pytrip.vdx2 import VdxCube
from pytrip.let import LETCube

def is_number(s):
        try:
                float(s)
                return True
        except ValueError:
                return False
def get_parameter(param):
        try:
                i = sys.argv.index(param)
                return sys.argv[i+1]
        except ValueError:
                return None
def use_operators(action,args):
	file_a = sys.argv[2]
	file_b = sys.argv[3]
        if is_number(file_a) is True:
                a = float(file_a)
        else:
                a = Cube()
                a.read_trip_data_file(file_a)
        if is_number(file_b) is True:
                b = float(file_b)
        else:
                b = Cube()
                b.read_trip_data_file(file_b)
        if action == "add":
		c = a+b
	if action == "sub":
		c = a-b
	if action == "div":
		c = a/b
	if action == "mul":
		c = a*b
        return c
def create_cube():
        type = get_parameter("-t")
	headerfile = get_parameter('-hed')
       	vdxfile = get_parameter('-structures')
	if vdxfile == None:
		vdxfile = os.path.splitext(headerfile)[0] + ".vdx"
        if type == "dose":
                dose = get_parameter('-dose')
                if dose == None:
                        dose = 1000
                c = Cube()
                c.read_trip_header_file(headerfile)
                v = VdxCube("")
                v.import_vdx(vdxfile)
                voi = v.get_voi_by_name(voiname)

                c.load_from_structure(voi,dose)
                return c
        elif type == 'lvh':
                voiname = get_parameter('-target')
                l = LETCube()
                l.read_trip_data_file(os.path.splitext(headerfile)[0] + ".dos")
                v = VdxCube("")
                v.import_vdx(vdxfile)
                voi = v.get_voi_by_name(voiname)
                l.write_lvh_to_file(voi,get_parameter('-o'))



action = sys.argv[1]
output = "cube"
if "-h" in sys.argv:
	f = open("help")
	content = f.read()
	f.close()
	print content
if "-o" in sys.argv:
	output = sys.argv[sys.argv.index("-o")+1]
if action in ["add","div","sub","mul"]:
        c = use_operators(action,sys.argv)
elif action == "create":
        c = create_cube()
else:
        exit()
if c is not None:
        c.write_trip_data(output + ".dos")
        c.write_trip_header(output + ".hed")
                


     
