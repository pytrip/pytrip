import sys
from pytrip.cube import Cube
def is_number(s):
        try:
                float(s)
                return True
        except ValueError:
                return False
   
action = sys.argv[1]
output = "cube"
if "-o" in sys.argv:
	output = sys.argv[sys.argv.index("-o")+1]
if action in ["add","div","sub","mul"]:
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
	c.write_trip_data(output + ".dos")
	c.write_trip_header(output + ".hed")
                


     
