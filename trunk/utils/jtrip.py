import sys

action = sys.argv[1]
if action is ["add","div","sub","mul"]:
        if is_number(argv[2]) is True:
                a = float(argv[2])
        else:
                a = Cube()
                a.read_trip_data_file(argv[2])
        if is_number(argv[3]) is True:
                b = float(argv[3])
        else:
                b = Cube()
                b.read_trip_data_file(argv[3])
        if action is "add":
                


def is_number(s):
        try:
                float(s)
                return True
        except ValueError:
                return False
        
