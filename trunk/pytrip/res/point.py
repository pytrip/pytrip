def point_in_polygon(x,y,polygon):
	intersects = 0
	n = len(polygon)
	x1 = polygon[0][0]
	y1 = polygon[0][1]
	for i in range(n+1):
		x2=polygon[i%n][0]
		y2=polygon[i%n][1]
		if y > min(y1,y2):
			if y <= max(y1,y2):
				if x <= max(x1,x2):
					if y1 != y2:
						xinters = (y-y1)*(x2-x1)/(y2-y1)+x1
					if x1 == x2 or x <= xinters:
						intersects+=1
		x1=x2
		y1=y2
	if intersects%2 != 0:
			return True
	else:
			return False 
def array_to_point_array(points):
	point = [[points[3*i],points[3*i+1],points[3*i+2]] for i in range(len(points)/3)]
	return point
#find a short distance problaby not the shortest
def short_distance_polygon_idx(poly1,poly2):
	d=10000000
	n1 = len(poly1)
	n2 = len(poly2)
	i1 = 0
	i2 = 0
	for i in range(n2):
		d1 = (poly2[i][0]-poly1[i1][0])**2+(poly2[i][1]-poly1[i1][1])**2
		if d1 < d:
			i2 = i
			d = d1
	for i in range(n1):
		d2 = (poly2[i2][0]-poly1[i][0])**2+(poly2[i2][1]-poly1[i][1])**2
		if d2 < d:
			i1 = i
			d = d2
	return (i1,i2,float(d)**0.5)
	
