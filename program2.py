#####################################################################################
#   Tim Lanzi                                                                       #                                                                   #
#                                                                                   #
#   COSC 420                                                                        #
#   Program 2    																	#
#																					#
#   This program makes a random triangle of unit area. It then procedes to pick 	#
#	random points r, s, and t on sides A, B, and C of the triangle, respectively.	#
#	It does this random point picking N times, where N is a number passed through	#
#	as a command-line argument by the user. As the program generates the N random	#
#	triangles, it determines the area of each inscribed triangle and places this 	#
#	area into a frequency table based on which bin it falls under. In this case,	#
#	each bin is 0.001. Using this frequency information, a probability distribution	#
#	of the areas is created and plotted. Then a Monte Carlo simulation is done 		#
#	using the already randomized areas of inscribed triangles to determine the 		#
#	average area of a triangle inscribed in a triangle of unit area.				#
#	                                                                				#
#####################################################################################

from mpi4py import MPI
import numpy as np
import sys, random, time
import matplotlib.pyplot as plt

# Makes the "big" triangle of unit area
def makeBigBoi():
	random.seed(431)

	# Start at origin (0, 0) and this is a point A
	pointA = np.array([0, 0])

	# Pick a y coordinate on the origin line (0, y) and this is some point B. Now we have a base.
	baseY = 50 * random.random()
	pointB = np.array([0, baseY])

	# The area equation will be 0.5 * base * height = 1
	# From this, we get height = 2 / base
	# Now we have an x coordinate for height. So we just need to pick a random y coordinate for point C
	heightX = 2 / baseY
	heightY = 50 * random.random()
	pointC = np.array([heightX, heightY])
	'''
	area = abs( (pointA[0] * (pointB[1] - pointC[1]) + pointB[0] * (pointC[1] - pointA[1]) + pointC[0] * (pointA[1] - pointB[1])) / 2 )
	print('Big area', area)
	'''
	return pointA, pointB, pointC

# Makes the little triangles
def makeLittleBoi(pointA, pointB, pointC):

	# Randomly pick 1 point on each of the 3 sides. No 2 points can be the same.
	r = pointA + (np.random.random() * (pointB - pointA))
	s = pointB + (np.random.random() * (pointC - pointB))
	while r[0] == s[0] and r[1] == s[1]:
		s = pointB + (np.random.random() * (pointC - pointB))			
	t = pointC + (np.random.random() * (pointA - pointC))
	while (t[0] == s[0] and t[1] == s[1]) or (t[0] == r[0] and t[1] == r[1]):
		t = pointC + (np.random.random() * (pointA - pointC))

	return r, s, t

# Main function
if __name__ == '__main__':

	# Create MPI world, get the size of the world, and get each process' rank
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# Initial input validation for if an argument isn't passed
	if (rank == 0 and len(sys.argv) == 1):
		print("You need to pass an argument for the number of triangles")
		quit()
	elif (rank != 0 and len(sys.argv) == 1):
		quit()

	t0 = time.time()

	# Grab the three points that make up the big triangle
	pointA, pointB, pointC = makeBigBoi()

	# Find the even workload for each process and the extra work to be done by the root process
	workload = int(sys.argv[1])/size
	extra = int(sys.argv[1]) - (size * int(workload))

	# Create frequency array. The size is the number of bins there will be.
	# Ex) 1000 bins means an error margin of 0.001.
	freq = np.zeros(1000, dtype='int')

	# Each process gets their workload
	if rank == 0:
		workload = int(workload) + int(extra)
	else:
		workload = int(workload)
	
	'''
	xTri = np.array([pointA[0], pointB[0], pointC[0], pointA[0]])
	yTri = np.array([pointA[1], pointB[1], pointC[1], pointA[1]])
	plt.plot(xTri, yTri)
	'''

	# Generates the little triangles, finds the areas, and places them in a freqency table
	# according to the bin they fall under
	for i in range(workload):
		r, s, t = makeLittleBoi(pointA, pointB, pointC)

		area = abs( (r[0] * (s[1] - t[1]) + s[0] * (t[1] - r[1]) + t[0] * (r[1] - s[1])) / 2 )
		
		bin = int(1000 * area)
		
		freq[bin] += 1

	# Reduce all freqency arrays into one on the root
	combinedFreqs = np.zeros(1000, dtype='int')
	comm.Reduce(freq, combinedFreqs, op=MPI.SUM, root=0)

	
	# Plot probability density distribution, mean triangle area, and mode triangle area.
	# Also uses a Monte Carlo simulation using the random triangle areas to calculate
	# the average inscribed triangle area.
	if rank == 0:
		t1 = time.time()
		totalTime = t1 - t0
		print('Time Elapsed: ', totalTime)

		dx = 0.001
		n = np.sum(combinedFreqs)
		x = np.arange(0,1,dx)+dx/2
		pmf = combinedFreqs/n
		pdf = pmf/dx
		label = "Mean Triangle Area: {:.4f}".format(np.sum(x*pdf*dx))
		imax = np.argmax(pdf)

		# Plotting stuff
		plt.plot(x,pdf,label=label)
		plt.plot(x[imax], pdf[imax], "o", label="Mode: {:.3f}".format(x[imax]))
		plt.legend()
		plt.grid(True)
		plt.xlabel('Triangle Area')
		plt.ylabel('Probability Density - /area')
		plt.show()

	
		'''
		xTri = np.array([r[0], s[0], t[0], r[0]])
		yTri = np.array([r[1], s[1], t[1], r[1]])
		plt.plot(xTri, yTri)
		print('r: {}, s: {}, t:{}'.format(r,s,t))
		'''
	
	#plt.show()
	
