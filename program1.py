#####################################################################################
#   Tim Lanzi                                                                       #
#                                                                                   #
#   COSC 420                                                                        #
#   Program 1                                                                       #
#                                                                                   #
#   Takes two numbers from the command line as arguments. The first number, N, is   #
#   the size of the array, and the second, M, is the highest number that can be     #
#   randomly generated. The program will take this imformation and generate an      #
#   array of size N of integers from 0 - M. A frequency table for the numbers in    #
#   the array will be generated, as well as the number of even numbers, the         #
#   number of prime numbers, the percentage of prime numbers, and the disctinct     #
#   prime number values will be reported.                                           #
#####################################################################################

from mpi4py import MPI
import numpy as np
import sys

# Function checks whether a number is prime or not
def isPrime(num):
    if num == 2:
        return True
    elif num < 2 or num % 2 == 0:
        return False
    else:
        for n in range(3, int((num**0.5) + 2), 2):
            if num % n == 0:
                return False
        return True

# Function checks whether a number is even or not
def isEven(num):
    if num % 2 == 0:
        return True
    return False

# Main function
if __name__ == '__main__':

    # Make MPI world. Get the world size and the processes individual ranks
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initial black to catch if the two arguments are present or not
    if (rank == 0 and (len(sys.argv) == 1 or len(sys.argv) == 2)):
        print("You need to pass 2 arguments (The size of the array and the max integer value, respectively)")
        quit()
    elif (rank != 0 and (len(sys.argv) == 1 or len(sys.argv) == 2)):
        quit()

    # Set array size N to the first arg and the highest random int M to the second arg
    N = sys.argv[1]
    M = sys.argv[2]

    # Make the random array in the root process and make space for it in the other ones
    if rank == 0:
        arr = np.random.randint(low=0, high=int(M)+1, size=int(N), dtype='i')
    else:
        arr = np.empty(int(N), dtype='i')

    # Broadcast entire array to all processes
    comm.Bcast(arr, root=0)

    # Split up process workloads
    procArr = np.empty(0, dtype='i')
    for i in range(rank, len(arr), size):
        procArr = np.append(procArr, arr[i])

    #print("Rank: {}, Nums: {}".format(rank, procArr))

    # Acquire frequencies of numbers
    freqs = np.zeros(int(M)+1, dtype=int)
    for i in procArr:
        freqs[i] += 1

    #print("Rank: {}, Freqs: {}".format(rank, freqs))

    # Find the number of even numbers
    evens = np.zeros(1, dtype=int)
    for num in procArr:
        if isEven(num):
            evens[0] += 1

    #print("Rank: {}, Evens: {}".format(rank, evens))

    # Find the number of prime numbers
    primes = np.zeros(1, dtype=int)
    for num in procArr:
        if isPrime(num):
            primes[0] += 1

    #print("Rank: {}, primes: {}".format(rank, primes))

    # Create buffers for reducing the frequencies, number of primes, and number
    # of even into the root process
    numOfEvens = np.zeros(1, dtype=int)
    numOfPrimes = np.zeros(1, dtype=int)
    combinedFreqs = np.zeros(int(M)+1, dtype=int)

    # Reduce the frequencies, number of primes, and number of evens into process 0
    comm.Reduce(evens, numOfEvens, op=MPI.SUM, root=0)
    comm.Reduce(primes, numOfPrimes, op=MPI.SUM, root=0)
    comm.Reduce(freqs, combinedFreqs, op=MPI.SUM, root=0)

    # Print frequency table, number of evens, number of primes, percentage of
    # primes, and distinct prime numbers
    if rank == 0:
        print('    Number    |    Frequency')
        print('--------------------------------')
        for i in range(len(combinedFreqs)):
            print('{:14d}|{}'.format(i, combinedFreqs[i]))


        print('Number of evens: {}'.format(numOfEvens[0]))
        percentPrime = 100 * (numOfPrimes[0]/len(arr))
        print('Number of primes: {}'.format(numOfPrimes[0]))
        print('Percent of primes: {}%'.format(percentPrime))
        print('    Distinct Primes')
        print('--------------------------------')
        for i in range(len(combinedFreqs)):
            if combinedFreqs[i] != 0 and isPrime(i):
                print('{}  '.format(i), end='')
        print('')
