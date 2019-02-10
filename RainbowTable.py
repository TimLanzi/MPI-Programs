'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Tim Lanzi

Simulation of a distributed password cracker using rainbow tables. Only using the 
MD5 hashing algorithm and lower case letters for the purposes of this program. 
The user specifies the number of random passwords to be generated. The program
will split this number as evenly as possible (and give the spares the the root 
process) and generate their tables using this number. Once the tables have been 
generated, a random password is generated and each process will attempt to find it
in their tables. The results for each process will be reported.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from hashlib import md5
from random import randrange
from mpi4py import MPI
import numpy as np
from optparse import OptionParser

'''
Added an option parser to help clean up the main function. The chain length is
the number of times a password is hashed and reduced before it reaches the 
endpoint. The password length is the number of characters in the passwords.
Rows is the number of start points and endpoints generated overall. There
are default values for all options
'''
parser = OptionParser()
parser.add_option('-c', '--chain', dest='chainLength',
                  action='store', default='5')
parser.add_option('-p', '--passwd', dest='pwLength',
                  action='store', default='8')
parser.add_option('-r', '--rows', dest='rows',
                  action='store', default='1000')
(options, args) = parser.parse_args()

# Only using lowercase letters for the purpose of this program
chars = 'abcdefghijklmnopqrstuvwxyz'

'''
Generates a random password of the length the user specified using the characters 
in chars.
'''
def genPasswd(length):
  passwd = ''
  charsLength = len(chars)
  for i in range(length):
    passwd += chars[randrange(charsLength)]
  return passwd

'''
Encodes a password and uses the MD5 algorithm to hash it.
'''
def hash(word):
  word = word.encode('utf-8')   # The word must be encoded before it is hashed. The error message said so.
  return md5(word).hexdigest()

'''
Reducing the hash text back into plain text. Does not reverse the hash.
'''
def reduce(hText, link, pwLength):
  results = []

  # Need to convert hash into bytes before we reduce
  bytes = []   
  remaining = int(hText, 16)  # Casting the hash string into a base-16 int

  # Creating the list of bytes
  while remaining > 0:
    bytes.append(remaining % 256)
    remaining //= 256

  # Converting the bytes into plain text
  for i in range(pwLength):
    index = bytes[(i + link) % len(bytes)]
    newChar = chars[index % len(chars)]
    results.append(newChar)
  return ''.join(results)   # Return the reduced hash

'''
Creates the chain of hashes and plain text. Only returns the end hash 
to save. 
'''
def createChain(passwd, chainLength, pwLength):
  for i in range(chainLength):
    hText = hash(passwd)
    passwd = reduce(hText, i, pwLength)
  return hText

'''
Creates the actual rainbow table itself. The table is stored in a .txt 
file for every process. The name of which is passed into the function
as 'output'.
'''
def create_rainbow_table(output, chainLength, pwLength, rows):
  table = open(output, 'w')
  for i in range(rows):
    start = genPasswd(pwLength)
    endpoint = createChain(start, chainLength, pwLength)
    table.write('{},{}\n'.format(start, endpoint))

  table.close()

'''
Attempts to find the plain text password that matches with the hash passed
into the function as 'target'.
'''
def crackPW(target, table, chainLength, pwLength):
  rt = open(table, 'r')
  rows = rt.readlines()

  for link in range(chainLength):

    # Finds what would be the endpoint for the target hash. As the outer
    # for loop increments, the starting position for the chain will change.
    endpoint = target
    for i in range(link, chainLength-1):
      passwd = reduce(endpoint, i, pwLength)
      endpoint = hash(passwd)
      
    # Attempts to match the target endopint to an endpoint in the table.
    # Creates a list of start points from the table if there are matches.
    passwdList = []
    for row in rows:
      row = row.strip()
      splitline = row.split(',')
      if splitline[1] == endpoint:
        passwdList.append(splitline[0])

    # If there were matches, try to find the target hash in the chain
    for pw in passwdList:
      result = findHash(pw, target, chainLength)
      if result != None:
        return result
  
  return 'Password not found'

'''
Searches for the target hash within its chain.
'''
def findHash(start, target, chainLength):
  # If the start point's hash is the target, the start is the password.
  hText = hash(start)
  if hText == target:
    return start

  # Otherwise, hash and reduce to attempt to find the target
  i = 0
  while i < chainLength:
    passwd = reduce(hText, i, len(start))
    hText = hash(passwd)
    if hText == target:
      return passwd
    i += 1
  
  return None

'''
Main function
'''
def main():
  # Initialize MPI stuff
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  # Make output file name
  output = "rainbow{}.txt".format(rank)

  # Get options from parser
  pwLength = options.pwLength
  chainLength = options.chainLength
  rows = options.rows

  # Split workloads
  workload = int(rows)/size
  extra = int(rows) - (size * int(workload))

  # Make rainbow tables
  if rank == 0:
    workload += extra
    create_rainbow_table(output=output, chainLength=int(chainLength), pwLength=int(pwLength), rows=int(workload))
  else:
    create_rainbow_table(output=output, chainLength=int(chainLength), pwLength=int(pwLength), rows=int(workload))

  # Make one target password randomly and broadcast it to all processes
  if rank == 0:
    targetPW = hash(genPasswd(int(pwLength)))
  else:
    targetPW = None
  targetPW = comm.bcast(targetPW, root=0)

  # Attempt to find the password in the tables and print the results for each process
  result = crackPW(targetPW, output, int(chainLength), int(pwLength))
  print('Rank: {}\nResult: {}\n'.format(rank, result))

'''
Start program
'''
if __name__ == '__main__':
  main()