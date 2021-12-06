#import os

#os.system("echo test")

#os.system("



import subprocess

# by running this command, you don't suspend the python script from running
test = subprocess.Popen(["echo", "test"])

# this script would have the python script wait till the os interaction finishes
test.wait()
