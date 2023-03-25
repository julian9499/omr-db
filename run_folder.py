import os
from os import listdir
from os.path import isfile, join
from dbiyo_main import getScore
import shutil

path = "processed"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
   print("Made dir: processed")

path = "toscan"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
   print("Made dir: toscan")

path = "./toscan"
onlyfiles = [f for f in listdir("./toscan") if isfile(join("./toscan", f))]


csvFile = open("results.csv", "x")
csvFile.close()

csvFile = open("results.csv", "a")

print(onlyfiles)
for f in onlyfiles:
    totalFileName = join("./toscan",f)
    csvLine = getScore(totalFileName)
    csvFile.write(csvLine + "\n")
    csvFile.flush()
    shutil.move(totalFileName, join("./processed", f), copy_function=shutil.copy2)

csvFile.close()

