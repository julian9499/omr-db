from os import listdir
from os.path import isfile, join
from dbiyo_main import getScore
path = "./toscan"
onlyfiles = [join("./toscan",f) for f in listdir("./toscan") if isfile(join("./toscan", f))]
print(onlyfiles)
for i in onlyfiles:
    getScore(i)
