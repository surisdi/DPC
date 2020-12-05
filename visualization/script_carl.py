import json
from pprint import pprint
import shutil, os
import random

# Read labels
labelmap = {}
for line in open("/proj/vondrick/datasets/Hollywood2/class_Ind/class_Ind_Hier.txt"):
    name,id,lvl = line.strip().split()
    labelmap[int(id)] = name
pprint(labelmap)

# Read hierarchy
parents = {}
for line in open("/proj/vondrick/datasets/Hollywood2/class_Ind/class_Relation.txt"):
    parent,child = line.strip().split()
    parents[child] = parent

# Load hollywood
data = json.load(open('/proj/vondrick/didac/results/hollywood2_data.json'))

# Shuffle data so we don't bias by filename order
random.seed(0)
random.shuffle(data)

# Output file
fd = open("index.html", "w")
fd.write("<div style='white-space: nowrap'>") # disable word-wrap

for id, (p1,p2, gt1,gt2, radius, filename) in enumerate(data):
    print(filename)

    # Copy frames for visualization
    for fr in range(1,30,5):
        shutil.copyfile(os.path.join(filename, "image_%05d.jpg"%fr), "img/%05d_fr%05d.jpg"%(id,fr))
        fd.write("<img src='img/%05d_fr%05d.jpg'>"%(id,fr))

    fd.write("<br>")
    fd.write("Pred: %s ---> %s<br>" % (parents[labelmap[p1]], labelmap[p1]))
    fd.write("GT: %s ---> %s<br>" % (parents[labelmap[gt1]], labelmap[gt1]))
    fd.write("Radius: %s<br>" % radius)
    fd.write("Filename: %s<br>" % filename)
    fd.write("<hr>\n")
    fd.flush()

    # Stop after 50 videos
    if id > 50:
        break

fd.write("</div>")