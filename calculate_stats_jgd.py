import os, sys

dpPick = 0
dpPlace = 0
vlmPick = 0
vlmPlace = 0
total = 0
dirs = sys.argv[1:]
for bigdir in dirs:
    for dir in os.listdir(bigdir):
        sf = dir.split('_')[1]
            
        if "VLM" in dir:
            if sf[0] == "s":
                vlmPick += 1
            if sf[1] == "s":
                vlmPlace += 1
        else:
            if sf[0] == "s":
                dpPick += 1
            if sf[1] == "s":
                dpPlace += 1
            total += 1

print(f"dpPick: {dpPick}, dpPlace: {dpPlace}, vlmPick: {vlmPick}, vlmPlace: {vlmPlace}")
print(f"total: {total}")
