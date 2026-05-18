import os, sys

dpPick = 0
dpPlace = 0
vlmPick = 0
vlmPlace = 0
pcPick = 0
pcPlace = 0
total = 0
dirs = sys.argv[1:]
for bigdir in dirs:
    for dir in os.listdir(bigdir):
        # if int(dir.split('_')[0])<=10:
        if "poorcritic" in dir: 
            sf = dir.split('_')[2]   
            if sf[0] == "s":
                pcPick += 1
            if sf[1] == "s":
                pcPlace += 1
        elif "VLM" in dir:
            sf = dir.split('_')[1]
            if sf[0] == "s":
                vlmPick += 1
            if sf[1] == "s":
                vlmPlace += 1
        else:
            sf = dir.split('_')[1]
            if sf[0] == "s":
                dpPick += 1
            if sf[1] == "s":
                dpPlace += 1
            total += 1

print(f"dpPick: {dpPick}, dpPlace: {dpPlace}, vlmPick: {vlmPick}, vlmPlace: {vlmPlace}, pcPick: {pcPick}, pcPlace: {pcPlace}")
print(f"total: {total}")
