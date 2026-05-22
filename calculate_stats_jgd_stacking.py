import os, sys

dp = 0
vlm = 0
pc = 0
dp_total = 0
vlm_total = 0
pc_total = 0
dirs = sys.argv[1:]
for bigdir in dirs:
    for dir in os.listdir(bigdir):
        if "poorcritic" in dir: 
            pc_total += 1
        elif "VLM" in dir:
            vlm_total += 1
        else:
            dp_total += 1

        if dir.split('_')[1]!='f':
            if "poorcritic" in dir: 
                sf = dir.split('_')[2]
                if sf[1] == "s":
                    pc += 1
            elif "VLM" in dir:
                sf = dir.split('_')[1]
                if sf[1] == "s":
                    vlm += 1
            else:
                sf = dir.split('_')[1]
                if sf[1] == "s":
                    dp += 1
 
print(f"dp: {dp}, vlm: {vlm}, pc: {pc}")
print(f"dp_total: {dp_total}, vlm_total: {vlm_total}, pc_total: {pc_total}")
