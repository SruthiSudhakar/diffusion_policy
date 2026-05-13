import os

dpPick = 0
dpPlace = 0
vlmPick = 0
vlmPlace = 0
total = 0
dirs = ["/home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.06/23.17.10_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0200-train_loss=0.0116", "/home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.06/23.17.10_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/epoch=0150-train_loss=0.0160"]
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
