import os
import glob

save_dir = "/home/fourwheel/Desktop/project/code/video"
files = glob.glob(os.path.join(save_dir, "video_*.jpg"))

straight = 0
left_soft = 0
left_hard = 0
right_soft = 0
right_hard = 0
others = 0

for f in files:
    basename = os.path.basename(f)
    if "L+050_R+050" in basename:
        straight += 1
    elif "L+030_R+050" in basename:
        left_soft += 1
    elif "L+005_R+050" in basename:
        left_hard += 1
    elif "L+050_R+030" in basename:
        right_soft += 1
    elif "L+050_R+005" in basename:
        right_hard += 1
    else:
        others += 1

total = len(files)
left_total = left_soft + left_hard
right_total = right_soft + right_hard

print(f"Total files: {total}")
print(f"Straight (직진): {straight} ({straight/total*100:.2f}%)")
print(f"Left (좌회전): {left_total} ({left_total/total*100:.2f}%)")
print(f"  - Soft Left: {left_soft}")
print(f"  - Hard Left: {left_hard}")
print(f"Right (우회전): {right_total} ({right_total/total*100:.2f}%)")
print(f"  - Soft Right: {right_soft}")
print(f"  - Hard Right: {right_hard}")
print(f"Others: {others}")
