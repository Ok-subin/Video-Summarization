import os


videoDir = "datasets/tvsum/video_5"
videoNames = os.listdir(videoDir)


i = 1

for name in videoNames:
    src = os.path.join(videoDir, name)
    newName = str(i) + ".jpg"
    newName = os.path.join(videoDir, newName)
    os.rename(src, newName)
    i += 1