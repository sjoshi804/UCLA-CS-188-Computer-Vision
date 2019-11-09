import numpy as np
import cv2
from skimage import data
from skimage.feature import match_template
import matplotlib.pyplot as plot

#Initializing variables for video
cap = cv2.VideoCapture('video.mp4')
count = 0
frames = []

#Loop through video to break into  frames
while (cap.isOpened()):
    #Print status
    count += 1
    print("Processing frame " + str(count))

    #Read in frame
    ret, frame = cap.read()
    if not ret:
        break

    #Add frame to list
    frames.append(frame)
    #frames.append(cv2.rectangle(frame, (600, 440), (1600, 1076), (0, 0, 255), 2))

    '''
    #Print frame
    name = 'frames/frame' + str(count) + '.jpg'
    print("Writing " + name)
    cv2.imwrite(name, frames[count - 1])
    '''

    #Make image grayscale
    frames[count - 1] = cv2.cvtColor(frames[count - 1], cv2.COLOR_BGR2GRAY)

#Draw rectangle on first frame
template_image = cv2.rectangle(frames[0], (1275, 595), (1370, 710), (0, 255, 0), 2)
print("Printing template on image")
cv2.imwrite("template_image.jpg", template_image)
 
#Initialize counter
count = 0

#Crop template
template_cropped = template_image[595:710, 1275:1370]

#List of tuples
max_i_cood = []
max_j_cood = []

#Template matching
for source in frames:
    count += 1
    source_cropped = source[440:1076, 660:1600]
    print("Matching template for frame " + str(count))
    result = match_template(source_cropped, template_cropped, pad_input=True)

    #Extract max point from each result
    current_max_i = 0
    current_max_j = 0
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            if result[i][j] > result[current_max_i][current_max_j]:
                current_max_i = i
                current_max_j = j
    max_i_cood.append(current_max_i)
    max_j_cood.append(current_max_j)

#Plot intensities for one 'result' of the above template matching
print("Displaying grayscale plot of one 'result'")
plot.imshow(result, cmap="gray")
plot.show()

#Plot maximum intensity point's movement graph
print("Plotting max i and max j across all results")
plot.scatter(max_i_cood, max_j_cood)
plot.show()

final_image = frame[0]
for i in range(1, len(frames)):
    for x in range(1, len(frames[i])):
        for y in range(1, len(frames[i][x])):
            final_image[x][y] += frames[i][x - max_i_cood[i]][y - max_j_cood[i]]

cv2.imwrite("result.jpg", final_image)

#Project cleanup    
cap.release()
cv2.destroyAllWindows()