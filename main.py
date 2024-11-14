import cv2
import numpy as np
import matplotlib.pyplot as plt

print("This is a testing program using OpenCV library");
print(cv2.__version__)

img = cv2.imread("shiba-imu.jpg");

if img is None:
    print("Image is not loaded correctly");
else:
    print("OpenCV is installed and working correctly!");
    
# cv2.imshow("Shiba Imu", img);
# cv2.waitKey(0);

# plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 

#RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#GRAYSCALE_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
#edges = cv2.Canny(image = RGB_img, threshold1 = 100, threshold2 = 100)

# Create subplots
#fig, axs = plt.subplots(1, 2, figsize = (7, 4))

# Plot the original image
#axs[0].imshow(RGB_img)
#axs[0].set_title('Original Image')

# Plot the blurred image
#axs[1].imshow(edges)
#axs[1].set_title('Image edges')

# Remove ticks from the subplots
#for ax in axs:
#    ax.set_xticks([])
#    ax.set_yticks([])



# Display the subplots
#plt.tight_layout()
plt.plot(histr)
plt.show()


# Display using matplotlib.pyplot
# plt.imshow(GRAYSCALE_img)
# plt.waitforbuttonpress()
# plt.close('all')