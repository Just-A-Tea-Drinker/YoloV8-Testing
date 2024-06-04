from ultralytics import YOLO
import cv2
#loading the model
model = YOLO(r'D:\Other apps\Python\runs\detect\train\weights\last.pt')
frame = cv2.imread(r"C:\Users\antho\Downloads\brain-tumor.jpg")
results = model(frame)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen                   


# cv2.imshow("IMAGE",frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()