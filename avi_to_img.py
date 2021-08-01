import cv2
vidcap = cv2.VideoCapture('video_out.avi')
success,image = vidcap.read()
count = 0

while success:
  cv2.imwrite("data/img/%d.jpg" % int(count/10), image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1