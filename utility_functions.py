import os
import cv2
import sys
import glob


def count_frames(input_video):
    ''' Get frame counts and FPS for a video '''
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("[Error] video={} can not be opened.".format(input_video))
        sys.exit(-6)

    frame_count = int(video.get(7))
    frame_rate = video.get(5)
    if not frame_rate or frame_rate != frame_rate:
        frame_rate = 29.97
    return frame_count, frame_rate



def get_video_frames(input_video, start_frame, frame_dir, num_frames=16):
    ''' Extract frames from a video using opencv '''

    # check output directory
    if os.path.isdir(frame_dir):
        pass
    else:
        os.makedirs(frame_dir)

    # get number of frames
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print ("[Error] video={} can not be opened.".format(input_video))
        sys.exit(-6)

    # move to start_frame
    video.set(1, start_frame)

    # grab each frame and save
    for frame_count in range(num_frames):
        frame_num = frame_count + start_frame
        success, frame = video.read()
        if not success:
            print ("[Error] Frame extraction was not successful")
            sys.exit(-7)

        frame_file = os.path.join(frame_dir,'{0:06f}.jpg'.format(frame_num))
        cv2.imwrite(frame_file, frame)
    return


def generate_video_from_image(source,output):
    images = []
    for filename in glob.glob(source+'/*.jpg'):
        image = cv2.imread(filename)
        height, width, layers = image.shape
        size = (width,height)
        images.append(image)
    
    output = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    
    for i in range(len(images)):
        output.write(images[i])
    
    output.release()