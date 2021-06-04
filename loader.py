import os
import numpy as np


def fetch_videos(video_path):
    videos=[]
    with open(video_path, 'r') as file:
        for line in file:
            videos.append(line.strip())
    return videos


def fetch_training_data(training_data_path): #returns just video frames

    data_path_abnormal =training_data_path

    data_path_normal = training_data_path

    batch_size = 30

    abnormals = 170

    normals =160

    abnormal_list = np.random.permutation(abnormals)
    abnormal_list=abnormal_list[:batch_size]

    normal_list = np.random.permutation(normals)
    normal_list = normal_list[:batch_size]
    
    videos = fetch_videos(data_path_abnormal+"/anomaly.txt")
    

    abnormal_features=[]
    for i in abnormal_list:
        vid_path=os.path.join(data_path_abnormal,videos[i] )
        with open(vid_path,'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            abnormal_features.append(np.float32(line.split()))

    abnormal_features=np.array(abnormal_features)    
    
    videos = fetch_videos(data_path_normal+"/normal.txt")
    normal_features=[]

    for i in normal_list:
        vid_path=os.path.join(data_path_normal,videos[i] )
        if (os.path.isfile(vid_path) ):
            with open(vid_path,'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                  normal_features.append(np.float32(line.split()))

    normal_features=np.array(normal_features)
    
    return abnormal_features, normal_features


def fetch_testing_data(test_set_path): # returns videos with dictionary
    data_path_abnormal = test_set_path

    data_path_normal = test_set_path

    batch_size = 30

    abnormals = 170

    normals =160

    abnormal_list = np.random.permutation(abnormals)
    abnormal_list=abnormal_list[:batch_size]

    normal_list = np.random.permutation(normals)
    normal_list = normal_list[:batch_size]

    videos = fetch_videos(data_path_abnormal+"/anomaly.txt")
    
    abnormal_features ={}

    for i in abnormal_list:
        video_path=os.path.join(data_path_abnormal,videos[i] )
        with open(video_path,'r') as f:
            lines = f.read().splitlines()
            key=os.path.basename(videos[i])
            key=os.path.splitext(key)[0]
            #print(key)
            abnormal_features[key]=[]
        for line in lines:
            abnormal_features[key].append(np.float32(line.split()))
        abnormal_features[key]=np.array(abnormal_features[key])
    
    videos = fetch_videos(data_path_normal+"/normal.txt")
    normal_features={}

    for i in normal_list:
        video_path=os.path.join(data_path_normal,videos[i] )
        if (os.path.isfile(video_path) ):
            with open(video_path,'r') as f:
                lines = f.read().splitlines()
                key=os.path.basename(videos[i])
                key=os.path.splitext(key)[0]
                normal_features[key]=[]
            for line in lines:
                  normal_features[key].append(np.float32(line.split()))
            
            normal_features[key]=np.array(normal_features[key])

    return abnormal_features, normal_features


def extract_video_features(video_name, video_path):
    
    videos = fetch_videos(video_path+"/features.txt")
        
    for i,video in enumerate(videos):
        
        if video_name in video:
            vid_path=os.path.join(video_path,videos[i] )
            with open(vid_path,'r') as f:
                lines = f.read().splitlines()
            features=[]
            for line in lines:
                features.append(np.float32(line.split()))
            return np.array(features)



def get_test_data_normal():
    path="./data"
    videos = fetch_videos("./data/testing.txt")
    
    features =[]
    for i,video in enumerate(videos):
        vid_path=os.path.join(path,video)
        with open(vid_path,'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            features.append(np.float32(line.split()))
    
    return np.array(features)
        
