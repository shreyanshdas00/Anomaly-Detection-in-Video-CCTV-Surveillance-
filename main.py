import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import imageio
from PIL import Image,ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import smtplib
import base64
from network import *
from loader import *
from utility_functions import *
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

loc,video,delay="",None,0

ph = tf.placeholder('float', [None, None])
output , parameters_1 , parameters_2, parameters_3 = define_network(ph)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saved_model_loader = tf.train.Saver()  # to load the saved model

saved_model_loader.restore(sess, './model/model')
f_nos,red_frames=0,[]

def detect_anomalous_segment(input_video,input_video_name):
    max_processing_sec = 599
    frame_increase=16
    num_frames = 16
    start_frame = 1
    
    frame_count,frame_rate=count_frames(input_video)
    
    end_frame = min(frame_count, int(max_processing_sec * frame_rate)) -num_frames
    
    features = extract_video_features(input_video_name,"./data")
    model_output=sess.run(output,feed_dict={ph: features})
    
    start_frames=list(range(start_frame, end_frame, frame_increase))
    
    frame_bin=np.round(np.linspace(1, len(start_frames), num=33))
    
    model_output=model_output.flatten()
    predicted=np.where(model_output>0.4)[0]
    
    video_segments=[]
    i=0

    
    while(i<len(predicted)-1):
        segment_start=frame_bin[predicted[i]]*16
        if (predicted[i+1] - predicted[i] ==1):
            while(i<len(predicted)-1 and predicted[i+1]-predicted[i]==1):
                i +=1
            segment_end=frame_bin[predicted[i]]*16+16
        else:
            i += 1
            segment_end=segment_start+16
        video_segments.append([segment_start,segment_end])
        
    video_map=frame_bin[predicted]*16
    return video_segments,video_map,model_output 
    

def get_anomalous_segment_from_video(input_video,input_video_name,output_dir):
    video_segments, video_map, model_output=detect_anomalous_segment(input_video,input_video_name)
    
    #path to store Video segment
    path = os.path.join(output_dir,input_video_name)
    os.makedirs(path, exist_ok=True)
    
    for start, end in video_segments:
        num_frames = int(end-start)
        get_video_frames(input_video,start,path,num_frames)
     
    output_video=os.path.join(path,input_video_name)
    if len(video_segments)>0:
        generate_video_from_image(path,output_video+".avi")
    
    return video_map, model_output


def send_email(filename, frames):
    from_addr = 'cctvsurveillanceApp@gmail.com'
    password = 'Test0123'
    # input receiver email address.
    to_addr = 'shreyanshdas00@gmail.com'
    # input smtp server ip address:
    smtp_server = 'smtp.gmail.com'
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = Header('UNUSUAL ACTIVITY ALERT!!', 'utf-8').encode()
    
    # attache a MIMEText object to save email 
    for frame in frames:
        msg_content = MIMEText('Unusual activity detected at time: '+str(frame/delay)+' sec\n', 'plain', 'utf-8')
        msg.attach(msg_content)
    msg_content = MIMEText('Please find attached the captured images of the unusual activity.', 'plain', 'utf-8')
    msg.attach(msg_content)
    i=0
    # to add an attachment is just add a MIMEBase object to read a picture locally.
    for file in glob.glob('./frames/'+filename+'/*.jpg'):
        with open(file, 'rb') as f:
            # set attachment mime and file name, the image type is png
            mime = MIMEBase('image', 'jpg', filename=file+'.jpg')
            # add required header data:
            mime.add_header('Content-Disposition', 'attachment', filename='img'+str(i)+'.png')
            mime.add_header('X-Attachment-Id', str(i))
            mime.add_header('Content-ID', '<'+str(i)+'>')
            # read attachment file content into the MIMEBase object
            mime.set_payload(f.read())
            # encode with base64
            encoders.encode_base64(mime)
            # add MIMEBase object to MIMEMultipart object
            msg.attach(mime)
            i+=1
            
    server = smtplib.SMTP(smtp_server,587)
    server.ehlo()
    server.starttls()
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()
    
fct_id=-1
def stream():
    global fct_id,f_nos
    try:
        image=video.get_next_data()
        frame_image=Image.fromarray(image)
        frame_image=ImageTk.PhotoImage(frame_image)
        label_file_selected.config(image=frame_image)
        label_file_selected.image=frame_image
        f_nos+=1
        if red_frames and f_nos in red_frames:
            label_disp.configure(bg="red")
            fct_id+=1
        elif red_frames:
            if fct_id<len(red_frames) and f_nos>red_frames[fct_id]+25:
                label_disp.configure(bg="green")
        label_file_selected.after(delay,lambda: stream())
    except:
        video.close()
        label_file_selected.config(text="Please choose a different file")
        return
        

def browseFiles(): 
    global loc,video,delay,f_nos,red_frames
    label_file_selected.config(text="Processing...")
    f_nos=0
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("Video files", 
                                                        "*.mp4*"), 
                                                       ("all files", 
                                                        "*.*")))        
    loc= filename
    
    video=imageio.get_reader(loc)
    delay= int(1000/video.get_meta_data()['fps'])
    anomalous_frames,model_output=[],0
   
    try:
        video_name = ""
        name_start1 = loc.rfind("/")
        name_start2 = loc.rfind("\\")
        if (name_start1==-1 and name_start2==-1):
            video_name = loc.replace('.mp4', '')
        elif (name_start1==-1 and name_start2>-1):
            video_name = loc[name_start2+1:].replace('.mp4', '')
        elif (name_start1>-1 and name_start2==-1):
            video_name = loc[name_start1+1:].replace('.mp4', '')
        anomalous_frames, model_output=get_anomalous_segment_from_video(loc,video_name,"./frames")
    except:
        anomalous_frames = []
        model_output = 0
   
    if len(anomalous_frames)>0:
        label_file_explorer.config(text="Anomalies found! Check Email for specific frames.",fg="red",font=("Helvetica",12))
        send_email(video_name, anomalous_frames)
        red_frames=anomalous_frames.copy()
        red_frames= list(map(int,red_frames))
    stream()
    

def close():
    global f_nos
    f_nos=0
    if video is not None:
        video.close()
        
   
       
if __name__ == '__main__':   
                                                                                               
    window = tk.Tk() 
    window.title('Surveillance') 
    window.geometry("600x400") 
    window.config(background = "black") 
       
    label_file_explorer = tk.Label(window,height=2
                                ,text = "Find Suspicious Activities!", 
                                 font=("Helvetica",18),fg="blue",bg="Black") 
       
    label_file_selected = tk.Label(window, text="Select a Video",bg="Black"
                                   ,fg="red",font=("Helvetica",12))
    
    button_explore = ttk.Button(window,  
                            text = "Browse Files", 
                            command = browseFiles)
    
    label_disp = tk.Label(window, text="", bg="green",relief="sunken",width=10)
    button_close_video= ttk.Button(window, text="Stop",
                                   command=lambda: close())   
    
    label_file_explorer.pack(fill=tk.X) 
    label_file_selected.pack(fill=tk.X,pady=20)
    button_explore.place(x=170,y=350) 
    label_disp.place(x=270,y=350)
    button_close_video.place(x=370,y=350) 
       
    window.mainloop() 
