__author__ = "kdhht5022@gmail.com"
import numpy as np
from PIL import Image
import time, os
from multiprocessing import Queue, Value
import configparser, uuid

import emoji

import asyncio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
from fabulous.color import fg256

from my_util.videomgr import VideoMgr
from my_util.detect_util import draw_results_ssd
from my_util.fer_util import nn_output


async def async_handle_video(camInfo):
    videoSaveOut = None
    config = camInfo["conf"]    
    VideoHandler = VideoMgr(int(config['url']), config['name'])        
    VideoHandler.open(config)
   
    loop = asyncio.get_event_loop()

    global key
    global f
    global image_queue
    global cam_check
    
    global do_hand_gesture
    
    global image_batch
    
    global valence, arousal
    global fd_signal
    
    global emotion_list  # Pseudo-discrete emotion labels ["angry", "sad", "happy", "pleased", "neutral"]
    global emot_region
    
    def sleep():
        time.sleep(0.02)
        return 
    def resize(img, size):
        return cv2.resize(img, size)
    try:
        f = 0  # just for counting :)
        while(True):
            if cam_check[cname[VideoHandler.camName]] == 0:
                ret, orig_image = await loop.run_in_executor(None, VideoHandler.camCtx.read)
                if cname[VideoHandler.camName]%2 == 1:
                    orig_image = np.fliplr(orig_image)
                    
                # Trigger using some condition
                if f % 2 == 0:
                    do_hand_gesture = True
                    if len(image_batch) > 10: 
                        image_batch[:5] = []
                    image_batch.append(orig_image)
                    
                    if type(valence) is torch.Tensor:  # type(valence) is not np.ndarray or 
                        valence = valence.detach().cpu().numpy()
                        arousal = arousal.detach().cpu().numpy()
                    if np.abs(valence) < 0.1 and np.abs(arousal) < 0.1:
                        final_emot = emotion_list[4]  # neutral
                    elif valence > 0.1 and arousal > 0.2:
                        final_emot = emotion_list[2]  # happy
                    elif valence < -0.1 and arousal > 0.1:
                        final_emot = emotion_list[0]  # angry
                    elif valence < -0.1 and arousal < -0.1:
                        final_emot = emotion_list[1]  # sad
                    elif valence > 0.1 and arousal < -0.1:
                        final_emot = emotion_list[3]  # pleased
                        
                    if np.sign(valence) == 1 and np.sign(arousal) == 1:
                        emot_region = "1R"
                    elif np.sign(valence) == -1 and np.sign(arousal) == 1:
                        emot_region = "2R"
                    elif np.sign(valence) == -1 and np.sign(arousal) == -1:
                        emot_region = "3R"
                    elif np.sign(valence) == 1 and np.sign(arousal) == -1:
                        emot_region = "4R"
                    
                if (config['camshow'] == 'on'):
                    valence_value = np.round(float(valence),2)
                    arousal_value = np.round(float(arousal),2)
                    
                    cv2.rectangle(orig_image, (0,0), (430,600), (255,255,255), -1);  cv2.rectangle(orig_image, (0,0), (430,600), (0,0,0), 3)
                    
                    cv2.rectangle(orig_image, (50,150), (300,400), (0,0,0), -1)
                    cv2.line(orig_image, (50,275), (300,275), (255,255,255), 1)
                    cv2.line(orig_image, (175,150), (175,400), (255,255,255), 1)
                    cv2.putText(orig_image, 'Happy', (200, 212), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
                    cv2.putText(orig_image, 'Pleased', (200, 337), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
                    cv2.putText(orig_image, 'Angry', (85, 212), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
                    cv2.putText(orig_image, 'Sad', (85, 337), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
                    
                    cv2.line(orig_image, (175+int(valence_value*150),275-int(arousal_value*150)), (175+int(valence_value*150),275-int(arousal_value*150)), (0,255,0), 7)  # point
                    
                    cv2.putText(orig_image, 'Valence: {0:.2f}'.format(valence_value), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(orig_image, 'Arousal: {0:.2f}'.format(arousal_value), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(orig_image, 'Discrete emotion: {}'.format(final_emot), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    cv2.putText(orig_image, 'Arousal-Valence space', (5, 440), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2)
                    cv2.putText(orig_image, 'Horizontal axis: Valence', (5, 480), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1)
                    cv2.putText(orig_image, 'Vertical axis: Arousal', (5, 520), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1)
                    cv2.putText(orig_image, 'ELIM pre-trained weights @ INHA', (5, 575), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2)
                    if fd_signal == 1:
                        cv2.putText(orig_image, 'Face is detected', (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                    elif fd_signal == 0:
                        cv2.putText(orig_image, 'Face is NOT detected', (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    cv2.imshow(config['name'], orig_image)

                    if VideoHandler.camName == '1th_left':
                        key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                f += 1
                    
            else:
                await loop.run_in_executor(None, sleep)
                if sum(cam_check) == 4:
                    cam_check = [0,0,0,0]


        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()

    except asyncio.CancelledError:        
        if config['tracking_flag'] == 'off' and config['videosave'] == 'on':        
            videoSaveOut.release()
            
        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()


async def handle_video_analysis():    
    def hand_detection():

        global hand_gesture_config
        global hand_gesture_sleep
        global do_hand_gesture
        
        global image_batch
        
        global faces
        global net
        global encoder, regressor, task_header
        global f
        
        global valence, arousal
        global fd_signal
        
        offset_v, offset_a = 0.55, 0.25  # manual offset to origin alighment
        if do_hand_gesture == 1:
            
            input_img = image_batch[-1]
            img_h, img_w, _ = np.shape(input_img)
            
            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()

            faces = np.empty((detected.shape[2], 224, 224, 3))
            cropped_face, fd_signal = draw_results_ssd(detected,input_img,faces,0.1,224,img_w,img_h,0,0,0)
        
            croppted_face_tr = torch.from_numpy(cropped_face.transpose(0,3,1,2)[0]/255.)
            cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(croppted_face_tr)
    
            latent_feature = encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.cuda.FloatTensor))
            va_output = task_header(regressor(latent_feature))
            
            valence = va_output.detach().cpu().numpy()[0][0] + offset_v
            arousal = va_output.detach().cpu().numpy()[0][1] + offset_a
            

            do_hand_gesture = False
        else:
            hand_gesture_sleep = True

    loop = asyncio.get_event_loop()
    try:
        while(True):
            await loop.run_in_executor(None, hand_detection)
            if key == ord('q'):
               break
    except asyncio.CancelledError:        
        pass
        
        
# agent algorithm
async def Agent():
    global action

    def agent():
        
        global open_algorithm
        global action
        global something
        global key
        
        if open_algorithm == 1:
            something = time.time()
        if something > 0:
            if time.time() - something > 0.8:
                something = 0
                action = 'something'

          
        # re-initialize global variables
        time.sleep(2e-2)
        action = None

    loop = asyncio.get_event_loop()
    try:
        while(True):
            await loop.run_in_executor(None, agent)
            if key == ord('q'):
               break

    except asyncio.CancelledError:        
        pass


async def async_handle_video_run(camInfos):  
    futures = [asyncio.ensure_future(async_handle_video(cam_info)) for cam_info in camInfos]\
             +[asyncio.ensure_future(handle_video_analysis())]\
             +[asyncio.ensure_future(Agent())]

    await asyncio.gather(*futures)


class Config():
    """  Configuration for Label Convert Tool """
    def __init__(self):
        global ini             
        self.inifile = ini
        self.ini = {}
        self.debug = False
        self.camera_count = 0
        self.cam = []        
        self.parser = configparser.ConfigParser()
        self.set_ini_config(self.inifile)        
               
    def set_ini_config(self, inifile):
        self.parser.read(inifile)
        
        for section in self.parser.sections():
            self.ini[section] = {}
            for option in self.parser.options(section):
                self.ini[section][option] = self.parser.get(section, option)
            if 'CAMERA' in section:
                self.cam.append(self.ini[section])


"""
 *******************************************************************************
 * [CLASS] ELIM FER Demo
 *******************************************************************************
"""
class FER_INT_ALG():

    def __init__(self):
        global ini
        ini = 'config.ini'
        self.Config = Config()
        self.trackingQueue = [Queue() for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count']))]        
        self.vaQueue = Queue()
        self.isReady = Value('i', 0)
        self.camTbl = {}

        global open_algorithm
        global something; global action
        
        global key; global f
        global image_queue; global image_queue_move
        global cam_check
        
        global cname

        global do_hand_gesture
        global hand_gesture_config
        global hand_gesture_sleep
        
        global image_batch
        
        global faces
        faces = np.empty((200, 224, 224, 3))
        global net
        
        global valence, arousal
        valence, arousal = torch.zeros(1), torch.zeros(1)
        global fd_signal
        fd_signal = 1
        
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detector",
            "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        
        global encoder, regressor, task_header
        
        encoder, regressor, task_header = nn_output()
        encoder.load_state_dict(torch.load('weights/enc.t7'), strict=False)
        regressor.load_state_dict(torch.load('weights/reg.t7'), strict=False)
        task_header.load_state_dict(torch.load('weights/header.t7'), strict=False)
        
        global emotion_list
        global emot_region
        emotion_list = ["angry", "sad", "happy", "pleased", "neutral"]
        emot_region = ""
        
        encoder.train(False)
        regressor.train(False)

        open_algorithm = True
        something = 0; action = None
        do_hand_gesture = False

        hand_gesture_sleep = True

        image_queue = []
        image_queue_move = []
        image_batch = []
        key = [0,0,0,0]; f = 0

        # We can set multiple cameras in the future!
        cname = {'1th_left'  : 0,}
#                 '1th_right' : 1,}

        cam_check = [0,0,0,0]
        
    def run(self):        
        camInfoList = []        
        camTbl = {}        
        global key

        for idx in range(0, int(self.Config.ini['COMMON']['camera_count'])):
            camInfo = {}
            camUUID = uuid.uuid4()
            
            camInfo.update({"uuid": camUUID})
            camInfo.update({"isready": self.isReady})
            camInfo.update({"tqueue": self.trackingQueue[idx]})
            camInfo.update({"vaqueue": self.vaQueue})            
            camInfo.update({"conf": self.Config.cam[idx]})
            camInfo.update({"globconf": self.Config})
            
            camInfoList.append(camInfo)
            camTbl.update({camUUID: camInfo})
        
        while (True):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_handle_video_run(camInfoList))
            loop.close()
            if key == ord('q'):
                break
                

    def close(self):
        for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count'])):
            self.trackingQueue[idx].close()
            self.trackingQueue[idx].join_thread()
        self.vaQueue.close()
        self.vaQueue.join_thread()
        

if __name__ == "__main__":
    fer_int_alg = FER_INT_ALG()
    print ('Start... ELIM FER Demo')
    fer_int_alg.run()
    fer_int_alg.close()
    print("Completed ELIM FER Demo")









