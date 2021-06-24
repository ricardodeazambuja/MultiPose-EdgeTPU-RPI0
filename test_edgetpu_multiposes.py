import numpy as np

from multipose import decodeMultiplePoses

import picamera

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    print("Did you install the TFLite Runtime? https://www.tensorflow.org/lite/guide/python")
    
    
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]        

edgetpu_model_file = 'downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite'
    
import io
import time
import numpy as np


class EdgeTPU_MultiPoses(io.IOBase):
    '''
    Capturing Image from a Raspicam (V2.1)
    '''
    def __init__(self, frameWidth=257, frameHeight=257, scoreThreshold=0.5, debug = False):
        # Init the stuff we are inheriting from
        super(__class__, self).__init__()

        self.debug = debug

        # Set video frame parameters
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        
        self.scoreThreshold = scoreThreshold
        self.poses = []

        self.prev_time = time.time()
        
        #
        # EdgeTPU Accelerator
        #
        device = [] # I have only one USB accelerator...
        self.tflite_interpreter = tflite.Interpreter(model_path=edgetpu_model_file, 
                                                experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})])
        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()
        
        
        # Picamera doesn't like 257x257, so the image will be padded
        self.final_output = np.zeros((257,257,3), dtype=np.uint8)
        self.tmp_output = np.empty((frameWidth, frameHeight, 3), dtype=np.uint8)

        preprocessed_img = np.expand_dims((self.final_output.astype('float32')-127).astype('int8'), axis=0)

        # The first time is slower, so let's do it here!
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'], preprocessed_img)
        self.tflite_interpreter.invoke()



    def writable(self):
        '''
        To be a nice file, you must have this method
        '''
        return True

    def write(self, b):
        '''
        Here is where the image data is received and made available at self.output
        '''

        try:
            # b is the numpy array of the image, 3 bytes of color depth
            img = np.reshape(np.frombuffer(b, dtype=np.uint8), (self.frameHeight, self.frameWidth, 3))
            
            self.get_poses(img)

            if self.debug:
                print(f"EdgeTPU_MultiPoses - Poses found: {len(self.poses)}")
                print("EdgeTPU_MultiPoses - Pose scores:", [f"{(i['score']):2.2f}" for i in self.poses])
                print("EdgeTPU_MultiPoses - Running at {:2.2f} Hz".format(1/(time.time()-self.prev_time)))

            self.prev_time = time.time()

        except Exception as e:
            print("ImgCap error: {}".format(e))
        finally:
            return len(b)
        
    def get_poses(self, img):        
        self.final_output.fill(0)
        self.final_output[:self.frameWidth, :self.frameHeight, :] = img

        preprocessed_img = np.expand_dims((self.final_output.astype('float32')-127).astype('int8'), axis=0)
        # The first time is slower
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'], preprocessed_img)
        self.tflite_interpreter.invoke()


        formated_outputs = {"heatmap": {}, 'offset': {}, 'displacement_fwd': {}, 'displacement_bwd': {}}
        for o in self.output_details:
            for name in formated_outputs.keys():
                if name in o['name']:
                    # https://www.tensorflow.org/lite/performance/quantization_spec
                    zero_point = o['quantization'][1]
                    scale = o['quantization'][0]
                    formated_outputs[name] = (self.tflite_interpreter.get_tensor(o['index'])[0].astype(float)-zero_point)*scale

        self.poses = decodeMultiplePoses(formated_outputs['heatmap'], 
                                         formated_outputs['offset'], 
                                         formated_outputs['displacement_fwd'], 
                                         formated_outputs['displacement_bwd'],
                                         scoreThreshold=self.scoreThreshold)


if __name__ == "__main__":
    frameWidth = 256 
    frameHeight = 256
    frameRate = 20

    # Set the picamera parametertaob
    camera = picamera.PiCamera()
    camera.resolution = (frameWidth, frameHeight)
    camera.framerate = frameRate

    # Start the video process
    with EdgeTPU_MultiPoses(frameWidth, frameHeight, debug=True) as img:
        camera.start_recording(img, format='rgb', splitter_port = 1)
        try:
            while True:
                camera.wait_recording(timeout=1/frameRate) # using timeout=0, default, it'll return immediately  

        except KeyboardInterrupt:
            pass
        finally:
            camera.stop_recording(splitter_port = 1)