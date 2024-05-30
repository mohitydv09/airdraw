import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2

import imutils


# Defining the model class
class airdraw(nn.Module):
    def __init__(self, backbone, freeze_backbone = True):
        super(airdraw, self).__init__()
        self.backbone = backbone            ## Output is Channnels : 2048, Size is 14*14 for a image size of 448,448
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), ## 1024, 14, 14
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), ## 1024, 14, 14
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), ## 1024, 7, 7
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), ## 1024, 7, 7
            nn.LeakyReLU(0.1),
            nn.Flatten(), ## 512*7*7
            nn.Linear(512*7*7, 2028), ## 2028
            nn.LeakyReLU(0.1),
            nn.Linear(2028, 7*7*24), ## 1176
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(-1, 2048, 14, 14)
        x = self.predictor(x)
        x = x.reshape(-1, 7, 7, 24)
        return x




def get_bb(output,img_wid,img_hgt): ## Ouput will be of shape 1,7,7,24 
    ## Drop the first dimension.
    output = output.squeeze(0)
    bbx_pred_idx_x, bbx_pred_idx_y = torch.where(output[..., 19] > 0.5)
    
    bboxes = []
    for i in range(bbx_pred_idx_x.shape[0]):
        
        #normalised starting values of row and column of a given grid
        row = bbx_pred_idx_x[i].item()
        col = bbx_pred_idx_y[i].item()
        
        m= torch.max(output[row,col,0:19])
        m_ind=(output[row,col,0:19] == m).nonzero(as_tuple=True)[0].item()
        
        if m_ind in [4,7]:
            res=1
        elif m_ind in [8,11]:
            res=2
        else:
            res=0
        
        
        x, y, w, h = output[row, col, 20:24]
        x, y, w, h = x.item(), y.item(), w.item(), h.item()
        # print(x,y,w,h)
        
        # adding col and row to upscale from grid to image level
        x = (x + col)/7
        y = (y + row)/7
        w = w/7
        h = h/7

        ## TopLeft point
        x = x - w/2
        y = y - h/2
        
        
        ## BottomRight point
        x2 = x + w
        y2 = y + h

        ## Unnormalize the points.
        x = int(x*img_wid)
        y = int(y*img_hgt)
        x2 = int(x2*img_wid)
        y2 = int(y2*img_hgt)
        
        bboxes.append([x, y, x2, y2,res])
        
    return bboxes




def main():
    
    try:
        
        #Import and freeze the backbone
        model_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the last two layers
        model_resnet.avgpool = nn.Identity()
        model_resnet.fc = nn.Identity()

        model = airdraw(model_resnet, freeze_backbone=True)

        #loading pretrained model and moving it to GPU
        model = torch.load(r"resnet_based_cnn.pth")
        model.to(device)
        model.eval()

        cap = cv2.VideoCapture(0)

        #setting video camera 
        wid= 1280 #cap.get(3) #1280
        hgt= 720 #cap.get(4) #720
        
        print(wid,hgt)
        cap.set(3,wid)
        cap.set(4,hgt)
        
        #resizing the dimensions of header images to  match the videocapture window
        header_r = cv2.imread(r'1.png')
        # header_r= imutils.resize(header_r,width=wid,height=hgt)

        header_g = cv2.imread(r'2.png')
        # header_g= imutils.resize(header_g,width=wid,height=hgt)

        header_b = cv2.imread(r'3.png')
        # header_b= imutils.resize(header_b,width=wid,height=hgt)

        header_e = cv2.imread(r'4.png')
        # header_e= imutils.resize(header_e,width=wid,height=hgt)
        
        #we use color for both coloring and identifying coloring mode
        color=(255,0,0)
        #storing previous locations of left corner of our drawing pose bounding box
        xp, yp = 0,0
        
        #imgcanvas is our virtual canvas which stores our drawing
        imgCanvas = np.zeros((hgt, wid,3), np.uint8  )
        
        
        while cap.isOpened():
          success, image = cap.read()
          img_hgt,img_wid,_=image.shape
          # print(img_wid,img_hgt)
          
          #resizing the dimensions of header images to  match the videocapture window
          header_r= imutils.resize(header_r,width=img_wid,height=img_hgt)
          header_g= imutils.resize(header_g,width=img_wid,height=img_hgt)
          header_b= imutils.resize(header_b,width=img_wid,height=img_hgt)
          header_e= imutils.resize(header_e,width=img_wid,height=img_hgt)
          
          if not success:
              break
          image = cv2.flip(image, 1)
          
          # image[0:60,0:640] = header[0:60,0:640]

          ## Process the image to transform it for model input
          model_input = cv2.resize(image, (448, 448))
          model_input = transforms.ToTensor()(model_input)
          
          model_input = model_input.unsqueeze(0)
          model_input = model_input.to("cuda")

          ## Send the image to model.
          with torch.no_grad():
              result = model(model_input)
          right=[]
          right.append(result.to('cpu'))
          bboxes = get_bb(result,img_wid,img_hgt)
          
          #acquiring current color and setting the header
          if color==(0,0,255):
              image[0:int(np.ceil(80*img_hgt/480)),0:img_wid] = header_r[0:int(np.ceil(80*img_hgt/480)),0:img_wid]
          elif color == (0,255,0):
              image[0:int(np.ceil(80*img_hgt/480)),0:img_wid] = header_g[0:int(np.ceil(80*img_hgt/480)),0:img_wid]
          elif color ==(255,0,0):
              image[0:int(np.ceil(80*img_hgt/480)),0:img_wid] = header_b[0:int(np.ceil(80*img_hgt/480)),0:img_wid]
          elif color ==(0,0,0):
                  image[0:int(np.ceil(80*img_hgt/480)),0:img_wid] = header_e[0:int(np.ceil(80*img_hgt/480)),0:img_wid]
          
            
          
          for box in bboxes:
              # box has x,y of top left corner of bounding box followed by x,y of buttom right corner of bounding box and 
              # then class of pose 1 for draw, 2 for start/stop drawing, 0 for other poses
              
             #drawing bounding boxes
             image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
             
             if box[4]==1:
                #drawing mode
                #changing the initial cordinates from zero to current position
                if xp==0 and yp==0:
                    xp, yp = int(box[0]), int(box[1])
                if color == (0,0,0):
                    #drawing line from previous point to current point
                    cv2.line(imgCanvas, (xp, yp) , (int(box[0]), int(box[1])), color, 50)
                    #point to indicate the eraser
                    image = cv2.circle(image,(int(box[0]), int(box[1])), 50, (172,168,166), -1)
                else:
                    #drawing line from previous point to current point
                    cv2.line(imgCanvas, (xp, yp) , (int(box[0]), int(box[1])), color, 10)
                xp, yp = int(box[0]), int(box[1])
                
             elif box[4]==2:
                #selection mode
                if int(box[1]) <=120:
                    #red color
                    if int(box[0]) >=200 and int(box[0]) <= 280:
                        xp, yp = 0,0
                        color= (0,0,255)
                    #green color
                    elif int(box[0]) >400 and int(box[0]) <= 480:
                        xp, yp = 0,0
                        color=(0,255,0)
                    #blue color
                    elif int(box[0]) >=600 and int(box[0]) <= 680:
                        xp, yp = 0,0
                        color=(255,0,0)
                    #eraser
                    elif int(box[0]) >=820 and int(box[0]) <= 950:
                        xp, yp = 0,0
                        color=(0,0,0)
                    #clear all
                    elif int(box[0]) >=1000:
                        imgCanvas = np.zeros((hgt, wid,3), np.uint8)
                
          #imgGray to locate all colors 
          imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
          _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
          
          imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
          
          
          
          imgInv = cv2.resize(imgInv, (image.shape[1], image.shape[0]))
          imgCanvas = cv2.resize(imgCanvas, (image.shape[1], image.shape[0]))
          
          #to remove the pixels of image where we have drawn
          image = cv2.bitwise_and(image, imgInv)
          #drawing our canvas on image
          image = cv2.bitwise_or(image, imgCanvas)
          
          image= imutils.resize(image,width=wid,height=hgt)
          
          cv2.imshow("Image",image)
          
          
          #clearing the data on GPU 
          del result
          del model_input
          torch.cuda.empty_cache()
              
              
          # print(image.shape)
          if cv2.waitKey(10) & 0xFF == ord('q'):
              break

    finally:
        #clearing the data on GPU 
        del model
        torch.cuda.empty_cache()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        torch.cuda.memory_summary(device=None, abbreviated=False)
        import gc
        torch.cuda.empty_cache()
        gc.collect()

if __name__=='__main__':
    main()
    
    
    
    


