import cv2
import numpy as np 
import os 
import math
import glob
import time

angles_pred=[]
angles_gt=[]
error=[]
store_dir_pred='postprocessed_ternausnet_preds/'
store_dir_gt='postprocessed_gt'
nearZero=0.00001

def getTipDistance(line_pts):
    return math.sqrt((line_pts[0,0]-line_pts[1,0])**2+(line_pts[0,1]-line_pts[1,1])**2)

def shortest_length(mask_x1,mask_y1,mask_x2,mask_y2,im_w,im_h):
    
    mask=dict()
    mask2=dict()
    center=dict()
    eqn2=dict()
    intersect=dict()
    
    mask['x']=mask_x1
    mask['y']=mask_y1
    mask2['x']=mask_x2
    mask2['y']=mask_y2
    center['x']=im_w/2
    center['y']=im_h/2
    
    if mask['x']==mask2['x']:
        grad=float(mask['y']-mask2['y'])/nearZero
    elif mask['y']==mask2['y']:
        grad=nearZero/(mask['x']-mask2['x'])
    else:
        grad=float(mask['y']-mask2['y'])/(mask['x']-mask2['x'])
        
    if grad!=0:
        grad_inv=-1/grad
    else:
        grad_inv=nearZero
    #equation
    eqn2['c']=mask2['y']-grad*mask2['x']
    intersect['x']=(eqn2['c']+grad_inv*center['x']-center['y'])/(grad_inv-grad)
    intersect['y']=grad_inv*intersect['x']-grad_inv*center['x']+center['y']
    
    #shortest length
    length=math.sqrt((im_w/2-intersect['x'])**2+(im_h/2-intersect['y'])**2)
    
    #print('intersect',intersect, eqn2['c'],grad,grad_inv)
    #print((eqn2['c']+grad_inv*center['x']-center['y']),grad_inv-grad)
    #print(mask['y']-mask2['y'],(mask['x']-mask2['x']))
    #print('__',mask['y'],mask2['y'],mask['x'],mask2['x'])
        
    return length, intersect

    
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = np.array(P2X - P1X)
    dY = np.array(P2Y - P1Y)
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    #print 'imageW',imageW,imageH
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    #print itbuffer.shape,img.shape
    #itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]
    line_pts=itbuffer[:,:2]
    #print line_pts.shape
    return itbuffer

def getTail(ep_pts,ep_grad,xlimit,ylimit):
        if ep_pts[0,0] < ep_pts[1,0] :
            tail='tr'
        else:
            tail='tl'

        #problematic
        #for tail at tr
        if tail=='tr':
            #print('doing tr')
            x_translate=(-ep_pts[0,1])/ep_grad
            x_end=x_translate+ep_pts[0,0]
        
            if x_end <0 or x_end>xlimit:
                y_translate=(xlimit-ep_pts[0,0])*ep_grad
                y_end=y_translate+ep_pts[0,1]
                line_pts=[[ep_pts[0,0],ep_pts[0,1]],[xlimit,int(np.floor(y_end))]]
            else:
                line_pts=[[ep_pts[0,0],ep_pts[0,1]],[int(np.floor(x_end)),0]]
            
            #for point at y=ylimit
            x_translate_low=(ylimit-ep_pts[0,1])/ep_grad
            x_end_low=x_translate_low+ep_pts[0,0]
            x_end_low=max(0,x_end_low-1)
            low_pts=[x_end_low,ylimit-1]
            
            if x_end_low<0 or x_end_low>xlimit:
                y_translate_low=(-ep_pts[0,0])*ep_grad
                y_end_low=y_translate_low+ep_pts[0,1]
                low_pts=[0,y_end_low]
            
        elif tail=='tl':
            x_translate=(-ep_pts[0,1])/ep_grad
            x_end=ep_pts[0,0]+x_translate
            if x_end <0 or x_end>xlimit:
                y_translate=(-ep_pts[0,0])*ep_grad
                y_end=ep_pts[0,1]+y_translate
                line_pts=[[ep_pts[0,0],ep_pts[0,1]],[0,int(np.floor(y_end))]]
            else:
                tmp=np.floor(x_end)
                if np.isnan(tmp):
                    tmp=0
                if tmp == np.NaN:
                    tmp=0
                line_pts=[[ep_pts[0,0],ep_pts[0,1]],[int(tmp),0]]   
                
            #for point at y=ylimit
            x_translate_low=(ylimit-ep_pts[0,1])/ep_grad
            #print('x',x_translate_low,ep_grad)
            x_end_low=x_translate_low+ep_pts[0,0]
            x_end_low=max(0,x_end_low-1)
            low_pts=[x_end_low,ylimit-1]
            
            if x_end_low<0 or x_end_low>xlimit:
                y_translate_low=(xlimit-ep_pts[0,0])*ep_grad
                y_end_low=y_translate_low+ep_pts[0,1]
                low_pts=[xlimit-1,y_end_low]
        return low_pts,line_pts

def getLinePts(box,img,xlimit,ylimit,lineColor=(0,0,255)):
        edge1=math.sqrt((box[1,0]-box[0,0])**2+(box[1,1]-box[0,1])**2)
        edge2=math.sqrt((box[2,0]-box[1,0])**2+(box[2,1]-box[1,1])**2)
        
        ep_pts=[]
        shift_amt=[(box[0,0]-box[1,0])/2,(box[0,1] - box[1,1])/2]
        new_pt1=[box[1,0]+shift_amt[0],box[1,1]+shift_amt[1]]
        new_pt2=[box[2,0]+shift_amt[0],box[2,1]+shift_amt[1]]
        
        shift_amt=[(box[2,0]-box[1,0])/2,(box[2,1] - box[1,1])/2]
        new_pt1=[box[0,0]+shift_amt[0],box[0,1]+shift_amt[1]]
        new_pt2=[box[1,0]+shift_amt[0],box[1,1]+shift_amt[1]]
        
        if edge2>=edge1:
            shift_amt=[(box[0,0]-box[1,0])/2,(box[0,1] - box[1,1])/2]
            new_pt1=[box[1,0]+shift_amt[0],box[1,1]+shift_amt[1]]
            new_pt2=[box[2,0]+shift_amt[0],box[2,1]+shift_amt[1]]        
             
            if box[1,1]<box[2,1]:
                ep_pts.append(new_pt2)
                ep_pts.append(new_pt1)
            else:
                ep_pts.append(new_pt1)
                ep_pts.append(new_pt2)
            
        if edge1>edge2:
            
            shift_amt=[(box[2,0]-box[1,0])/2,(box[2,1] - box[1,1])/2]
            new_pt1=[box[0,0]+shift_amt[0],box[0,1]+shift_amt[1]]
            new_pt2=[box[1,0]+shift_amt[0],box[1,1]+shift_amt[1]]
            if box[0,1]<box[1,1]:
                ep_pts.append(new_pt2) #point with higher y value (i.e. located lower)
                ep_pts.append(new_pt1) #point with lower y value (i.e. located higher)
            else:
                ep_pts.append(new_pt1)
                ep_pts.append(new_pt2)
        ep_pts=np.array(ep_pts)
        
        #ep_grad=(ep_pts[1,1]-ep_pts[0,1])/(ep_pts[1,0]-ep_pts[0,0])
        
        if ep_pts[1,0]==ep_pts[0,0]:
            ep_grad=(ep_pts[1,1]-ep_pts[0,1])/nearZero
        else:
            ep_grad=(ep_pts[1,1]-ep_pts[0,1])/(ep_pts[1,0]-ep_pts[0,0])
        
        # prevent ep_grad==0 from creating NaN or infinity
        if ep_grad==0:
            ep_grad=nearZero
        
        # get line tail
        low_pts,line_pts=getTail(ep_pts,ep_grad,xlimit,ylimit)
                
        line_pts=np.array(line_pts)
        line_pts= (np.floor(line_pts)).astype(int)
        
        # red line, jiayi
        cv2.line(img,(line_pts[0,0],line_pts[0,1]),(line_pts[1,0],line_pts[1,1]),lineColor,1) 
        
        # generate image with line only
        imgNewDice=np.zeros(img.shape)
        cv2.line(imgNewDice,(line_pts[0,0],line_pts[0,1]),(line_pts[1,0],line_pts[1,1]),(255,255,255),2) 
        
        return line_pts,low_pts,img,imgNewDice

def addArrow(line_pts,img,low_pts,lineColor):
        arrow_tail=(line_pts[0,1],line_pts[0,0])
        arrow_head=(int(np.floor(low_pts[1])),int(np.floor(low_pts[0])))
        #print 'low_pts',low_pts
        #print 'arrow',arrow_tail,arrow_head, type(arrow_tail),type((1,1))
        it=createLineIterator(arrow_tail,arrow_head,img)
        #print('aa',arrow_tail,arrow_head,it,not it,low_pts)
        xlimit=img.shape[0]
        ylimit=img.shape[1]
        
        #print 'problem', it[-2:]
        itIsEmpty=0
        if np.array(it.shape[0])==0:
            if (not it)==True:
                itIsEmpty=1
                it=np.array([np.array(arrow_tail), np.array(arrow_head)])
                #print('empty',len(it),it,it.shape)
        #print(it.shape,it)
        
        for i in range(len(it)):
            x=int(np.floor(it[i,0]))
            y=int(np.floor(it[i,1]))
            
            x=min(x,xlimit-1)
            y=min(y,ylimit-1)
            
            #print(x, y, type(x), i, '/',len(it))
            if i%3==0:
                img[x,y]=lineColor
        
        # ensure there are 2 distinct points to form arrow
        it[it==0]=1
        #print('it',it,it.shape)
        if itIsEmpty==1:
            arrow_pt1=(it[-2,1],it[-2,0])
            arrow_pt2=(it[-1,1],it[-1,0])
        elif it.shape[0]<=2:
            arrow_pt1=(it[-1,1],it[-1,0])
            arrow_pt2=(it[-1,1],it[-1,0]-1)
        else:
            arrow_pt1=(it[-3,1],it[-3,0])
            arrow_pt2=(it[-2,1],it[-2,0])
        arrow_pt1=(int(arrow_pt1[0]),int(arrow_pt1[1]))
        arrow_pt2=(int(arrow_pt2[0]),int(arrow_pt2[1]))
        #print('arrow',arrow_pt1,arrow_pt2)
        cv2.arrowedLine(img,arrow_pt1,arrow_pt2,(0,255,0),tipLength=6)
        return img,arrow_pt1,arrow_pt2

def postprocess(img, usBorderMask, mode='pred',maskcolor=(255,255,255),bboxmode='fill',borderWeight=2):
    # read sample image
    #img=cv2.imread('/media/mmlab/data/jiayi/data32_washeem_v6/test/p8_23_010_mask.png')
    
    # initialise visualise dict
    imVisualise=dict()
    
    # ensure image is not empty
    #assert img!='noImage'
    
    # ensure borderMask is not empty
    #assert usBorderMask != 'noborder'
    
    # remove these afterwards
    #cv2.imshow('B1_usBorder',usBorderMask)
    #cv2.imshow('A2_img',img)    
    imVisualise['B1_usBorder']=usBorderMask.copy()
    imVisualise['A2_img']=img.copy()
    
    # force blank images to be non-blank
    if np.unique(img).shape[0]==1:
        #print('empty')
        img[1,:,:]=[255,255,255]
        
    # convert to grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # get binary mask
    ret, threshed_img = cv2.threshold(gray,127, 255, cv2.THRESH_BINARY)
    _ ,contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # get max contour
    c_max = max(contours, key = cv2.contourArea)
    xlimit=img.shape[1]
    ylimit=img.shape[0]
    
    for c in [1]:#contours[:1]:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c_max)

        # draw a green rectangle to visualize the bounding rect
        # imgRect=img.copy()
        # cv2.rectangle(imgRect, (x, y), (x+w, y+h), (0, 255, 0), 1) # green box
        
        # create a white & nonWhite rectMask (rmb to change back from 1 to cv2.FILLED)
        imgRect=np.zeros((img.shape))
        imgRectNoPP=np.zeros((img.shape))
        
        if bboxmode=='fill':
            cv2.rectangle(imgRect, (x, 0), (x+w, y+h), maskcolor, cv2.FILLED)
            cv2.rectangle(imgRectNoPP, (x, y), (x+w, y+h), maskcolor, cv2.FILLED)
        elif bboxmode=='nofill':
            cv2.rectangle(imgRect, (x, 0), (x+w, y+h), maskcolor, borderWeight)
            cv2.rectangle(imgRectNoPP, (x, y), (x+w, y+h), maskcolor, borderWeight)
        
        # removed these lines afterwards
        imgVisualiseBBox=np.zeros((gray.shape))
        cv2.drawContours(imgVisualiseBBox, [c_max], -1, maskcolor,-1)
        #cv2.imshow('A3_maxContour',imgVisualiseBBox)
        imVisualise['A3_maxCnt']=imgVisualiseBBox.copy()
        
        # get the min area rect
        rect = cv2.minAreaRect(c_max)
        box = cv2.boxPoints(rect)
        box_int = np.int0(box)

        # draw a red 'nghien' rectangle
        img_blank=np.zeros(img.shape)
        cv2.drawContours(img_blank, [box_int], 0, maskcolor,-1) #red box, jiayi
        img_blank=img_blank.astype(np.uint8)
        img=img & img_blank
        
        # remove these lines afterwards
        #cv2.rectangle(imgVisualiseBBox, (x, y), (x+w, y+h), maskcolor, 1)
        cv2.drawContours(imgVisualiseBBox, [box_int], 0, maskcolor,borderWeight)
        #cv2.imshow('A4_maxContourBBox',imgVisualiseBBox)
        imVisualise['A4_maxCntBbox']=imgVisualiseBBox.copy()
        imgTmp=np.zeros(imgVisualiseBBox.shape)
        cv2.drawContours(imgTmp, [box_int], 0, maskcolor,borderWeight)
        #cv2.imshow('A5_maxContourBBox',imgTmp)
        imVisualise['A5_maxCntBbox']=imgTmp.copy()
        
        
        # get angle
        angle=-rect[2]
        
        # angle if line is straight
        if angle==0:
            angle=90
            
        # adjust line colors
        if mode=='gt':
            upperLineColor=(0,255,0)
            lowerLineColor=(0,255,0)
        elif mode=='pred':
            upperLineColor=(0,0,255)
            lowerLineColor=(0,0,255)
        
        # get Line Points
        line_pts,low_pts,img,imgNewDice=getLinePts(box,img,xlimit,ylimit,upperLineColor);
        line_mid=np.array([(line_pts[0,0]+line_pts[1,0])//2,(line_pts[0,1]+line_pts[1,1])//2])
        
        # remove these lines afterwards
        imgVisualiseBBox=imgNewDice.copy()
        #cv2.rectangle(imgVisualiseBBox, (x, y), (x+w, y+h), maskcolor, 1)
        cv2.drawContours(imgVisualiseBBox, [box_int], 0, maskcolor,borderWeight)
        #cv2.imshow('A6_linearisedBBox',imgVisualiseBBox)
        imVisualise['A6_linearBbox']=imgVisualiseBBox.copy()
        #cv2.imshow('A7_linearised',imgNewDice)
        imVisualise['A7_linear']=imgNewDice.copy()
        
        
        # addArrow
        img,arrow_pt1,arrow_pt2=addArrow(line_pts,img.copy(),low_pts,lowerLineColor)
                
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0,0,255)) #red box, jiayi
        
        # get shortest length
        slen, intersectSlen=shortest_length(line_pts[1,0],line_pts[1,1],line_mid[0],line_mid[1],xlimit,ylimit)
        
        # draw line for shortest length
        cv2.line(img,(int(xlimit/2),int(ylimit/2)),(int(intersectSlen['x']),int(intersectSlen['y'])),(255,0,0),1)
        
        # needle tail at top left
        if line_pts[1,0]<=line_pts[0,0] and line_pts[1,1]<=line_pts[0,1] : 
            cv2.putText(img,'{}:{:.1f}deg {:.1f}pixels'.format(mode,angle,slen),(xlimit-70,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
        # needle tail at top right
        elif line_pts[1,0]>=line_pts[0,0] and line_pts[1,1]<=line_pts[0,1]: 
            angle=180-angle
            cv2.putText(img,'{}:{:.1f}deg {:.1f}pixels'.format(mode,angle,slen),(30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
        # needle tail at bottom right
        elif line_pts[1,0]>=line_pts[0,0] and line_pts[1,1]>=line_pts[0,1]: 
            cv2.putText(img,'{}:{:.1f}deg {:.1f}pixels'.format(mode,angle,slen),(30,ylimit-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
        # needle tail at bottom left
        elif line_pts[1,0]<=line_pts[0,0] and line_pts[1,1]>=line_pts[0,1]: 
            angle=180-angle
            cv2.putText(img,'{}:{:.1f}deg {:.1f}pixels'.format(mode,angle,slen),(xlimit-30,ylimit-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
                
        # save img
        imgMaxContour=img.copy()
        
        # get tip distance
        tipDistance=getTipDistance(line_pts)

        # remove pixels outside ultrasound scan borders
        imgNewDice=imgNewDice.astype(np.uint8)
        #cv2.imshow('A7_predExtended',imgNewDice)
        #imVisualise['A7_predExtend']=imgNewDice.copy()
        imgNewDice=imgNewDice & usBorderMask
        
        #cv2.imshow('A8_predExtendedConstrained',imgNewDice)
        imVisualise['A8_predExtendContrained']=imgNewDice.copy()
        imgNewDice=cv2.cvtColor(imgNewDice, cv2.COLOR_BGR2GRAY)
        _ , contoursNewDice, hier \
        = cv2.findContours(imgNewDice \
                           , cv2.RETR_TREE \
                           , cv2.CHAIN_APPROX_SIMPLE)
        if contoursNewDice != []:
            c_max = max(contoursNewDice, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_max)
            imgNewRect2=np.zeros(imgNewDice.shape)
            if bboxmode=='fill':
                cv2.rectangle(imgNewRect2, (x, y), (x+w, y+h), maskcolor, -1)
            elif bboxmode=='nofill':
                cv2.rectangle(imgNewRect2, (x, y), (x+w, y+h), maskcolor, borderWeight)
            
            # remove these lines afterwards
            imgNewRect3=imgNewDice.copy()
            cv2.rectangle(imgNewRect3, (x, y), (x+w, y+h), maskcolor, borderWeight)
            imgNewRect3=imgNewRect3.astype(np.uint8)
            #cv2.imshow('A9_predExtendedConstrainedRect',imgNewRect3)
            imVisualise['A9_predExtendContrainedRect']=imgNewRect3.copy()
            cv2.rectangle(imgNewRect3, (x, y), (x+w, y+h), maskcolor, -1)
            #cv2.imshow('A10_predExtendedConstrainedRectFilled',imgNewRect3)
            imVisualise['A10_predExtendContrainedRect']=imgNewRect3.copy()
            
                        
            imgNewRect2=imgNewRect2.astype(np.uint8)
            imgNewRect2=cv2.cvtColor(imgNewRect2, cv2.COLOR_GRAY2BGR)
            imgNewDice=cv2.cvtColor(imgNewDice, cv2.COLOR_GRAY2BGR)
        else:
            imgNewRect2=np.zeros(img.shape)
            imgNewDice=cv2.cvtColor(imgNewDice, cv2.COLOR_GRAY2BGR)    

        # visualise image        
        debugImage=False
        if debugImage:
            #imgRect=np.concatenate((imgRect,imgRect),axis=1)
            cv2.imshow('rect',imgRect)
            cv2.imshow('rectNoPP',imgRectNoPP)
            cv2.imshow('img',gray)
            cv2.imshow('noPP',imgNewDice)
            cv2.imshow('noPP2',imgNewDice2)
            #cv2.imshow('imgMaxContour',imgMaxContour)
            
            if cv2.waitKey(0) == ord('q'):
                raise KeyboardInterrupt
        
    cv2.drawContours(img, contours, -1, (255, 255, 0), borderWeight)
    
    debugImage=False
    if debugImage==True:
        if not (grayAfter==gray).all():
            cv2.imshow('before',gray)
            cv2.imshow('after',grayAfter)
            cv2.imshow('noPP',imgNewDice)
            cv2.imshow('rectNoPP',imgRectNoPP)
            
            if ord('q') == cv2.waitKey(0):
                raise KeyboardInterrupt
    

#    return angle,slen,imgMaxContour,tipDistance,imgNewDice,imgRect,imgRectNoPP
    return angle,slen,imgMaxContour,tipDistance,imgNewDice,imgNewRect2,imgRectNoPP,imVisualise
    
def main(pred_path="pred/*_mask.png",gt_dir='/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_reduced2/'):


    if not os.path.exists(store_dir_pred):
        #os.mkdir(store_dir_pred)
        print("created", store_dir_pred)
        
    if not os.path.exists(store_dir_gt):
        #os.mkdir(store_dir_gt)
        print("created", store_dir_gt)


    #img_all=glob.glob("/media/leejiayi/DATA/ubuntu/pytorch/20181010_preds_pytorch_linknet_dataset_v5_retest2/linknet_preds_all/*_mask.png")
    #img_all=glob.glob("/media/leejiayi/DATA/ubuntu/pytorch/linknet_ssp/preds_linknet_4nov_spp14e/*_mask.png")
    #img_all=glob.glob("/media/leejiayi/DATA/ubuntu/pytorch/linknet_ssp/preds_densenet_reduced_test/*_mask.png")

    #img_all=glob.glob("/media/leejiayi/DATA/ubuntu/pytorch/linknet_ssp/preds_densenet_v2/*_mask.png")
    #gt_dir='/media/leejiayi/DATA/ubuntu/data27_test_v5_final/test_v5_raw/'

    img_all=glob.glob(pred_path)
    

    #/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_reduced2/**_mask.png

    for i in range(len(img_all)): #original: len(img_all)
        
        img=cv2.imread(img_all[i])
        if len(np.unique(img)) ==1:
            #print(img_all[i])
            error.append(90**2)
            
            continue
        
        angles_pred.append(postprocess(img_all[i],'pred'))
        gt_path=os.path.split(img_all[i])[-1]
        gt_path=os.path.join(gt_dir,gt_path)
        #print(gt_path)
        angles_gt.append(postprocess(gt_path,'gt'))
        #print(i)
        if i%20==0:
            pass
            #print('done',i)

    for i in range(len(angles_pred)):
        error.append((angles_pred[i]-angles_gt[i])**2)
            
    #print(error[19], angles_pred[19],angles_gt[19])
    print('-'*30) 
    print("rms error (deg)",math.sqrt(np.mean(error)))

if __name__=='__main__':
    #main()
    postprocess()
    
    
