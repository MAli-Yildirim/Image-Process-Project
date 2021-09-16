#MehmetAli
#Yıldırım
#160403008
#Updated at 25.12.2020
import numpy as np
from PIL import Image
from math import floor,ceil,acos,pi,sqrt,cos,atan,degrees,log1p
import random
from scipy.signal.signaltools import wiener


def ReSizePixel(row,coulum,image,QProgressBar):
    #ConvertToArray
    arr = np.array(image)       
    
    #--Variables--
    summ = 0
    realrow    = len(arr)
    realcoulum   = len(arr[0])
    rowRatio   = floor(realrow/row)
    coulumRatio  = floor(realcoulum/coulum)        
    MissRow = realrow - rowRatio*row
    MissCoulum = realcoulum - coulumRatio*coulum    
    
    #--AllPixelCount
    pieces = (coulumRatio*rowRatio)+(coulumRatio+rowRatio)+1
    progress = 0
    #-----------RecreatedPixels--------
    
    #-------------GetDimension-----------------
    try:
        length = len(arr[0][0])
    except:
        length = 0
        
        
    #--StartReconstruct--
    #--Startloopfor-eachNewPixelChunk---
    for h in range(coulumRatio):
        for k in range(rowRatio):
    #-----------------------------------
    #---------ScanEachPixelChunk--------
            for l in range(length):
                for i in range(row):
                    for j in range(coulum):
                        #FindSummation
                        summ += arr[i+row*k][j+coulum*h][l]    
                for i in range(row):
                    for j in range(coulum):
                        #ReplaceValuesToAverage
                        arr[i+row*k][j+coulum*h][l] = floor(summ/(row*coulum))
                summ = 0
    #--------ProgressTracking-----------
            progress += 1
            progressvalue = (100.0/pieces)*progress
            QProgressBar.setValue(progressvalue)
                    
    #------ForAnyLeftOverPixels-----------
    
        
    #--Startloopfor-eachMissingPixelChunkOnRow---
    for h in range(coulumRatio):   
    #-----------------------------------
    #---------ScanEachPixelChunk--------
        for l in range(length):
            for i in range(MissRow):
                for j in range(coulum):
                #FindSummation
                    summ += arr[i+row*rowRatio][j+coulum*h][l]    
            for i in range(MissRow):
                for j in range(coulum):
                #ReplaceValuesToAverage
                    arr[i+row*rowRatio][j+coulum*h][l] = floor(summ/(MissRow*coulum))
            summ = 0
    #--------ProgressTracking-----------
        progress += 1
        progressvalue = (100.0/pieces)*progress
        QProgressBar.setValue(progressvalue)
            
        
    #--Startloopfor-eachMissingPixelChunkOnCoulum---
    for h in range(rowRatio):   
    #-----------------------------------
    #---------ScanEachPixelChunk--------
        for l in range(length):
            for i in range(row):
                for j in range(MissCoulum):
                    #FindSummation
                    summ += arr[i+h*row][j+coulum*coulumRatio][l]    
            for i in range(row):
                for j in range(MissCoulum):
                    #ReplaceValuesToAverage
                    arr[i+row*h][j+coulum*coulumRatio][l] = floor(summ/(MissCoulum*row))
            summ = 0
    #--------ProgressTracking-----------
        progress += 1
        progressvalue = (100.0/pieces)*progress
        QProgressBar.setValue(progressvalue)
    #--Startloopfor-LastMissingPixelChunk---
    #-----------------------------------
    #---------ScanEachPixelChunk--------
    for l in range(length):
        for i in range(MissRow):
            for j in range(MissCoulum):
                            #FindSummation
                summ += arr[i+rowRatio*row][j+coulum*coulumRatio][l]    
        for i in range(MissRow):
            for j in range(MissCoulum):
                            #ReplaceValuesToAverage
                arr[i+row*rowRatio][j+coulum*coulumRatio][l] = floor(summ/(MissCoulum*MissRow))
        summ = 0
    #--------ProgressTracking-----------
    progress += 1
    progressvalue = (100.0/pieces)*progress
    QProgressBar.setValue(progressvalue)
            
            #------ConversionOfArraysIntoImageobjects-------
    image = Image.fromarray(np.uint8(arr))       
    return image

def ReSizeScale(image,scale):   
    # 3-case sceneario Scale lessThanOne-EqualToOne-HigherThanOne
    #First Equal to One
    
    if scale == 1:
        
        return image
    
    #-----------Variables--------------
    #ConvertToArray
    arr = np.array(image)
    #GetWidthAndHeigthInfo
    realRow    = len(arr)
    realCoulum = len(arr[0])
    #NewImageSizes
    if scale < 1:
        newRow    = ceil(scale*realRow)
        newCoulum = ceil(scale*realCoulum)
        samplerate = floor(1.0/scale)
        
    if scale > 1:
        newRow    = ceil(scale)*realRow
        newCoulum = ceil(scale)*realCoulum
        
        
    
    #MakeResultArr
    newArr = np.zeros((newRow,newCoulum,3))
    
    
    #Secondly Less Than one
    if scale < 1:
       
        #PlaceAllTheNewValuesToResultArray
        for l in range(3):
            for i in range(newRow):
                for j in range(newCoulum):
                    #SamplerateMustBeLimitedImageWouldBeCropped
                    if 10%samplerate == 0:
                        newArr[i][j][l] = arr[i*samplerate][j*samplerate][l]
                    else:
                        return image
               
          
            
                    
        #------ConversionOfArraysIntoImageobjects-------
        image = Image.fromarray(np.uint8(newArr))  
        return image               
        
    #At last HigherThanOne
    if scale > 1:
        scaletemp = ceil(scale)
        
        #PlaceAllTheNewValuesToResultArray
        for l in range(3):
            for i in range(realRow):
                for j in range(realCoulum):
                    for k in range(scaletemp):
                        for x in range(scaletemp):
                            #ScaledUpSquarelyForEachPixelMustBeLimitedToIntValues
                                newArr[scaletemp*i+k][scaletemp*j+x][l] = arr[i][j][l]
        
        #ChangeForm
        image = Image.fromarray(np.uint8(newArr))
        #SendResult
        return image
        
   
def Reflection(image,Head):
    
    #-----------Variables--------------
    #ConvertToArray
    arr = np.array(image)
    row    = len(arr)
    coulum   = len(arr[0])
    newarr = np.zeros((row,coulum,3))
    
    #If Its Horizontal Reflection StartReadingBackwardFromRow
    if Head == "Horizontal":
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[row-i-1][j][l]
    #If Its Vertical Reflection StartReadingBackwardFromCoulum
    if Head == "Vertical":
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[i][coulum-j-1][l]
        
        
        
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #SendResult
    return image
        
        
def Shift(image,Srow,Scoulum):
    #-----------Variables--------------
    #ConvertToArray
    arr = np.array(image)
    row    = len(arr)-abs(Srow)
    coulum   = len(arr[0])-abs(Scoulum)
    newarr = np.zeros((row,coulum,3))
    #DecisionForShiftingImage
    #FourCase
    #Bottom-Left-Shifting
    if Srow <= 0 and Scoulum <= 0:
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[i][j-Scoulum][l]
                    
    #Bottom-Right-Shifting
    if Srow <= 0 and Scoulum >= 0:
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[i][j][l]
                   
                    
    #Up-Left-Shifting
    if Srow >= 0 and Scoulum <= 0:
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[i+Srow][j-Scoulum][l]
                    
                    
    #Up-Right-Shifting
    if Srow >= 0 and Scoulum >= 0:
        for l in range(3):
            for i in range(row):
                for j in range(coulum):
                    newarr[i][j][l] = arr[i+Srow][j][l]
    

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #SendResult
    return image   

    
def RGBtoHSI(image):
    #ConvertToArray
    arr = np.array(image)
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum,3))
    
    for i in range(row):
        for j in range(coulum):
            #TakeColurValuesFromPixel
            R = arr[i][j][0]/255
            G = arr[i][j][1]/255
            B = arr[i][j][2]/255
            #Find H,S and I
            #Find H
            numerator = ((1/2)*((R-G)+(R-B)))
            denumerator = sqrt(pow((R-G),2)+((R-B)*(G-B)))
            H = acos(numerator/(denumerator+0.0000000001))
            
            
            #If B > G than change Degree
            if B>G:
                H = (2*pi)-H
                
            #FixDecimal
            H = H/(2*pi)
            #Find S
            S = 1-(3/(R+G+B+0.00001))*min(R,G,B)
            
                
            #Find I
            I = (1/3)*(R+G+B)
            #AddPixelIntoNewArray

            newarr[i][j][0] = H*255
            newarr[i][j][1] = S*255
            newarr[i][j][2] = I*255

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    
    #SendResult
    return image

def HSItoRGB(image):
    #ConvertToArray
    arr = np.array(image)
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum,3))
    
    for i in range(row):
        for j in range(coulum):
            
            #TakingValues
            H = (arr[i][j][0]/255)*2*pi
            
            S = (arr[i][j][1]/255)
            I = (arr[i][j][2]/255)
            
            
            #ThreeCases
            #1-0<H<120 Region
            if 0 <= H and H < 2*pi/3:
                #Conversion
                r = (1/3)*(1+(S*cos(H))/((cos((pi/3)-H))+0.02))
                b = (1-S)/3
                g = 1-b-r
                #ConversionAndSetting
                newarr[i][j][0] = 3*I*r*255
                newarr[i][j][1] = 3*I*g*255
                newarr[i][j][2] = 3*I*b*255
            elif 2*pi/3 <= H and H < 4*pi/3:
                #Conversion
                H = H - 2*pi/3 
                g = (1/3)*(1+(S*cos(H))/((cos((pi/3)-H))+0.02))
                r = (1-S)/3
                b = 1-g-r
                #ConversionAndSetting
                newarr[i][j][0] = 3*I*r*255
                newarr[i][j][1] = 3*I*g*255
                newarr[i][j][2] = 3*I*b*255
            elif 4*pi/3 <= H and H <= 2*pi:
                #Conversion
                H = H - 4*pi/3
                b = (1/3)*(1+(S*cos(H))/((cos((pi/3)-H))+0.02))              
                g = (1-S)/3
                r = 1-g-b
                #ConversionAndSetting
                newarr[i][j][0] = 3*I*r*255
                newarr[i][j][1] = 3*I*g*255
                newarr[i][j][2] = 3*I*b*255
                
            

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #SendResult
    return image 
    
    
    
    
def HistStr(image):
    
    
    arr = np.array(image)
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
    #ConvertingToLimunanceImage
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = 0.299*arr[i][j][0]+0.587*arr[i][j][1]+0.114*arr[i][j][2]
    
    #FindMax&Min
    minPixel = newarr.min()
    maxPixel = newarr.max()
    #HistStrechFunc
    for i in range(row):
        for j in range(coulum):
            newarr[i][j]=255*((newarr[i][j]-minPixel)/(maxPixel-minPixel))
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image
    
    
def HistEqu(image):
    
    arr = np.array(image)
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
    size     = row*coulum
    func      = np.zeros(256)
    #ConvertingToLimunanceImage
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = 0.299*arr[i][j][0]+0.587*arr[i][j][1]+0.114*arr[i][j][2]
    
    newarr = np.uint8(newarr)
    
    
    #PdfCalculate
    for i in range(row):
        for j in range(coulum):
            func[newarr[i][j]] += 1
    #CdfCalculate
    for x in range(255):
        func[x+1] = func[x]+func[x+1]
    func =  np.round(255*np.divide(func,size))
    #ConstructTheNewImage
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = func[newarr[i][j]]
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))

    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    
    
    
def Functf(image,lower,upper):
    
    arr = np.array(image)
    
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum,3))
    
       
    
    for k in range(3):
        for i in range(row):
            for j in range(coulum):
                newarr[i][j][k] = (arr[i][j][k])*(upper-lower)/255+lower
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))

    #ConvertBackToRGB
    #image = image.convert('RGB')
    return image    

def GammaFt(image,gamma):
    
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
    #ApplyTf
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = 255*pow((arr[i][j]/255),gamma)           
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image
        


            
            
def FilterImage(image,ft):
    
    arr = np.array(image)
    
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
    newarr2  = np.zeros((row,coulum))
    
    #ConvertingToLimunanceImage
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = 0.299*arr[i][j][0]+0.587*arr[i][j][1]+0.114*arr[i][j][2]
    #-------------MinFilter---------------------------------
    if ft == 0:
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(newarr[i+l][j+k])
                newarr2[l+1][k+1] = min(samplearray)
    #-------------------------------------------------------
    #-------------MaxFilter---------------------------------
    if ft == 1:
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(newarr[i+l][j+k])
                newarr2[l+1][k+1] = max(samplearray)
    #-------------------------------------------------------
    #-------------AverageFilter---------------------------------
    if ft == 2:
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(newarr[i+l][j+k])
                newarr2[l+1][k+1] = (int)(sum(samplearray)/9)
    #-------------------------------------------------------
    #-------------MedianFilter---------------------------------
    if ft == 3:
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(newarr[i+l][j+k])
                newarr2[l+1][k+1] = (int)(np.median(samplearray))
    #-------------------------------------------------------
    #-------------LaplacianScaledFilter---------------------------------
    if ft == 4:
        
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
    #MaskFilter
        mask = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                newarr2[l+1][k+1] = sum(samplearray)
        #ScaleTheLaplacianInto0-255
        minn = newarr2.min()
        maxx = newarr2.max()
        for i in range(row-2):
            for j in range(coulum-2):
                newarr2[i+1][j+1] = (newarr2[i+1][j+1] - minn)*(255/(maxx-minn))
     #-------------Median-LaplacianScaledFilter---------------------------------
    if ft == 5:
        
        image = FilterImage(image,3)
        arr = np.array(image)
        #ConvertingToLimunanceImage
        for i in range(row):
            for j in range(coulum):
                newarr[i][j] = 0.299*arr[i][j][0]+0.587*arr[i][j][1]+0.114*arr[i][j][2]
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
    #MaskFilter
        mask = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                newarr2[l+1][k+1] = sum(samplearray)
        #ScaleTheLaplacianInto0-255
        minn = newarr2.min()
        maxx = newarr2.max()
        for i in range(row-2):
            for j in range(coulum-2):
                newarr2[i+1][j+1] = (newarr2[i+1][j+1] - minn)**(255/(maxx-minn))
    
    #-------------AbsLaplacianFilter---------------------------------
    if ft == 6:
        
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
    #MaskFilter
        mask = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                newarr2[l+1][k+1] = abs(sum(samplearray))
    #-------------SharpFilter---------------------------------
    if ft == 7:
        temparr = np.zeros((row,coulum))
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
    #MaskFilter
        mask = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        #FindLaplacianOfOrginalImage
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                newarr2[l+1][k+1] = abs(sum(samplearray))
        #ScaleTheLaplacianInto0-255
        minn = newarr2.min()
        maxx = newarr2.max()
        for i in range(row-2):
            for j in range(coulum-2):
                newarr2[i+1][j+1] = (newarr2[i+1][j+1] - minn)*(255/(maxx-minn))
        #SharpenedImage
        temparr = np.add(newarr,newarr2)
        #RescaleSharpenedImage
        minn = temparr.min()
        maxx = temparr.max()
        for i in range(row-2):
            for j in range(coulum-2):
                temparr[i+1][j+1] = (temparr[i+1][j+1] - minn)*(255/(maxx-minn))
        newarr2 = temparr
    #-------------SobelGradiant---------------------------------
    if ft == 8:
        Gx = np.zeros((row,coulum))
        Gy = np.zeros((row,coulum))
        G  = np.zeros((row,coulum))
    #TakeSamplesFrom3X3MatrixWhichIsShiftingOnTheOrginalImage
    #MaskFilter
        mask = [[-1,0,+1],[-2,0,2],[-1,0,1]]
        #FindVertical
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                Gx[l+1][k+1] = abs(sum(samplearray))
        #MaskFilter      
        mask = [[1,2,1],[0,0,0],[-1,-2,-1]]
        #FindHorizontal
        for l in range(row-2):
            for k in range(coulum-2):
                samplearray = []
                for i in range(3):
                    for j in range(3):
                        samplearray.append(mask[i][j]*newarr[i+l][j+k])
                Gy[l+1][k+1] = abs(sum(samplearray))
        #CombineTwo
        for i in range(row):
            for j in range(coulum):
                G[i][j] = sqrt(pow(Gx[i][j],2)+pow(Gy[i][j],2))
        #ScaleTheLastImage
        minn = G.min()
        maxx = G.max()
        for i in range(row):
            for j in range(coulum):
                G[i][j] = (G[i][j] - minn)*(255/(maxx-minn))
        newarr2 = G
    #-------------MaskImage---------------------------------
    if ft == 9:
        mImage = np.zeros((row,coulum))
        #Take Neccessary Image Tranfromations
        #ConvertingToLimunanceImageArrays
        SharpImage   = np.array(FilterImage(image,7).convert("L"))
        
        SobelImage   = FilterImage(image,8)
        SobelAvImage = np.array(FilterImage(SobelImage, 2).convert("L"))
        #MaskImage
        for i in range(row):
            for j in range(coulum):
                mImage[i][j] = sqrt(pow(SobelAvImage[i][j],2)+pow(SharpImage[i][j],2))
        #ScaleTheLastImage
        minn = mImage.min()
        maxx = mImage.max()
        for i in range(row):
            for j in range(coulum):
                mImage[i][j] = (mImage[i][j] - minn)*(255/(maxx-minn))
        newarr2 = mImage
    #-------------MaskImageSharp---------------------------------
    if ft == 10:
    
        MImage = np.array(FilterImage(image,9).convert("L"))
        newarr2 = np.add(MImage,newarr)
        #ScaleTheLastImage
        minn = newarr2.min()
        maxx = newarr2.max()
        for i in range(row):
            for j in range(coulum):
                newarr2[i][j] = (newarr2[i][j] - minn)*(255/(maxx-minn))
        
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr2))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image
            
            
def LowPass(image,value,n):
    
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    
    Parr     = np.zeros((2*row,2*coulum))
    Filter   = np.zeros((2*row,2*coulum))
    #It takes forever to Fourier Transform it so I am using numpy library
    """
    for x in range(row):
        for y in range(coulum):
            a = 0
            for i in range(row):
                for j in range(coulum):
                    a += arr[i][j]*np.exp(-2j*np.pi*(((x*i)/row)+((y*j)/coulum)))
    """
    #oConstructFilterButterWorth
    
    for i in range(2*row):
        for j in range(2*coulum):
            #FilterDefinition
            D = pow( pow( row-i , 2 ) + pow( coulum-j , 2 ) , 0.5 )
            #Filter[i][j] = np.exp((-1*pow(D[i][j],2))/(2*value*value))
            Filter[i][j] = 1/((1 +  pow( D/value , 2*n )  ))
            
                

    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
    for i in range(2*row):
        for j in range(2*coulum):
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
    
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    
    #ApplyFilter
    Parr = np.multiply(Parr,Filter)
    #InverseFourierTransform
    Parr = np.fft.ifft2(Parr)
    
    #CenterTransform
    for i in range(row):
        for j in range(coulum):
            arr[i][j] = pow(-1,i+j)*Parr[i][j].real
    
    
    
    
    
    
    
    #ChangeForm
    image = Image.fromarray(np.uint8(arr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image

def HighPass(image,value,n):
    
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
    Parr     = np.zeros((2*row,2*coulum))
    Filter   = np.zeros((2*row,2*coulum))
    #It takes forever to Fourier Transform it so I am using numpy library
    """
    for x in range(row):
        for y in range(coulum):
            a = 0
            for i in range(row):
                for j in range(coulum):
                    a += arr[i][j]*np.exp(-2j*np.pi*(((x*i)/row)+((y*j)/coulum)))
    """
    #ConstructFilter
    for i in range(2*row):
        for j in range(2*coulum):
            #FilterDefinition
            D = sqrt(pow(i-row,2)+pow(j-coulum,2))
            #Filter[i][j] = (1 - np.exp((-1*pow(D[i][j],2))/(2*value*value)))
            try:
                Filter[i][j] = 1/(1+pow((value/D),2*n))
            except:
                Filter[i][j] = 1
                
            
    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    #ApplyFilter
    Parr = np.multiply(Parr,Filter)
    #InverseFourierTransform
    Parr = np.fft.ifft2(Parr)
    #CenterTransform
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = abs(pow(-1,i+j)*Parr[i][j].real)
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    
def Threshold(image,value):
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    newarr   = np.zeros((row,coulum))
        
    for i in range(row):
        for j in range(coulum):
            if arr[i][j] > value:
                newarr[i][j] = 255
            else:
                newarr[i][j] = 0
       
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    

def CSpectrum(image):
    
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    Parr     = np.zeros((2*row,2*coulum))
    a     = np.zeros((2*row,2*coulum))
    #It takes forever to Fourier Transform it so I am using numpy library
    """
    for x in range(row):
        for y in range(coulum):
            a = 0
            for i in range(row):
                for j in range(coulum):
                    a += arr[i][j]*np.exp(-2j*np.pi*(((x*i)/row)+((y*j)/coulum)))
    """
    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
    
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    #ApplyFilter
    for i in range(2*row):
        for j in range(2*coulum):
            a[i][j] = sqrt(pow(Parr[i][j].real,2)+pow(Parr[i][j].imag,2))
            
    #LogarithmicScale
    maxx = a.max()
    c = 255/log1p(maxx)
    for i in range(2*row):
        for j in range(2*coulum):
            #a[i][j] = (a[i][j] - minn)*(255/(maxx-minn))
            a[i][j] = c*log1p(a[i][j])
            

    #ChangeForm
    image = Image.fromarray(np.uint8(a))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image         
            
            
def Spectrum(image):
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    Parr     = np.zeros((2*row,2*coulum))
    a     = np.zeros((row,coulum))
    #It takes forever to Fourier Transform it so I am using numpy library
    """
    for x in range(row):
        for y in range(coulum):
            a = 0
            for i in range(row):
                for j in range(coulum):
                    a += arr[i][j]*np.exp(-2j*np.pi*(((x*i)/row)+((y*j)/coulum)))
    """
    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
    
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    #ApplyFilter
    for i in range(row):
        for j in range(coulum):
            a[i][j] = sqrt(pow(Parr[i][j].real,2)+pow(Parr[i][j].imag,2))
            
    #LogarithmicScale
    maxx = a.max()
    c = 255/log1p(maxx)
    for i in range(row):
        for j in range(coulum):
            #a[i][j] = (a[i][j] - minn)*(255/(maxx-minn))
            a[i][j] = c*log1p(a[i][j])
    #ChangeForm
    image = Image.fromarray(np.uint8(a))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image         

def BReject(image,value,width,order):
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    Parr     = np.zeros((2*row,2*coulum))
    newarr   = np.zeros((row,coulum))
    
    Filter   = np.ones((2*row,2*coulum))
    #ConstructFilter
    for i in range(2*row):
        for j in range(2*coulum):
            
            D = pow(pow(i - row, 2) + pow(j - coulum, 2), 0.5)
            try:
                Filter[i][j] = 1/( 1 + pow( (width / (D*D-value*value) ) , 2*order ) )
            except:
                Filter[i][j] = 1
            
            #Filter[i][j] *= (1.0 / (1 + pow((value * value) / (1+d1 * d2), 4))) 
                
                
                
                
    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    #ApplyFilter
    Parr = np.multiply(Parr,Filter)
    #InverseFourierTransform
    Parr = np.fft.ifft2(Parr)
    #CenterTransform
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = abs(pow(-1,i+j)*Parr[i][j].real)
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    



def SPnoise(image,freq):
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    
    for i in range(row):
        for j in range(coulum):
            value = random.random()
           
            #between 0 and z   //Salt
            if value <= freq/200:
                
                arr[i][j] = 255
                
            #between z and 2*z  //Pepper  
            elif value > freq/200 and freq/100 >= value:
                arr[i][j] = 0
                
            #between 2*z and 1  // Orginal Image  
            elif value > freq/100:
                arr[i][j] = arr[i][j]

    #ChangeForm
    image = Image.fromarray(np.uint8(arr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    

def Gnoise(image,mean,variance):
    arr = np.array(image.convert('L'))
    #Variables
    row      = len(arr)
    coulum   = len(arr[0])
    noise   = np.zeros((row,coulum))
    #Create Noise
    noise = np.random.normal(float(mean),variance,(row,coulum))
   
    #Add Noise
    last = np.add(noise,arr)
    
    #ScaleTheLastImage
    minn = last.min()
    maxx = last.max()
    for i in range(row):
        for j in range(coulum):
            last[i][j] = (last[i][j] - minn)*(255/(maxx-minn))
    
    #ChangeForm
    image = Image.fromarray(np.uint8(last))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image 




def Eff(im1,im2):
    arr1 = np.array(im1.convert('L'))
    arr2 = np.array(im2.convert('L'))
    #Variables
    row      = len(arr1)
    coulum   = len(arr1[0])
    summ = 0
    N = row*coulum
    #CheckImageSizes
    if len(arr1) != len(arr2):
        return "Must be same size"
    #Mean Square Error
    for i in range(row):
        for j in range(coulum):
            summ += pow(arr1[i][j]-arr2[i][j] , 2)
            
    return str(int((100/255)*sqrt(summ/N)))
    
    
    
def winF(image):
    arr = np.array(image.convert('L'))
    
    row      = len(arr)
    coulum   = len(arr[0])
    newarr = np.zeros((row,coulum))
    
  
    H = np.zeros((2*row,2*coulum),dtype=np.complex_)
 
    Parr     = np.zeros((2*row,2*coulum))
    #Constrcut Filter
    for i in range(2*row):
        for k in range(2*coulum):
            try:
                H[i][k] = (1 /( (np.pi) * ( 0.05*(row-i+coulum-k-2) ) ) ) * np.sin(np.pi*0.05*(-2+row-i+coulum-k))*np.exp(-1j*np.pi*(0.05*(-2+row-i+coulum-k)))
            except:
                H[i][k] = 1

    #ConstructPaddedImage
    for i in range(row):
        for j in range(coulum):
            Parr[i][j] = arr[i][j]
            #TransformCenter
            Parr[i][j] = pow(-1,i+j)*Parr[i][j]
            
    #FastFourierTransform
    Parr = np.fft.fft2(Parr)
    #ApplyFilter
    Parr = Parr*np.conj(H) / (np.abs(H) ** 2 + 0.01)
    #InverseFourierTransform
    Parr = np.fft.ifft2(Parr)
    #CenterTransform
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = abs(pow(-1,i+j)*Parr[i][j].real)
    
    #ScaleTheLastImage
    minn = newarr.min()
    maxx = newarr.max()
    for i in range(row):
        for j in range(coulum):
            newarr[i][j] = (newarr[i][j] - minn)*(255/(maxx-minn))
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image    
    
def erosion(image,size = 5):
    arr = np.array(image.convert('L'))
    
    row      = len(arr)
    coulum   = len(arr[0])
    newarr = np.zeros((row,coulum))
    
            
    #-------------MinFilter---------------------------------
    #TakeSamplesFrom5X5MatrixWhichIsShiftingOnTheOrginalImage
    for l in range(row-size-1):
        for k in range(coulum-size-1):
            samplearray = []
            for i in range(size):
                for j in range(size):
                    samplearray.append(arr[i+l][j+k])
            newarr[l+2][k+2] = min(samplearray)
            

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image

def dilation(image,size = 5):
    arr = np.array(image.convert('L'))
    
    row      = len(arr)
    coulum   = len(arr[0])
    newarr = np.zeros((row,coulum))
    
            
    #-------------MinFilter---------------------------------
    #TakeSamplesFrom5X5MatrixWhichIsShiftingOnTheOrginalImage
    for l in range(row-size-1):
        for k in range(row-size-1):
            samplearray = []
            for i in range(size):
                for j in range(size):
                    samplearray.append(arr[i+l][j+k])
            newarr[l+2][k+2] = max(samplearray)
            

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    return image
    
def morpGrad(image):    
    #Morp Grad 
    newarr = np.subtract(np.array(dilation(image).convert('L')) , np.array(erosion(image).convert('L'))) 
    

    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    
    return image
    
def topHat(image):    
    #Top hat Transformation
    newarr =  np.subtract(np.array(image.convert('L')),np.array(erosion(dilation(image,5),5).convert('L')))
    
    #ChangeForm
    image = Image.fromarray(np.uint8(newarr))
    #ConvertBackToRGB
    image = image.convert('RGB')
    
    return image
    
    
    
    
    
    
            
        

            
            
            