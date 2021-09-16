#MehmetAli
#Yıldırım
#160403008
#ImageProcessHomework
#Updated at 25.12.2020


#-------Libraries-----------
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from PyQt5 import QtGui
import numpy as np
from PIL import Image,ImageQt

import ImageProcessLib

    

#----------InterfaceClass-------
class ImageProcessInterface(QMainWindow):
    #------InitilizationOnStart-----------
    def __init__(self):
        QMainWindow.__init__(self)
        QtGui.QImageReader.supportedImageFormats()
        
        #-------Variables----------------
        self.OrginalImage = []
        self.EnableCrop   = 0
        self.OngoingImage = Image.Image
        
        #-------Load-Interface------------
        loadUi("ImageProcessDesign.ui",self)
        
        #-------Set-Title-----------------
        self.setWindowTitle("ImageProcessingHomeworks#")
        
        #-------Call-Backs----------------
        self.pushButton_2.clicked.connect(self.LoadImage)     #Load Image
        self.pushButton.clicked.connect(lambda: self.Show(2,self.Re_sizePixels()))
        self.pushButton_3.clicked.connect(self.SaveImage)
        self.pushButton_4.clicked.connect(self.Re_sizeScale)
        self.pushButton_7.clicked.connect(lambda: self.Reflection("Vertical"))
        self.pushButton_5.clicked.connect(lambda: self.Reflection("Horizontal"))
        self.pushButton_6.clicked.connect(self.SetGlobalImage)
        self.pushButton_8.clicked.connect(self.ToggleCrop)
        self.pushButton_9.clicked.connect(self.Shift)
        self.pushButton_10.clicked.connect(self.RgbTOHsi)
        self.pushButton_11.clicked.connect(self.HsiToRgb)
        self.pushButton_12.clicked.connect(self.HistSt)
        self.pushButton_13.clicked.connect(self.HistEqu)
        self.pushButton_14.clicked.connect(self.funcTf)
        self.pushButton_15.clicked.connect(self.filtmask)
        self.pushButton_16.clicked.connect(self.GammaTf)
        self.pushButton_17.clicked.connect(self.LPfilter)
        self.pushButton_18.clicked.connect(self.HPfilter)
        self.pushButton_19.clicked.connect(self.Thold)
        self.pushButton_20.clicked.connect(self.CSpec)
        self.pushButton_21.clicked.connect(self.Spec)
        self.pushButton_24.clicked.connect(self.BRF)
        self.pushButton_22.clicked.connect(self.SPN)
        self.pushButton_23.clicked.connect(self.Efficiency)
        self.pushButton_25.clicked.connect(self.GN)
        self.pushButton_26.clicked.connect(self.WF)
        self.pushButton_27.clicked.connect(self.Er)
        self.pushButton_28.clicked.connect(self.Dl)
        self.pushButton_29.clicked.connect(self.MG)
        self.pushButton_30.clicked.connect(self.TH)
        
        #-----SliderCallBacks--------------------------------
        self.horizontalSlider.valueChanged.connect(lambda: self.fixValue(1))
        self.horizontalSlider_2.valueChanged.connect(lambda: self.fixValue(2))
        
        #InitialValues
        self.lineEdit.setText("3")
        self.lineEdit_2.setText("3")
        self.lineEdit_3.setText("100")
        self.lineEdit_4.setText("100")
        
        #EventCall
        self.label.mousePressEvent = self.getPixel
        
        self.label.mouseReleaseEvent = self.Crop
    
        

        
    def getPixel(self,event):
        if self.EnableCrop == 1:
            #GetRowAndCoulumInfo
            self.Srow = event.pos().y()
            self.Scoulum = event.pos().x()
            
    
    def Crop(self,event):
        #GetRowAndCoulumInfo
        if self.EnableCrop == 1:
            Frow = event.pos().y()
            Fcoulum = event.pos().x()
            
            #----SetVariables----------------
            im = self.OrginalImage
            arr = np.array(im)
            #----DecisonOfCroppedSquare------
            #Bottom-Right-Square
            if Frow > self.Srow and Fcoulum > self.Scoulum:
                #ActualCropSize
                row    = Frow-self.Srow
                coulum = Fcoulum-self.Scoulum
                newarr = np.zeros((row,coulum,3))
                for l in range(3):
                    for i in range(row):
                        for j in range(coulum):
                            newarr[i][j][l] = arr[i+self.Srow][j+self.Scoulum][l]
            #Bottom-Left-Square
            if Frow > self.Srow and Fcoulum < self.Scoulum:
                #ActualCropSize
                row    = Frow-self.Srow
                coulum = self.Scoulum-Fcoulum
                newarr = np.zeros((row,coulum,3))
                for l in range(3):
                    for i in range(row):
                        for j in range(coulum):
                            newarr[i][j][l] = arr[i+self.Srow][Fcoulum+j][l]
            #Up-Left-Square
            if Frow < self.Srow and Fcoulum < self.Scoulum:
                #ActualCropSize
                row    = self.Srow-Frow
                coulum = self.Scoulum-Fcoulum
                newarr = np.zeros((row,coulum,3))
                for l in range(3):
                    for i in range(row):
                        for j in range(coulum):
                            newarr[i][j][l] = arr[i+Frow][Fcoulum+j][l]
            #Up-Right-Square
            if Frow < self.Srow and Fcoulum > self.Scoulum:
                #ActualCropSize
                row    = self.Srow-Frow
                coulum = Fcoulum-self.Scoulum
                newarr = np.zeros((row,coulum,3))
                for l in range(3):
                    for i in range(row):
                        for j in range(coulum):
                            newarr[i][j][l] = arr[i+Frow][self.Scoulum+j][l]
                            
             #ChangeForm
            self.OngoingImage = Image.fromarray(np.uint8(newarr))
            self.Show(2,self.OngoingImage)
             #SendResult
             
    


            
            
            
    def ToggleCrop(self):
        
        #ToggleTheEnableofCroppingWhenColorofButtonischanging
        #GreenIfitsEnabled
        if self.EnableCrop == 1:
            self.pushButton_8.setStyleSheet("background-color:rgb(255, 0, 0)")
            self.EnableCrop = 0
        else:
            #RedifitsDisabled
            self.pushButton_8.setStyleSheet("background-color:rgb(34, 255, 48)")
            self.EnableCrop = 1
        
       
        
       
        
        
        
    #-------LoadImage-----------
    def LoadImage(self):
        #-----------GetAdressValue-------------
        try:
            filename = QFileDialog.getOpenFileName()
        except:
            return
        #-----------OpenImageOnPillowClassAndConvertItoRGBforProcess-----
        self.OrginalImage = Image.open(filename[0]).convert('RGB')
        
        #-----------ShowLoadedImage------------
        self.Show(1,self.OrginalImage)
        
    #--------SaveImage--------
    def SaveImage(self):
        text, ok = QInputDialog.getText(self,'Save Data','Name your save file')
        #CheckforName
        if ok:
            self.OngoingImage.save(text+".png")
    
    #SettingGlobalImageToNavigateAroundQuickly
    def SetGlobalImage(self):
        self.OrginalImage = self.OngoingImage
        self.Show(1,self.OrginalImage)
        
        
    #------ShowAnyPillowFormatImageOnPixmapWidgets----   
    def Show(self,chose,image):
        
        #------------ConvertImagePıltoQpixmap--------------
        im = ImageQt.ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(im)
        pixmap.detach()
        
        
        
        

        #--------------Chose-Pixmap----WhereToShow--------
        if chose == 1:
            #FixTheSizeOftheLabelForEventCrop
            row    =  pixmap.width()
            coulum =  pixmap.height()
            self.label.setFixedSize(row,coulum)
            self.label.setPixmap(pixmap)
        if chose == 2:
            self.label_2.setPixmap(pixmap)
            
    
    
            
            
            
            
            
    #FunctionCalling
    def Reflection(self,Head):
        self.OngoingImage = ImageProcessLib.Reflection(self.OrginalImage,Head)
        self.Show(2,self.OngoingImage)    
    #FunctionCalling
    def Re_sizeScale(self):
        self.OngoingImage = ImageProcessLib.ReSizeScale(self.OrginalImage,self.doubleSpinBox.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
        #FunctionCalling
    def Re_sizePixels(self):
        self.OngoingImage = ImageProcessLib.ReSizePixel(int(self.lineEdit_2.text()),int(self.lineEdit.text()),self.OrginalImage,self.progressBar)
        return self.OngoingImage
    #FunctionCalling
    def Shift(self):
        self.OngoingImage = ImageProcessLib.Shift(self.OrginalImage,int(self.lineEdit_4.text()),int(self.lineEdit_3.text()))
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def RgbTOHsi(self):
        self.OngoingImage = ImageProcessLib.RGBtoHSI(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def HsiToRgb(self):
        self.OngoingImage = ImageProcessLib.HSItoRGB(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def HistSt(self):
        self.OngoingImage = ImageProcessLib.HistStr(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def HistEqu(self):
        self.OngoingImage = ImageProcessLib.HistEqu(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def funcTf(self):
        self.OngoingImage = ImageProcessLib.Functf(self.OrginalImage,self.horizontalSlider.value(),self.horizontalSlider_2.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def filtmask(self):
        self.OngoingImage = ImageProcessLib.FilterImage(self.OrginalImage,self.comboBox.currentIndex())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def GammaTf(self):
        self.OngoingImage = ImageProcessLib.GammaFt(self.OrginalImage,float(self.lineEdit_5.text()))
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def LPfilter(self):
        self.OngoingImage = ImageProcessLib.LowPass(self.OrginalImage,self.spinBox.value(),self.spinBox_4.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def HPfilter(self):
        self.OngoingImage = ImageProcessLib.HighPass(self.OrginalImage,self.spinBox_2.value(),self.spinBox_5.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def Thold(self):
        self.OngoingImage = ImageProcessLib.Threshold(self.OrginalImage,self.spinBox_3.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def CSpec(self):
        self.OngoingImage = ImageProcessLib.CSpectrum(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def Spec(self):
        self.OngoingImage = ImageProcessLib.Spectrum(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def BRF(self):
        self.OngoingImage = ImageProcessLib.BReject(self.OrginalImage,self.spinBox_6.value(),self.spinBox_7.value(),self.spinBox_8.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def SPN(self):
        self.OngoingImage = ImageProcessLib.SPnoise(self.OrginalImage,self.spinBox_9.value())
        self.Show(2,self.OngoingImage)
    def Efficiency(self):
        self.label_21.setText( ImageProcessLib.Eff(self.OrginalImage,self.OngoingImage) + "%" )
    #FunctionCalling
    def GN(self):
        self.OngoingImage = ImageProcessLib.Gnoise(self.OrginalImage,self.spinBox_10.value(),self.doubleSpinBox_2.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def WF(self):
        self.OngoingImage = ImageProcessLib.winF(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def Er(self):
        self.OngoingImage = ImageProcessLib.erosion(self.OrginalImage,self.spinBox_11.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def Dl(self):
        self.OngoingImage = ImageProcessLib.dilation(self.OrginalImage,self.spinBox_12.value())
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def MG(self):
        self.OngoingImage = ImageProcessLib.morpGrad(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    #FunctionCalling
    def TH(self):
        self.OngoingImage = ImageProcessLib.topHat(self.OrginalImage)
        self.Show(2,self.OngoingImage)
    
        
    
    
    #For each value changing the transfer function updates itself and other variables too
    def fixValue(self,chose):
        if chose == 1:
            self.label_9.setText(str(self.horizontalSlider.value()))
        if chose == 2:
            self.label_10.setText(str(self.horizontalSlider_2.value()))
        #FunctionRepresent
        t = np.linspace(0,255)
        func = t*(self.horizontalSlider_2.value()-self.horizontalSlider.value())/255+self.horizontalSlider.value()   
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(t,func)
        self.MplWidget.canvas.draw()


app = QApplication([])
window = ImageProcessInterface()
window.show()
app.exec_()



