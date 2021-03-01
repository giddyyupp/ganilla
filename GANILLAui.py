from numpy.core.fromnumeric import var
from models import create_model
from data import CreateDataLoader
import sys
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext 
from options.test_options import TestOptions
from util.visualizer import save_images
from util import html
from PIL import ImageTk, Image

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self,text_area):
        self.text_area = text_area

class StdoutRedirector(IORedirector):
    '''A class for redirecting stdout to this Text widget.'''
    def write(self,str):
        sys.stderr.write(str)
        self.text_area.see('end')
        self.text_area.insert(END, str)
        window.update()

validSizes = [256, 320, 384, 448, 512, 576, 640, 704, 768, 848, 912, 976, 1040, 1104, 
              1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872,
              1936, 2000, 2064, 2128, 2192, 2256, 2320, 2384, 2448, 2512, 2576, 2640, 
              2704, 2768, 2832, 2896, 2960, 3024, 3088, 3152, 3216, 3280]

testOptions = TestOptions()

opt = testOptions.initOpt()
# hard-code some parameters for test
opt.num_threads = 1   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True    # no flip
opt.display_id = -1   # no visdom display  
opt.name = ""


img_list = []
img_name_list = []

def getDirectory():
    path = filedialog.askdirectory(initialdir = "/",

                                   title = "Select Dataroot")
    opt.dataroot = dataroot
    lblDataroot.configure(text="IMAGES DATAROOT: " + dataroot)
    
    
def setResultsDir():
    resultsPath = filedialog.askdirectory(initialdir = "/",
                                   title = "Select Results Target")
    opt.results_dir = resultsPath
    lblResultsDir.configure(text="TARGET RESULT DIRECTORY: " + resultsPath)

def getCheckpoints():
    checkpoints = filedialog.askdirectory(initialdir = "/",
                                title = "Select Folder Containing Model")
    opt.checkpoints_dir = checkpoints

    lblModelDir.configure(text="MODEL SOURCE: " + checkpoints)
    


def openImageView(event, obj):

    imageViewer = Toplevel(window)

    canvas = Canvas(imageViewer, width = 600, height = 600)  
    canvas.pack()  
    img = ImageTk.PhotoImage(Image.open("display_results/"+img_name_list[obj]))
    canvas.create_image(20, 20, anchor=NW, image=img)  

    imageViewer.mainloop()

def opeNewWindow(winType):

    newWindow = Toplevel(window)
    winTitle=""
    winDescription=""
    list_of_images = []
    blankImage = ImageTk.PhotoImage(Image.open("blank.png").resize((250,250), Image.ANTIALIAS))

    if winType == "results":
        winTitle="Results Window"
        winDescription="Below you'll find the translated images"

       
                    
        canvas  = Canvas(newWindow, width = 1000, height = 1000, bg='blue')
        
        nextBtn = Button(newWindow, text="Next Image", width=17)
        nextBtn.pack()
       


        for image in os.listdir("C:/Users/cdwor/Documents/GitHub/ganilla/display_results"):
            if image.endswith("png"):
                list_of_images.append(image)


        counterx=1
        countery=0
        img_counter=0
        for img in list_of_images:
            if counterx>7:
                countery+=250
                counterx=1
            image = ImageTk.PhotoImage(Image.open("display_results/"+img).resize((250,250), Image.ANTIALIAS))
            img_list.append(image)
            img_name_list.append(img)
            button = canvas.create_image((250*counterx) - 250,countery, image=image, anchor='nw')
            blank = canvas.create_image((250*counterx) - 250,countery, image=blankImage, state=NORMAL, anchor='nw')
            canvas.tag_bind(blank, "<Button-1>",lambda event, obj=img_counter: openImageView(event, obj))
            img_counter+=1
            counterx+=1
        
        
        
    else:
        winTitle="Information Window"
        winDescription="Infomration about the team and project!"
        
    
    newWindow.title(winTitle) 
    newWindow.geometry("500x500")
    
    Label(newWindow, text=winDescription).pack()
    canvas.pack(expand = YES, fill = BOTH)
    newWindow.mainloop()
    

def convert():
    if(chkGpuVar.get() == 0):
        opt.gpu_ids.clear()
    #opt.remove_images = chkDelVar.get()
    opt.epoch = drpEpochOp.get()
    opt.resize_or_crop = drpResizeOp.get()
    
    if(opt.resize_or_crop.__contains__('scale')):
        for i in range(len(validSizes) - 2):
            if (sclFineVar.get() < validSizes[i+1] and sclFineVar.get() >= validSizes[i]):
                opt.fineSize = validSizes[i]

    print(testOptions.return_options(opt))
    try:
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
        webpage = html.HTML(opt.results_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        
        #window.update()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            mess = 'processing (%04d)-th of %04d image... %s' % (i+1, len(dataset), img_path[0])
            print(mess)
            
            # Open a file with access mode 'a'
            file_object = open('progress.txt', 'a')
            # Append 'hello' at the end of file
            file_object.write(mess+'\n')
            # Close the file
            file_object.close()
            save_images(webpage, visuals, img_path,  save_both=opt.save_both, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            window.update()
            
            if(opt.remove_images):
                os.remove(img_path[0])
                print('removed image', img_path[0])
                
            # save the website
            webpage.save()
            
        print('finished')         
    except Exception as e:
        print(e)
        raise               

                                                                                                         
window = Tk()  
window.title('File Explorer') 

window.geometry("700x535")

window.resizable(False, False)
window.configure(bg="white")

frameGAN = Frame()
frameGAN.configure(bg="white")
frameEpochLabel = Frame()
frameEpochLabel.configure(bg="white")
frameEpoch = Frame()
frameEpoch.configure(bg="white")
frameModel = Frame()
frameModel.configure(bg="white")
frameInput = Frame()
frameInput.configure(bg="white")
frameConvert = Frame()
frameConvert.configure(bg="white")

frameEpochLabel = Frame()
frameEpoch = Frame()
frameModel = Frame()
frameInput = Frame()
frameConvert = Frame()

#setting varaibles
drpEpochOp = StringVar(window)
drpEpochOp.set("14")
drpResizeOp = StringVar(window)
drpResizeOp.set("scale_width")

#creating all the UI objects (lables, buttons, inputs)
lblTitle = Label(window, text='ROOT ENHANCE', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
lblSub = Label(window, text='CONVERT POOR QUALITY CAPTURES TO HIGH QUALITY CAPTURES', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
lblFoot = Label(window, text='CREATED BY THE ROOT ENHANCE TEAM', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
btnInfo = Button(window, text='INFORMATION', bg="black", fg="white", font='Helvetica 8 bold', width=10, height=1)


lblDataroot = Label(window, text='DATAROOT DIRECTORY: {}', font='Helvetica 10 bold')
lblCheckpoints = Label(window, text='CHECKPOINTS DIRECTORY: {}', font='Helvetica 10 bold')
lblModelDir = Label(window, text='MODEL SOURCE: {}', font='Helvetica 10 bold')
lblResultsDir = Label(window, text='TARGET RESULT DIRECTORY: {}', font='Helvetica 10 bold')

lblEpoch = Label(frameEpochLabel, text='EPOCH NO.', font='Helvetica 10 bold')
lblResize = Label(frameEpochLabel, text='PREPROCESS\nRESIZING', font='Helvetica 10 bold')
drpEpoch = OptionMenu(frameEpoch, drpEpochOp, "1", "2", "3","5","6","7","8","10","11","12","13","14", "15")
drpResize = OptionMenu(frameEpoch, drpResizeOp, "resize_and_crop", "scale_width", "scale_width_and_crop", "none")

lblFine = Label(window, text='FINE SIZE', font='Helvetica 10 bold')
sclFineVar = IntVar()
sclFine = Scale(window, from_=256, to=3216, orient=HORIZONTAL, length=200 ,resolution=16, variable = sclFineVar)

lblLoad = Label(window, text='LOAD SIZE', font='Helvetica 10 bold')
sclLoadVar = IntVar()
sclLoad = Scale(window, from_=0, to=3216, orient=HORIZONTAL, length=200 ,resolution=16, variable = sclLoadVar)
sclLoad.set('3216')
btnModel = Button(frameModel, text='SELECT MODEL', font='Helvetica 10', width=15, height=1, command= getCheckpoints)
chkGpuVar = IntVar()
chkGpu = Checkbutton(frameModel, text='Use GPU', onvalue=1, offvalue=0, variable  = chkGpuVar)

btnInput = Button(frameInput, text='SET INPUT DIRECTORY', font='Helvetica 10', width=16, height=1, command=getDataroot)
btnOutput = Button(frameInput, text='SET OUTPUT DIRECTORY', font='Helvetica 10', width=16, height= 1, command=setResultsDir)
btnConv = Button(frameConvert, text='CONVERT', font='Helvetica 10', width=10, height=1, command=convert)
btnResult = Button(frameConvert, text='RESULTS', font='Helvetica 10', width=10, height=1, command = openViewWin)
=======
lblGan=  Label(window, text='GAN Type', font='Helvetica 10 bold', bg="white")
btnGanilla = Button(frameGAN, text='GANILLA', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnGanilla), bg="white")
btnCycle = Button(frameGAN, text='CycleGAN', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnCycle), bg="white")

lblEpoch = Label(frameEpochLabel, text='Epoch no.', font='Helvetica 10 bold', bg="white")
lblResize = Label(frameEpochLabel, text='Resize', font='Helvetica 10 bold', bg="white")
drpEpoch = OptionMenu(frameEpoch, drpEpochOp, "1", "2", "3","5","6","7","8","10","11","12","13","14", "15")
drpEpoch.configure(bg="white")
drpResize = OptionMenu(frameEpoch, drpResizeOp, "resize_and_crop", "scale_width", "scale_width_and_crop", "none")
drpResize.configure(width=11, bg="white")

lblFine = Label(window, text='Fine Size', font='Helvetica 10 bold', bg="white")
sclFineVar = IntVar()
sclFine = Scale(window, from_=256, to=3216, orient=HORIZONTAL, length=225, resolution=16, variable = sclFineVar, bg="white")

lblLoad = Label(window, text='Load Size', font='Helvetica 10 bold', bg="white")
sclLoadVar = IntVar()
sclLoad = Scale(window, from_=256, to=3216, orient=HORIZONTAL, length=225, resolution=16, variable = sclLoadVar, bg="white")

btnModel = Button(frameModel, text='Select Model', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnCycle), bg="white")
chkGpuVar = IntVar()
chkGpu = Checkbutton(frameModel, text='Use GPU', onvalue=1, offvalue=0, variable  = chkGpuVar, bg="white")

btnInput = Button(frameInput, text='Input Directory', font='Helvetica 10', width=12, height=1, command=getDirectory, bg="white")
btnOutput = Button(frameInput, text='Output Directory', font='Helvetica 10', width=12, height=1, bg="white")
btnConv = Button(frameConvert, text='Convert', font='Helvetica 10', width=12, height=1, command=convert, bg="white")
btnResult = Button(frameConvert, text='Results', font='Helvetica 10', width=12, height=1, command=lambda: opeNewWindow("results"), bg="white")

#placing all the UI objects on screen
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)
#btnInfo.pack(ipadx=5, ipady=5)


# lblGan.pack(side=TOP, pady=10)
# frameGAN.pack(side = TOP)
# btnGanilla.pack(side = LEFT, padx=10)
# btnCycle.pack(side = LEFT, padx=10)

frameEpochLabel.pack(side = TOP, pady=10)
lblEpoch.pack(side = LEFT, padx=(0,40))
lblResize.pack(side = LEFT, padx=(20,0))
frameEpoch.pack(side = TOP)
drpEpoch.pack(side = LEFT)
drpResize.pack(side = LEFT, padx=(40,0))

lblFine.pack(side = TOP, pady=(10,0))
sclFine.pack(side = TOP)

lblLoad.pack(side = TOP, pady=(10,0))
sclLoad.pack(side = TOP)

frameModel.pack(side = TOP, pady=20)
btnModel.pack(side = LEFT, padx=(0,40))
chkGpu.pack(side = LEFT)
lblCheckpoints.pack(side=BOTTOM)
lblDataroot.pack(side=BOTTOM)
lblModelDir.pack(side=BOTTOM)
lblResultsDir.pack(side=BOTTOM)

frameInput.pack(side = TOP)
btnInput.pack(side = LEFT)
btnOutput.pack(side = LEFT, padx=(20,0))
frameConvert.pack(side = TOP, pady=20)
btnConv.pack(side = LEFT)
btnResult.pack(side = LEFT, padx=(20,0))
outputBox = scrolledtext.ScrolledText(window,  
                                      wrap = tk.WORD,  
                                      width = 87,  
                                      height = 15,  
                                      font = ("Arial", 
                                              10)) 

outputBox.pack(side=BOTTOM) 

sys.stdout = StdoutRedirector( outputBox )

# Let the window wait for any events 
window.mainloop()
