from numpy.core.fromnumeric import var
from models import create_model
from data import CreateDataLoader
import sys
import os
import tkinter as tk
from tkinter import *
from tkinter import tix
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext 
from options.test_options import TestOptions
from util.visualizer import save_images
from util import html
from PIL import ImageTk, Image

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self,  text_area):
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


def getDataroot():
    dataroot = filedialog.askdirectory(initialdir = "..", title = "Select Dataroot")
    opt.dataroot = dataroot
    print('\nDataroot set as ', opt.dataroot)

    
def setResultsDir():
    resultsPath = filedialog.askdirectory(initialdir = "./", title = "Select Results Target")
    opt.results_dir = resultsPath
    print('\nResults directory set as ', opt.results_dir)
    
def getCheckpoints():
    checkpoints = filedialog.askdirectory(initialdir ="./", title = "Select Folder Containing Model")
    opt.checkpoints_dir = checkpoints
    print('\nModel directory set as ', opt.checkpoints_dir)




def openImageView(event, obj):

    imageViewer = Toplevel(window)

    canvas = Canvas(imageViewer, width = 1000, height = 1000)  
    canvas.pack()  
    img = ImageTk.PhotoImage(Image.open(opt.results_dir + img_name_list[obj]).resize((1000,1000), Image.ANTIALIAS))
    canvas.create_image(20, 20, anchor=NW, image=img)

    imageViewer.mainloop()
    
# def openResultsDir:
#     filedialog.askdirectory(initialdir ="./", title = "Select Folder Containing Model")

def opeNewWindow(winType):

    newWindow = Toplevel(window)
    winTitle=""
    winDescription=""
    list_of_images = []
    #blankImage = ImageTk.PhotoImage(Image.open("blank.png").resize((250,250), Image.ANTIALIAS))

    if winType == "results":
        winTitle="Results Window"
        #winDescription="Below you'll find the translated images"


        for image in os.listdir(opt.results_dir):
            if image.endswith("png"):
                list_of_images.append(image)

        scroll_length = (len(list_of_images)/7) * 250


        frame=Frame(newWindow,width=500,height=500)
        frame.pack(expand=True, fill=BOTH) #.grid(row=0,column=0)
       
                    
        canvas  = Canvas(frame, width = 500, height = 500, bg='white', scrollregion=(0,0,1000,scroll_length))
        
        hbar=Scrollbar(frame,orient=HORIZONTAL)
        hbar.pack(side=BOTTOM,fill=X)
        hbar.config(command=canvas.xview)
        vbar=Scrollbar(frame,orient=VERTICAL)
        vbar.pack(side=RIGHT,fill=Y)
        vbar.config(command=canvas.yview)
        canvas.config(width=500,height=500)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        counterx=1
        countery=0
        img_counter=0
        for img in list_of_images:
            if counterx>7:
                countery+=250
                counterx=1
            image = ImageTk.PhotoImage(Image.open(opt.results_dir + img).resize((250,250), Image.ANTIALIAS))
            img_list.append(image)
            img_name_list.append(img)
            button = canvas.create_image((250*counterx) - 250,countery, image=image, anchor='nw')
            blank = canvas.create_image((250*counterx) - 250,countery, state=NORMAL, anchor='nw')
            canvas.tag_bind(blank, "<Button-1>",lambda event, obj=img_counter: openImageView(event, obj))
            img_counter+=1
            counterx+=1
        
        
        
    else:
        winTitle="Information Window"
        winDescription="Infomration about the team and project!"
        
    
    newWindow.title(winTitle) 
    newWindow.geometry("500x500")
    
    Label(newWindow, text=winDescription).pack()
    canvas.pack(side=LEFT,expand=True,fill=BOTH)
    
    #newWindow.mainloop()
    

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
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        
        progressbar.configure(maximum=len(dataset))
        #progressbar.start(len(dataset))
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
            file_object = open('conversion_progress.txt', 'a')
            # Append 'hello' at the end of file
            file_object.write(mess+'\n')
            # Close the file
            file_object.close()
            save_images(opt.results_dir, visuals, img_path,  save_both=opt.save_both, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            progress_var.set(i+1)
            if(opt.remove_images):
                os.remove(img_path[0])
                print('removed image', img_path[0])

        print('finished')
    except KeyboardInterrupt:
        progress_var.set(0)
        print("==============Cancelled==============")
        raise
    except Exception as e:
        print(e)
        raise

window = tix.Tk()  
window.title('Chickpea roots GANILLA User Interface') 
window.geometry("700x575")
window.resizable(False, False)
window.configure(bg="white")


progress_var = DoubleVar()
progressbar = ttk.Progressbar(variable=progress_var, length=230)

tip = tix.Balloon(window)
tip.label.configure(bd=0)

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

frameLabels = Frame(window)
frameLabels.configure(bg="white")

# frameCheckpoints

frameCheckpoints = Frame(frameLabels).pack(side=LEFT, padx=10)
frameDataroot = Frame(frameLabels).pack(side=LEFT, padx=10)
frameModelDir = Frame(frameLabels).pack(side=LEFT, padx=10)
frameResultsDir = Frame(frameLabels).pack(side=LEFT, padx=10)


# setting varaibles
drpEpochOp = StringVar(window)
drpEpochOp.set("14")
drpResizeOp = StringVar(window)
drpResizeOp.set("scale_width")

#creating all the UI objects (lables, buttons, inputs)
lblTitle = Label(window, text='ROOT ENHANCE', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
lblSub = Label(window, text='CONVERT POOR QUALITY CAPTURES TO HIGH QUALITY CAPTURES', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
lblFoot = Label(window, text='CREATED BY THE ROOT ENHANCE TEAM', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
btnInfo = Button(window, text='INFORMATION', bg="black", fg="white", font='Helvetica 8 bold', width=10, height=1)



lblEpoch = Label(frameEpochLabel, text='Epoch no.', font='Helvetica 10 bold', bg="white")
lblResize = Label(frameEpochLabel, text='Resize', font='Helvetica 10 bold', bg="white")
drpEpoch = OptionMenu(frameEpoch, drpEpochOp, "1", "2", "3","5","6","7","8","10","11","12","13","14", "15")
drpEpoch.configure(bg="white")
tip.bind_widget(drpEpoch, balloonmsg="test")
drpResize = OptionMenu(frameEpoch, drpResizeOp, "resize_and_crop", "scale_width", "scale_width_and_crop", "none")
drpResize.configure(width=11, bg="white")
tip.bind_widget(drpResize, balloonmsg="test")

lblFine = Label(window, text='Fine Size', font='Helvetica 10 bold', bg="white")
sclFineVar = IntVar()
sclFine = Scale(window, from_=256, to=3216, orient=HORIZONTAL, length=225, resolution=16, variable = sclFineVar, bg="white")
tip.bind_widget(sclFine, balloonmsg="test")

lblLoad = Label(window, text='Load Size', font='Helvetica 10 bold', bg="white")
sclLoadVar = IntVar()
sclLoad = Scale(window, from_=0, to=3216, orient=HORIZONTAL, length=225, resolution=16, variable = sclLoadVar, bg="white")
tip.bind_widget(sclLoad, balloonmsg="test")

btnModel = Button(frameModel, text='Select Model', font='Helvetica 10', width=12, height=1, command= getCheckpoints, bg="white")
tip.bind_widget(btnModel, balloonmsg="test")
chkGpuVar = IntVar()
chkGpu = Checkbutton(frameModel, text='Use GPU', onvalue=1, offvalue=0, variable  = chkGpuVar, bg="white")
tip.bind_widget(chkGpu, balloonmsg="test")

btnSetDataroot = Button(frameInput, text='Set Dataroot', font='Helvetica 10', width=12, height=1, command=getDataroot, bg="white")
tip.bind_widget(btnSetDataroot, balloonmsg="test")
btnSetResultsDir = Button(frameInput, text='Set Results Dir', font='Helvetica 10', width=12, height=1, bg="white", command=setResultsDir)
tip.bind_widget(btnSetResultsDir, balloonmsg="test")
btnConv = Button(frameConvert, text='Start Conversion', font='Helvetica 10', width=12, height=1, command=convert, bg="white")
tip.bind_widget(btnConv, balloonmsg="test")
btnResult = Button(frameConvert, text='Results Window', font='Helvetica 10', width=12, height=1, command=lambda: opeNewWindow("results"), bg="white")
tip.bind_widget(btnResult, balloonmsg="test")

#placing all the UI objects on screen
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)
#btnInfo.pack(ipadx=5, ipady=5)


frameEpochLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblEpoch.pack(side = LEFT, padx=(0,40))
lblResize.pack(side = LEFT, padx=(15,0))
frameEpoch.pack(side = TOP, anchor=W, padx=10)
drpEpoch.pack(side = LEFT)
drpResize.pack(side = LEFT, padx=(65,0))

lblFine.pack(side = TOP, pady=(10,0), padx=10, anchor=W)
sclFine.pack(side = TOP, padx=10, anchor=W)

lblLoad.pack(side = TOP, pady=(10,0), padx=10, anchor=W)
sclLoad.pack(side = TOP, padx=10, anchor=W)

frameModel.pack(side = TOP, pady=20, padx=10, anchor=W)
btnModel.pack(side = LEFT, padx=(0,15), anchor=W)
chkGpu.pack(side = LEFT, anchor=W)

frameInput.pack(side = TOP, padx=10, anchor=W)
btnSetDataroot.pack(side = LEFT, anchor=W)
btnSetResultsDir.pack(side = LEFT, padx=(20,0), anchor=W)
frameConvert.pack(side = TOP, pady=(20,0), padx=10, anchor=W)
btnConv.pack(side = LEFT, anchor=W)
btnResult.pack(side = LEFT, padx=(20,0), anchor=W)
progressbar.pack(side = LEFT, padx=10)



outputBox = scrolledtext.ScrolledText(window,
                                      padx = 5,
                                      wrap = tk.WORD,  
                                      width = 60,  
                                      height = 29,  
                                      font = ("Arial", 
                                              10)) 

outputBox.place(x=251, y=75) 

sys.stdout = StdoutRedirector( outputBox )
print("\nPlease select the folder containing the model and the folder containing the dataset. Followed by the target results directory if desired.")
window.mainloop()

