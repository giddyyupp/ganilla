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

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self, text_area):
        self.text_area = text_area

class StdoutRedirector(IORedirector):
    '''A class for redirecting stdout to this Text widget.'''
    def write(self,str):
        sys.stderr.write(str)
        self.text_area.insert(END, str)

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


def getDirectory():
    path = filedialog.askdirectory(initialdir = "/",
                                   title = "Select Dataroot")
    opt.dataroot = path
    #lblDataroot.configure(text="IMAGE DIRECTORY: \n"+path)
    
def getCheckpoints():
    checkpoints = filedialog.askdirectory(initialdir = "/",
                                title = "Select Folder Containing Model")
    opt.checkpoints_dir = checkpoints
    #lblCheckpoints.configure(text="IMAGE DIRECTORY: \n" + checkpoints)

def selectGAN(self):
    print(self.cget('text'))

def openViewWin(): 
    newWindow = Toplevel(window)

    newWindow.title("New Window") 
  
    # sets the geometry of toplevel 
    newWindow.geometry("200x200") 
  
    # A Label widget to show in toplevel 
    Label(newWindow,  text ="This is a new window").pack()
    
  

def convert():
    progressbar.start(250)
    if(chkGpuVar.get() == 0):
        opt.gpu_ids.clear()
    #opt.remove_images = chkDelVar.get()
    opt.epoch = drpEpochOp.get()
    opt.resize_or_crop = drpResizeOp.get()
    
    if(opt.resize_or_crop.__contains__('scale')):
        for i in range(len(validSizes) - 2):
            if (sclFineVar.get() < validSizes[i+1] and sclFineVar.get() >= validSizes[i]):
                opt.fineSize = validSizes[i]
            if (sclLoadVar.get() < validSizes[i+1] and sclLoadVar.get() >= validSizes[i]):
                opt.fineSize = validSizes[i]        
                             
    print(testOptions.return_options(opt))
    try:
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        
        window.update()
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
            save_images(webpage, visuals, img_path,  save_both=opt.save_both, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            
            if(opt.remove_images):
                os.remove(img_path[0])
                print('removed image', img_path[0])
                
            # save the website
            webpage.save()
            
        print('finished')
        progressbar.stop()         
    except Exception as e:
        print(e)
        raise                                                                                              

window = tix.Tk()  
window.title('File Explorer') 
window.geometry("700x575")
window.resizable(False, False)
window.configure(bg="white")

progressbar = ttk.Progressbar(length=230)

tip = tix.Balloon(window)
tip.label.configure(bd=0)

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

lblGan=  Label(window, text='GAN Type', font='Helvetica 10 bold', bg="white")
btnGanilla = Button(frameGAN, text='GANILLA', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnGanilla), bg="white")
tip.bind_widget(btnGanilla, balloonmsg="test")
btnCycle = Button(frameGAN, text='CycleGAN', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnCycle), bg="white")
tip.bind_widget(btnCycle, balloonmsg="test")

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
sclLoad = Scale(window, from_=256, to=3216, orient=HORIZONTAL, length=225, resolution=16, variable = sclLoadVar, bg="white")
tip.bind_widget(sclLoad, balloonmsg="test")

btnModel = Button(frameModel, text='Select Model', font='Helvetica 10', width=12, height=1, command= lambda: selectGAN(btnCycle), bg="white")
tip.bind_widget(btnModel, balloonmsg="test")
chkGpuVar = IntVar()
chkGpu = Checkbutton(frameModel, text='Use GPU', onvalue=1, offvalue=0, variable  = chkGpuVar, bg="white")
tip.bind_widget(chkGpu, balloonmsg="test")

btnInput = Button(frameInput, text='Input Directory', font='Helvetica 10', width=12, height=1, command=getDirectory, bg="white")
tip.bind_widget(btnInput, balloonmsg="test")
btnOutput = Button(frameInput, text='Output Directory', font='Helvetica 10', width=12, height=1, bg="white")
tip.bind_widget(btnOutput, balloonmsg="test")
btnConv = Button(frameConvert, text='Convert', font='Helvetica 10', width=12, height=1, command=convert, bg="white")
tip.bind_widget(btnConv, balloonmsg="test")
btnResult = Button(frameConvert, text='Results', font='Helvetica 10', width=12, height=1, command = openViewWin, bg="white")
tip.bind_widget(btnResult, balloonmsg="test")

#placing all the UI objects on screen
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)
#btnInfo.pack(ipadx=5, ipady=5)

lblGan.pack(side=TOP, pady=10, padx=10, anchor=W)
frameGAN.pack(side = TOP, anchor=W)
btnGanilla.pack(side = LEFT, padx=10)
btnCycle.pack(side = LEFT, padx=10)

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
btnInput.pack(side = LEFT, anchor=W)
btnOutput.pack(side = LEFT, padx=(20,0), anchor=W)
frameConvert.pack(side = TOP, pady=(20,0), padx=10, anchor=W)
btnConv.pack(side = LEFT, anchor=W)
btnResult.pack(side = LEFT, padx=(20,0), anchor=W)

progressbar.pack(side = LEFT, padx=10)

outputBox = scrolledtext.ScrolledText(window,  
                                      wrap = tk.WORD,  
                                      width = 60,  
                                      height = 29,  
                                      font = ("Arial", 
                                              10)) 

outputBox.place(x=251, y=75) 


sys.stdout = StdoutRedirector( outputBox )

# Let the window wait for any events 
window.mainloop()
# line_queue = getconsole.Queue(maxsize=1000)

# # create a process output reader
# reader = getconsole.ProcessOutputReader(line_queue, 'python3', params=['-u', 'test.py'])

# # create a console
# root = Tk()
# console = getconsole.MyConsole(root, line_queue)


# reader.start()   # start the process
# console.pack()   # make the console visible 
# root.mainloop()

# reader.stop()
# reader.join(timeout=5)  # give thread a chance to exit gracefully

# if reader.is_alive():
#     raise RuntimeError("process output reader failed to stop")
