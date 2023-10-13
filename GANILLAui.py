from numpy.core.fromnumeric import var
from models import create_model
from data import CreateDataLoader
import sys, os, math, threading, time
import tkinter as tk
import tkinter.font as font
from tkinter import *
from tkinter import tix,ttk, filedialog, scrolledtext
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

img_refrence = []
img_name_list = []
current_page = -1

running = False

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

def getImgeList(pageNum, imgOnPage):
    img_path_list = []
    for root, dirs, files in os.walk(opt.results_dir):
        for name in files:
            if name.endswith((".png")):
                img_path_list.append(root+"/"+name)

    chunks = [img_path_list[x:x+imgOnPage] for x in range(0, len(img_path_list), imgOnPage)]

    return chunks[pageNum]


def openImgViewWindow(event, obj):

    imageViewer = Toplevel(window)

    canvas = Canvas(imageViewer, width = 1000, height = 1000)  
    canvas.pack()  
    img = ImageTk.PhotoImage(Image.open(img_name_list[obj]).resize((1000,1000), Image.ANTIALIAS))
    canvas.create_image(20, 20, anchor=NW, image=img)

    imageViewer.mainloop()

#def loadImages(canvas):
    
    

def pageShift(button, canvas):

    max_img_page = 6 #max images per page
    image_size = 250
    global current_page

    if button=="next": current_page += 1
    else: current_page -= 1
        
    list_of_images = getImgeList(current_page, max_img_page)
    canvas.delete("all")
    img_name_list.clear()
    img_refrence.clear()

    counterx=1
    countery=0
    img_counter=0
    for img in list_of_images:
        if img_counter < max_img_page:

            if counterx>7:
                countery+=image_size #used as an offset for Y axis so we can display images underneath
                counterx=1 #counting number of images being displayed on the current X axis
                        
            #need to use try/catch as one of the images is a broken PNG which will casuse crashes (temp fix)
            try:
                image = ImageTk.PhotoImage(Image.open(img).resize((image_size,image_size), Image.ANTIALIAS))
            except:
                print("\nBroken PNG somewhere") 

            img_refrence.append(image) #saving the image OBJECT refrence so that they don't get cleaned on loop iterations
            img_name_list.append(img) #save image paths to be able to display them individually in openImgViewWindow()

            #setting up the invisible buttons to lay over the images so they can be individual clicked
            root_img = canvas.create_image((image_size*counterx) - image_size,countery, image=image, anchor='nw', state=NORMAL)
            #blank = canvas.create_image((image_size*counterx) - image_size,countery, image=blankImage, state=NORMAL, anchor='nw')
            canvas.tag_bind(root_img, "<Button-1>",lambda event, obj=img_counter: openImgViewWindow(event, obj))

            #counter to keep track of loop and images displayed
            img_counter+=1
            counterx+=1
        else:
            break 
    
# def openResultsDir:
#     filedialog.askdirectory(initialdir ="./", title = "Select Folder Containing Model")

def openResultWindow():

    #basic window infos and the 2 buttons are declared here
    resultsWindow = Toplevel(window)
    resultsWindow.title("Results Window") 

    #blankImage = ImageTk.PhotoImage(Image.open("blank.png").resize((image_size,image_size), Image.ANTIALIAS)) #used for the image "buttons" which are laid over with invisible buttons
    #list_of_images = getImgeList(current_page) #uses a function to crawl through the results directory and create a lists of all the paths to images
    #num_of_pages = math.ceil((len(list_of_images)) / max_img_page) #gets length of image list and divides is by the number of images allowed per page, we round the number up to
    
    #setup the frame and canvas which sits inside the frame. The canvas displays the images while the frame allows us to make it all scrollable
    frame=Frame(resultsWindow,width=1500,height=250)
    frame.pack(expand=True, fill=BOTH)             
    canvas  = Canvas(frame, width = 1500, height = 250, bg='white', scrollregion=(0,0,1500,250))

    btnNext = Button(frame, text='Next', bg="black", fg="white", font='Helvetica 8 bold', width=10, height=1, command= lambda: pageShift("next", canvas))
    btnPrev = Button(frame, text='Previous', bg="black", fg="white", font='Helvetica 8 bold', width=10, height=1, command= lambda: pageShift("prev", canvas))
    #canvas.tag_bind(root_img, "<Button-1>",lambda event, obj=img_counter: openImgViewWindow(event, obj))

    #setup the scroll bar
    hbar=Scrollbar(frame,orient=HORIZONTAL)
    hbar.pack(side=BOTTOM,fill=X)
    hbar.config(command=canvas.xview)
    vbar=Scrollbar(frame,orient=VERTICAL)
    vbar.pack(side=RIGHT,fill=Y)
    vbar.config(command=canvas.yview)
    canvas.config(width=1500,height=250)
    canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    #pack all the ui elemtns
    btnPrev.pack(side = BOTTOM, anchor=W)
    btnNext.pack(side = BOTTOM, anchor=W)
    canvas.pack(side=LEFT,expand=True,fill=BOTH)

    pageShift("next", canvas)
    
    resultsWindow.mainloop()

def cancel_convert():
    print("Cancelling Conversion...")
    global running
    running = False
    print("==============Cancelled=================")
    progress_var.set(0)
    
def start_convert():
    global running
    running = True
    thread = threading.Thread(target=convert)
    print("Starting Conversion...")
    thread.start()

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
            while running:
                if i >= opt.num_test or running == False:
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
                save_images(opt.results_dir, visuals, img_path,  save_both=opt.save_both, aspect_ratio=opt.aspect_ratio)
                progress_var.set(i+1)
                if(opt.remove_images):
                    os.remove(img_path[0])
                    print('removed image', img_path[0])
    except KeyboardInterrupt:
        progress_var.set(0)
        print("==============Cancelled==============")
        raise
    except Exception as e:
        print(e)
        raise

window = tix.Tk()  
window.title('GANILLA UI - TEST') 
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
lblTitle = Label(window, text='GANILLA UI - TEST', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
lblSub = Label(window, text='CONVERT POOR QUALITY CAPTURES TO HIGH QUALITY CAPTURES', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)

lblFoot = Label(window, text='CREATED BY GM, ND, & CD', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)

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
fontDir = font.Font(size=8)
btnSetResultsDir = Button(frameInput, text='Set Results Directory', font='Helvetica 10', bg="white", command=setResultsDir)
btnSetResultsDir['font'] = fontDir
btnSetResultsDir.img = PhotoImage()
btnSetResultsDir.config(height=20, width=100, image=btnSetResultsDir.img, compound=CENTER)

tip.bind_widget(btnSetResultsDir, balloonmsg="test")
btnConv = Button(frameConvert, text='Start Conversion', font='Helvetica 10', width=12, height=1, command=start_convert, bg="white")
tip.bind_widget(btnConv, balloonmsg="test")
btnResult = Button(frameConvert, text='Results Window', font='Helvetica 10', width=12, height=1, command=openResultWindow, bg="white")
tip.bind_widget(btnResult, balloonmsg="test")
btnCancel = Button(window, text='Cancel', font='Helvetica 10', width=12, height=1, command=cancel_convert, bg="white")

#placing all the UI objects on screen
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)

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


progressbar.pack(side = TOP, pady=(40,20), padx=10, anchor=W)
btnCancel.pack(side = TOP, padx=(10,0), anchor=W)
#progressbar.pack(side = LEFT, padx=10)


outputBox = scrolledtext.ScrolledText(window,
                                      padx = 5,
                                      pady = 5,
                                      wrap = tk.WORD,  
                                      width = 60,  
                                      height = 28,  
                                      font = ("Arial", 
                                              10)) 

outputBox.place(x=251, y=75) 

sys.stdout = StdoutRedirector( outputBox )
print("Please select the folder containing the model and the folder containing the dataset. Followed by the target results directory if desired.")
window.mainloop()