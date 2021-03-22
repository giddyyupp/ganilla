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
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self,  text_area):
        self.text_area = text_area

class StdoutRedirector(IORedirector):
    '''A class for redirecting stdout to this Text widget.'''
    def write(self,str):
        print(str, file=sys.stderr)
        #sys.stderr.write(str)
        self.text_area.see('end')
        self.text_area.insert(END, str)
        window.update()

#added () for test
opt = TrainOptions().initOpt()

# hard-code some parameters for test
opt.name = ""

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

<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes
def convert():
    if __name__ == '__main__':
        try:
            opt = TrainOptions().parse()
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)
            print('#training images = %d' % dataset_size)

            model = create_model(opt)
            model.setup(opt)
            visualizer = Visualizer(opt)
            total_steps = 0

            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                for i, data in enumerate(dataset):
                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    visualizer.reset()
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)
                    model.optimize_parameters()

                    if total_steps % opt.display_freq == 0:
                        save_result = total_steps % opt.update_html_freq == 0
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_steps % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                    if total_steps % opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_steps %d)' %
                            (epoch, total_steps))
                        model.save_networks('latest')

                    iter_data_time = time.time()
                if epoch % opt.save_epoch_freq == 0:
                    print('saving the model at the end of epoch %d, iters %d' %
                        (epoch, total_steps))
                    model.save_networks('latest')
                    model.save_networks(epoch)

                print('End of epoch %d / %d \t Time Taken: %d sec' %
                    (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
                model.update_learning_rate()
        except KeyboardInterrupt:
            print("==============Cancelled==============")
            raise
        except Exception as e:
            print(e)
            raise

window = tix.Tk()  
window.title('GANILLA UI - TRAIN') 
window.geometry("750x575")
window.resizable(False, False)
window.configure(bg="white")

<<<<<<< Updated upstream
=======
progress_var = DoubleVar()
progressbar = ttk.Progressbar(variable=progress_var, length=280)

>>>>>>> Stashed changes
frameDatasetLabel = Frame()
frameDatasetLabel.configure(bg="white")
frameDataset = Frame()
frameDataset.configure(bg="white")
frameLoadLabel = Frame()
frameLoadLabel.configure(bg="white")
frameLoad = Frame()
frameLoad.configure(bg="white")
frameEpochLabel = Frame()
frameEpochLabel.configure(bg="white")
frameEpoch = Frame()
frameEpoch.configure(bg="white")
frameResize = Frame()
frameResize.configure(bg="white")

#UI elements
outputBox = scrolledtext.ScrolledText(window,
                                      padx = 5,
                                      pady = 5,
                                      wrap = tk.WORD,  
                                      width = 60,  
                                      height = 29,  
                                      font = ("Arial", 
                                              10)) 

lblTitle = Label(window, text='GANILLA UI - TRAIN', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
lblSub = Label(window, text='TRAIN A MODEL WITH A DATASET...', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
<<<<<<< Updated upstream
lblFoot = Label(window, text='CREATED BY GM, CD & ND', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
=======
lblFoot = Label(window, text='CREATED BY GM, ND, & CD', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
>>>>>>> Stashed changes

lblName = Label(window, text='Model Name', font='Helvetica 10 bold', bg="white")
txtName = Entry(window, width = 20, bg="white") #textVariable = modelName

lblDataset = Label(frameDatasetLabel, text='Dataset Directory', font='Helvetica 10 bold', bg="white")
lblCheckpoints = Label(frameDatasetLabel, text='Checkpoints Directory', font='Helvetica 10 bold', bg="white")
btnDataset = Button(frameDataset, text='Select Dataset', font='Helvetica 10', width=14, bg="white")
btnCheckpoints = Button(frameDataset, text='Select Checkpoints', font='Helvetica 10', width=14, bg="white")

lblLoadSize = Label(frameLoadLabel, text='Load Size', font='Helvetica 10 bold', bg="white")
lblFineSize = Label(frameLoadLabel, text='Fine Size', font='Helvetica 10 bold', bg="white")
txtLoadSize = Entry(frameLoad, width = 10, bg="white")
txtFineSize = Entry(frameLoad, width = 10, bg="white")

#lblContinue = Label(window, text='Continue Checkpoints', font='Helvetica 10 bold', bg="white")
chkContinue = Checkbutton(window, text='Continue Checkpoints', onvalue=1, offvalue=0, bg="white")

lblEpochCount = Label(frameEpochLabel, text='Epoch Count', font='Helvetica 10 bold', bg="white")
lblEpoch = Label(frameEpochLabel, text='Load Epoch', font='Helvetica 10 bold', bg="white")
txtEpochCount = Entry(frameEpoch, width = 10, bg="white")
txtEpoch = Entry(frameEpoch, width = 10, bg="white")

lblResize = Label(window, text='Resize', font='Helvetica 10 bold', bg="white")
drpResize = OptionMenu(frameResize, "resize_and_crop", "scale_width", "scale_width_and_crop", "none")
drpResize.configure(width=13, bg="white")
<<<<<<< Updated upstream
chkGpu = Checkbutton(frameResize, text='Use GPU', onvalue=1, offvalue=0, bg="white")


=======
chkGpu = Checkbutton(frameResize, text='Use GPU', onvalue=3, offvalue=2, bg="white")

btnCancel = Button(window, text='Cancel', font='Helvetica 10', width=12, height=1, command=cancel_convert, bg="white")
>>>>>>> Stashed changes

#Placing UI elements
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)

lblName.pack(side=TOP, anchor=W, pady=10, padx=10)
txtName.pack(side=TOP, anchor=W, padx=10)

frameDatasetLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblDataset.pack(side=LEFT, padx=(0,30))
lblCheckpoints.pack(side=LEFT)
frameDataset.pack(side = TOP, padx=10, anchor=W)
btnDataset.pack(side=LEFT, padx=(0,30))
btnCheckpoints.pack(side=LEFT)

frameLoadLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblLoadSize.pack(side=LEFT, padx=(0,75))
lblFineSize.pack(side=LEFT)
frameLoad.pack(side = TOP, padx=10, anchor=W)
txtLoadSize.pack(side=LEFT, padx=(0,88))
txtFineSize.pack(side=LEFT)

<<<<<<< Updated upstream
#lblContinue.pack(side=TOP, pady=10, padx=10, anchor=W)
chkContinue.pack(side=TOP, padx=10, pady=(15,0), anchor=W)
=======
>>>>>>> Stashed changes

frameEpochLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblEpochCount.pack(side=LEFT, padx=(0,59))
lblEpoch.pack(side=LEFT)
frameEpoch.pack(side = TOP, padx=10, anchor=W)
txtEpochCount.pack(side=LEFT, padx=(0,85))
txtEpoch.pack(side=LEFT)

<<<<<<< Updated upstream
=======
#lblContinue.pack(side=TOP, pady=10, padx=10, anchor=W)
chkContinue.pack(side=TOP, padx=10, pady=(15,0), anchor=W)

>>>>>>> Stashed changes
lblResize.pack(side=TOP, pady=10, padx=10, anchor=W)
frameResize.pack(side = TOP, padx=10, anchor=W)
drpResize.pack(side=LEFT, padx=(0,20))
chkGpu.pack(side=LEFT)

<<<<<<< Updated upstream
=======
progressbar.pack(side = TOP, pady=(40,20), padx=10, anchor=W)
btnCancel.pack(side = TOP, padx=(10,0), anchor=W)

>>>>>>> Stashed changes
outputBox.place(x=300, y=75)

sys.stdout = StdoutRedirector( outputBox )
print("INSTRUCTIONS GO HERE")
window.mainloop()


def destroyer():
    window.quit()
    window.destroy()
    window.exit()

window.protocol("WM_DELETE_WINDOW", destroyer)