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
        

opt = trainOptions.initOpt()
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
window.title('Chickpea roots GANILLA User Interface - TRAIN') 
window.geometry("700x575")
window.resizable(False, False)
window.configure(bg="white")




outputBox = scrolledtext.ScrolledText(window,
                                      padx = 5,
                                      pady = 5,
                                      wrap = tk.WORD,  
                                      width = 60,  
                                      height = 29,  
                                      font = ("Arial", 
                                              10)) 

outputBox.place(x=251, y=75) 

sys.stdout = StdoutRedirector( outputBox )
print("Please select the folder containing the model and the folder containing the dataset. Followed by the target results directory if desired.")
window.mainloop()


def destroyer():
    window.quit()
    window.destroy()
    window.exit()

window.protocol("WM_DELETE_WINDOW", destroyer)


