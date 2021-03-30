from numpy.core.fromnumeric import var
from models import create_model
from data import CreateDataLoader
import sys, threading
import tkinter as tk
from tkinter import *
from tkinter import tix
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from options import train_options 
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

trainOptions = TrainOptions()

#added () for test
opt = trainOptions.initOpt()

# hard-code some parameters for test
opt.name = ""
opt.save_latest_freq = 1000
opt.save_epoch_freq = 1
opt.dataset_mode = 'unaligned'
opt.print_freq = 10

def getDataroot():
    dataroot = filedialog.askdirectory(initialdir = "..", title = "Select Dataroot")
    opt.dataroot = dataroot
    print('\nDataroot set as ', opt.dataroot)

    
def getCheckpoints():
    checkpoints = filedialog.askdirectory(initialdir ="./", title = "Select Folder Containing Model")
    opt.checkpoints_dir = checkpoints
    print('\nModel directory set as ', opt.checkpoints_dir)
    
def set_continue():
    if continue_train.get() == 1:
        opt.continue_train = True
    else:
        opt.continue_train = False

def cancel_train():
    print("Cancelling Training...")
    global running
    running = False
    
def start_train():
    global running
    running = True
    thread = threading.Thread(target=train)
    print("Starting Training...")
    thread.start()

def train():
    try:
        if opt.continue_train == 1:
            opt.epoch = int(txtEpoch.get())            
        opt.name = txtName.get()
        opt.loadSize = int(txtLoadSize.get())
        opt.fineSize = int(txtFineSize.get())
        opt.epoch_count = int(txtEpochCount.get())
        
    except ValueError as ve:
        print("\nPlease ensure all text boxes have only numbers")
        raise
    except Exception as e:
        print(e)
        raise
    if __name__ == '__main__':
        try:
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)
            print('#training images = %d' % dataset_size)

            model = create_model(opt)
            model.setup(opt)
            visualizer = Visualizer(opt)
            total_steps = 0
            
            print(trainOptions.return_options(opt))

            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                for i, data in enumerate(dataset):
                    global running
                    if running == False:
                        raise KeyboardInterrupt                   
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
window.geometry("815x575")
window.resizable(False, False)
window.configure(bg="white")


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
frameStart = Frame()
frameStart.configure(bg="white")

#UI elements
outputBox = scrolledtext.ScrolledText(window,
                                      padx = 5,
                                      pady = 5,
                                      wrap = tk.WORD,  
                                      width = 69,  
                                      height = 29,  
                                      font = ("Arial", 
                                              10)) 

lblTitle = Label(window, text='GANILLA UI - TRAIN', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
lblSub = Label(window, text='TRAIN A MODEL, WITH A DATASET, FOR USE IN TESTING', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
lblFoot = Label(window, text='CREATED BY GM, ND & CD', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)

lblName = Label(window, text='Model Name', font='Helvetica 10 bold', bg="white")
txtName = Entry(window, width = 20, bg="white") #textVariable = modelName
continue_train = tk.IntVar()
chkContinue = Checkbutton(window, text='Continue Checkpoints', variable=continue_train, onvalue=1, offvalue=0, bg="white", command=set_continue)
txtName.insert(END,"GANILLA")

lblDataset = Label(frameDatasetLabel, text='Dataset Directory', font='Helvetica 10 bold', bg="white")
lblCheckpoints = Label(frameDatasetLabel, text='Checkpoints Directory', font='Helvetica 10 bold', bg="white")
btnDataset = Button(frameDataset, text='Select Dataset', font='Helvetica 10', width=14, bg="white", command=getDataroot)
btnCheckpoints = Button(frameDataset, text='Select Checkpoints', font='Helvetica 10', width=14, bg="white",command=getCheckpoints)



lblEpochCount = Label(frameEpochLabel, text='Epoch Count', font='Helvetica 10 bold', bg="white")
lblEpoch = Label(frameEpochLabel, text='Load Epoch', font='Helvetica 10 bold', bg="white")
txtEpochCount = Entry(frameEpoch, width = 10, bg="white")

txtEpoch = Entry(frameEpoch, width = 10, bg="white")
#txtEpoch.insert(END,"1")
txtEpochCount.insert(END,"1")

lblLoadSize = Label(frameLoadLabel, text='Load Size', font='Helvetica 10 bold', bg="white")
lblFineSize = Label(frameLoadLabel, text='Fine Size', font='Helvetica 10 bold', bg="white")
txtLoadSize = Entry(frameLoad, width = 10, bg="white")
txtLoadSize.insert(END,"286")
txtFineSize = Entry(frameLoad, width = 10, bg="white")
txtFineSize.insert(END,"256")

#lblContinue = Label(window, text='Continue Checkpoints', font='Helvetica 10 bold', bg="white")



lblResize = Label(window, text='Resize', font='Helvetica 10 bold', bg="white")

drpResizeOp = StringVar(window)
drpResizeOp.set("resize_and_crop")
drpResize = OptionMenu(frameResize, drpResizeOp, "resize_and_crop", "scale_width", "scale_width_and_crop", "none")

drpResize.configure(width=20, bg="white", anchor="w")
chkGpu = Checkbutton(frameResize, text='Use GPU', onvalue=3, offvalue=2, bg="white")

btnStart = Button(frameStart, text='Start', font='Helvetica 10', width=14, height=1, command=start_train, bg="white")
btnCancel = Button(frameStart, text='Cancel', font='Helvetica 10', width=14, height=1, bg="white", command=cancel_train)

#Placing UI elements
lblTitle.pack(fill=X)
lblSub.pack(fill=X)
lblFoot.pack(fill=X, side=BOTTOM)

lblName.pack(side=TOP, anchor=W, pady=10, padx=10)
txtName.pack(side=TOP, anchor=W, padx=10)
chkContinue.pack(side=TOP, padx=10, pady=(15,0), anchor=W)

frameDatasetLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblDataset.pack(side=LEFT, padx=(0,30))
lblCheckpoints.pack(side=LEFT)
frameDataset.pack(side = TOP, padx=10, anchor=W)
btnDataset.pack(side=LEFT, padx=(0,30))
btnCheckpoints.pack(side=LEFT)


frameEpochLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblEpochCount.pack(side=LEFT, padx=(0,59))
lblEpoch.pack(side=LEFT)
frameEpoch.pack(side = TOP, padx=10, anchor=W)
txtEpochCount.pack(side=LEFT, padx=(0,85))
txtEpoch.pack(side=LEFT)

frameLoadLabel.pack(side = TOP, pady=10, padx=10, anchor=W)
lblLoadSize.pack(side=LEFT, padx=(0,75))
lblFineSize.pack(side=LEFT)
frameLoad.pack(side = TOP, padx=10, anchor=W)
txtLoadSize.pack(side=LEFT, padx=(0,88))
txtFineSize.pack(side=LEFT)

#lblContinue.pack(side=TOP, pady=10, padx=10, anchor=W)


#lblContinue.pack(side=TOP, pady=10, padx=10, anchor=W)
chkContinue.pack(side=TOP, padx=10, pady=(15,0), anchor=W)

lblResize.pack(side=TOP, pady=10, padx=10, anchor=W)
frameResize.pack(side = TOP, padx=10, anchor=W)
drpResize.pack(side=LEFT, padx=(0,20))
chkGpu.pack(side=LEFT)

frameStart.pack(side = TOP, pady=20, padx=10, anchor=W)
btnStart.pack(side = LEFT, anchor=W)
btnCancel.pack(side = LEFT, padx=(30,0), anchor=W)

outputBox.place(x=300, y=75)

sys.stdout = StdoutRedirector( outputBox )
print("INSTRUCTIONS:")
print("  1.  Enter Model Name (new or loaded)\n\tthis folder will be created in the checkpoints directory if none exists.\n")
print("  2.  Set the root Dataset Directory\n\tcontaining folders name 'trainA', and 'trainB' \n")
print("  3. Set root Checkpoints directory\n\tcontaining or to contain folder named after model\n")
print("  4. Set if training is to load model and continue, or train new model\n")
print("  5. Enter Epoch Count to be (a) epoch to continue training from or (b) total epochs if new model\n")
print("  6. If Continue checkpoints is set, enter epoch to be loaded in Load Epoch\n")
print("  7. Enter loadSize (size of images loaded into memory) and fineSize (resized images for training)\n")
print("  8. Choose Image Pre-Processing method\n")
print("  9. Choose whether to utilize GPU processing (CUDA must be installed)\n")
print("  10. Start train")

window.mainloop()


def destroyer():
    window.quit()
    window.destroy()
    window.exit()

window.protocol("WM_DELETE_WINDOW", destroyer)