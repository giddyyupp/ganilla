import os
from tkinter import *
from tkinter import filedialog

def getDirectory():
    path = filedialog.askdirectory(initialdir = "/",
                                   title = "Select a Folder")

    lbl4.configure(text="Folder Opened: "+path)

def selectGAN(choice):
    if choice == 1:
        lbl5.configure(text="DUALGAN SELECTED")
    elif choice == 2:
        lbl5.configure(text="GANILLA SELECTED")
    else:
        lbl5.configure(text="CYCLEGAN SELECTED")

def openViewWin():
    
    newWindow = Toplevel(window)

    newWindow.title("New Window") 
  
    # sets the geometry of toplevel 
    newWindow.geometry("200x200") 
  
    # A Label widget to show in toplevel 
    Label(newWindow,  text ="This is a new window").pack() 

           
                                                                                                         
window = Tk()  
window.title('File Explorer') 
window.geometry("500x500") 

#setting varaibles
dropdownop=StringVar(window)
dropdownop.set("1")

#creating all the UI objects (lables, buttons, inputs)
title=Label(window, text='ROOT ENHANCE', font='Helvetica 20 bold', fg="white", bg="black", anchor='nw', width=40, height=1)
subtitle=Label(window, text='CONVERT POOR QUALITY CAPTURES TO HIGH QUALITY CAPTURES', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)
lbl1=Label(window, text='GAN TYPE', font='Helvetica 10 bold')
lbl2=Label(window, text='EPOCH NO. (14 RECOMMENDED)', font='Helvetica 10 bold')
lbl3=Label(window, text='IMAGE CONVERSION:', font='Helvetica 10 bold')
lbl4=Label(window, text='IMAGE DIRECTORY:', font='Helvetica 10 bold')
lbl5=Label(window, text='{GAN SELECTED}')
dropdown=OptionMenu(window, dropdownop, "1", "2", "3", "4","5","6","7","8","10","11","12","13","14")
btn1 = Button(window, text='DUALGAN', font='Helvetica 10', width=10, height=1, command= lambda: selectGAN(1))
btn2 = Button(window, text='GANILLA', font='Helvetica 10', width=10, height=1, command= lambda: selectGAN(2))
btn3 = Button(window, text='CYCLEGAN', font='Helvetica 10', width=10, height=1, command= lambda: selectGAN(3))
btn4 = Button(window, text='BROWSE', font='Helvetica 10', width=10, height=1, command=getDirectory)
btn5 = Button(window, text='VIEW', font='Helvetica 10', width=10, height=1, command = openViewWin)
btnInfo = Button(window, text='INFORMATION', bg="black", fg="white", font='Helvetica 8 bold', width=10, height=1)
footer=Label(window, text='CREATED BY THE ROOT ENHANCE TEAM', font='Helvetica 10', fg="white", bg="black", anchor='nw', width=85, height=1)


#placing all the UI objects on screen
title.place(x=0, y=0)
subtitle.place(x=0, y=30)
lbl1.place(x=270, y=80)
lbl2.place(x=195, y=155)
lbl3.place(x=225, y=225)
lbl4.place(x=230, y=295)
lbl5.place(x=220, y=55)
dropdown.place(x=285, y=175)
btn1.place(x=165, y=100)
btn2.place(x=265, y=100)
btn3.place(x=365, y=100)
btn4.place(x=200, y=245)
btn5.place(x=320, y=245)
btnInfo.place(x=500, y=15)
footer.place(x=0, y=380)
   
# Let the window wait for any events 
window.mainloop() 


#def start():
    #os.system('python test.py --dataroot datasets/intense_roots --name chickpea_model --model test --gpu_id -1 --epoch 13 --loadSize 1024 --fineSize 1024')
    
