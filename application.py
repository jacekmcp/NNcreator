import os
from tkinter import *
from tkinter import font
import tkinter.filedialog
from tkinter import ttk
from NNGenerator import NNGenerator
import tkinter.messagebox


class App:
    
    def __init__(self, master):
        self.scene_GUI = []
        self.NNGenerator = NNGenerator()
        menubar = Menu(master)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Authors",command=self.start_scene)
        helpmenu.add_command(label="Exit",command=root.destroy)
        menubar.add_cascade(label="Program", menu=helpmenu)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import",command=self.uploaded_model)
        filemenu.add_separator()
        filemenu.add_command(label="Save", command = self.save_network)
        filemenu.add_separator()
        filemenu.add_command(label="Create New",command=self.new_neural_network_scene)
        menubar.add_cascade(label="Neural Network", menu=filemenu)
        dataset_menu = Menu(menubar,tearoff=0)
        dataset_menu.add_command(label="Uplaod new dataset",command=self.dataset_scene)
        menubar.add_cascade(label="Dataset",menu=dataset_menu)
        master.config(menu=menubar)
        
        self.start_scene()
        

    def start_scene(self):
        self.clear_GUI()
        first_label = Label(root, text="Neural Network\nCreator",font=("Helvetica", 60),
                    compound=CENTER,justify=CENTER,background="white")
        self.scene_GUI.append(first_label)
        first_label.place(x=300,y=400,anchor="center")
    
    def dataset_scene(self):
        self.clear_GUI()
        label = Label(root,text="Enter relative path to new dataset:")
        label.grid(row=0,column=0)
        dataset_path = Text(root, height=2, width=40)
        dataset_path.grid(row=2,column=0)
        self.scene_GUI.append(label)
        self.scene_GUI.append(dataset_path)
        buttonCommit = Button(root, text="Commit",command=lambda:self.load_dataset_action(self.scene_GUI[1].get("1.0",'end-1c')))
        buttonCommit.grid(row=3,column=0)
        self.scene_GUI.append(buttonCommit)

    def load_dataset_action(self,path):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, path + "/")
        print(filename)
        try:
            self.NNGenerator.read_dataset(filename)
            tkinter.messagebox.showinfo("Dataset status","Your new dataset has been imported")
        except:
            tkinter.messagebox.showinfo("ERROR","Wrong Input")

    def new_neural_network_scene(self):
        self.clear_GUI()
        head = Label(root,text="NEURAL NETWORK")
        head.grid(row=0,column=0)

        layers = Label(root,text="How many layers:")
        layers.grid(row=1,column=0)

        layers_number = Text(root, height=2, width=10)
        layers_number.grid(row=2,column=0)
        
        self.scene_GUI.append(layers_number)
        buttonCommit = Button(root, text="Commit",command=lambda:self.take_details_of_neural_network(self.scene_GUI[0].get("1.0",'end-1c')))
        buttonCommit.grid(row=3,column=0)

        self.scene_GUI.append(head)
        self.scene_GUI.append(layers)
        self.scene_GUI.append(buttonCommit)


    def uploaded_model(self):
        layers , functions = self.NNGenerator.read_model()
        self.clear_GUI()
        x = len(functions)
        for i in range(x):
            label = Label(root,text="how many neurons:")
            label.grid(row=4 + i*4,column=0)
            self.scene_GUI.append(label)
            default = IntVar(root, value=layers[i])
            # default.set(neurons[i])
            neurons = Spinbox(root,from_=1,to=1000, )
        
        #    self.scene_GUI.append(neurons)
            neurons.grid(row=5 + i*4,column=0)
            label2 = Label(root,text="activation function:")
            label2.grid(row=6 + i*4,column=0)
            self.scene_GUI.append(label2)
            default2 = StringVar(root)
            default2.set(functions[i])
            activation = Spinbox(root,values=("softmax","elu","selu","relu","sigmoid"))
            self.scene_GUI.append(activation)
            activation.grid(row=7 + i*4,column=0)
        submit_button = Button(root,text='Train',command= lambda:self.train(int(inputValue)))
        submit_button.grid(row=8 + x*4,column=0)
        self.scene_GUI.append(submit_button)
        label = Label(root,text=self.NNGenerator.get_nn_score())
        label.grid(row=9 + x * 4,column=0)
        self.scene_GUI.append(label)
    

    def take_details_of_neural_network(self,layers):
        if(int(layers) != 0):
            inputValue = layers
            x = int(inputValue)
            for i in range(int(inputValue)):
                label = Label(root,text="how many neurons:")
                label.grid(row=4 + i*4,column=0)
                self.scene_GUI.append(label)
                neurons = Spinbox(root,from_=1,to=1000)
                self.scene_GUI.append(neurons)
                neurons.grid(row=5 + i*4,column=0)

                label2 = Label(root,text="activation function:")
                label2.grid(row=6 + i*4,column=0)
                self.scene_GUI.append(label2)
                activation = Spinbox(root,values=("softmax","elu","selu","relu"))
                self.scene_GUI.append(activation)
                activation.grid(row=7 + i*4,column=0)
            submit_button = Button(root,text='Train',command= lambda:self.train(int(inputValue)))
            submit_button.grid(row=8 + x*4,column=0)
            self.scene_GUI.append(submit_button)
    
    def train(self,i):
        neurons = []
        fct = []

        index = 0
        c = 0
        for x in self.scene_GUI:
            if isinstance(x,Spinbox):
                if(c == 0):
                    neurons.append(int(x.get()))
                    c = 1
                else:
                    fct.append(x.get())
                    c = 0

        self.NNGenerator.create(neurons, fct)
        try:
            self.NNGenerator.train_nn()
        except:
            tkinter.messagebox.showinfo("ERROR","No dataset uploaded")
        label = Label(root,text=self.NNGenerator.get_nn_score())
        label.grid(row=9 + i * 4,column=0)
        self.scene_GUI.append(label)

    def import_neural_network(self):
        self.NNGenerator.read_model()
        tkinter.messagebox.showinfo("Neural network status","Your neural network has been imported")


    def clear_GUI(self):
        for gui_element in self.scene_GUI:
            gui_element.destroy()
        self.scene_GUI[:] = []

    def save_network(self):
        try:
            self.NNGenerator.save_model()
            tkinter.messagebox.showinfo("Neural network status","Your neural network has been saved")
        except:
            tkinter.messagebox.showinfo("ERROR","No neural network loaded")
        self.start_scene()

root = Tk()
style = ttk.Style()
style.theme_use('classic')
root.geometry("600x800")
root.title("Neural network")
root.resizable(0,0)

app = App(root)

root.mainloop()