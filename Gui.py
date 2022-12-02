import tkinter as tk
from tkinter import *
from tkinter import ttk
import Model
from Model import preprocess
from Model import DeepModel
from Model import Activation
# from tkinter import messagebox

def act_changed (event):
    activations[0]:Activation(activation_function_cmb['values'].index(activation_function_cmb.get()))

def get_hidden_layers():
    hl = int(HiddenLayers_txt.get())
    print(hl)
    return hl


def get_num_neurons():
    s_non = NumOfNeurons_txt.get()
    non = [int(s) for s in s_non.split(',')]
    print(non)
    return non


def get_eta():
    eta = float(eta_txt.get())
    print(eta)
    return eta


def get_epochs():
    epochs = int(epoch_txt.get())
    print(epochs)
    return epochs

def func():
   data = preprocess(data = Model.dataset)

   per = DeepModel(data=data,
                     num_layers =get_hidden_layers() ,
                     num_of_neurons = get_num_neurons(),
                     eta=get_eta(),
                     epoch=get_epochs(),
                     activation_function = activations,
                     bias=bias_checkbox_var.get(),)

   per.train()

gui = Tk()
gui.title('Task_3 GUI')
gui.geometry('540x650+550+60')

Title = Label(gui, text="User Input", fg='black', font=("Times New Roman", 20))
Title.place(x=210, y=40)

HiddenLayers_label = Label(gui, text="Enter number of hidden layers", fg='black', font=("Times New Roman", 14))
HiddenLayers_label.place(x=50, y=130)
HiddenLayers_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
HiddenLayers_txt.place(x=310, y=130)

NumOfNeurons_label = Label(gui, text="Enter number of neurons in\n each hidden layer", fg='black',
                           font=("Times New Roman", 14))
NumOfNeurons_label.place(x=50, y=200)
NumOfNeurons_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
NumOfNeurons_txt.place(x=310, y=205)

eta_lbl = Label(gui, text="Enter learning rate", fg='black', font=("Times New Roman", 14))
eta_lbl.place(x=50, y=270)
eta_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
eta_txt.place(x=310, y=270)

epoch_lbl = Label(gui, text="Enter number of epochs", fg='black', font=("Times New Roman", 14))
epoch_lbl.place(x=50, y=330)
epoch_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
epoch_txt.place(x=310, y=330)

activation_function_lbl = Label(gui, text="Choose activation function", fg='black', font=("Times New Roman", 14))
activation_function_lbl.place(x=50, y=400)
selected_activation_function = tk.StringVar()
activation_function_cmb = ttk.Combobox(gui, textvariable=selected_activation_function, width=27)
activation_function_cmb['values'] :  list = [e.name for e in Activation]#("Tanh", "sigmoid")
activation_function_cmb.place(x=310, y=400)
activations : list = [None]*2
activation_function_cmb.bind('<<ComboboxSelected>>', act_changed)


bias_lbl = Label(gui, text="Add bias or not", fg='black', font=("Times New Roman", 14))
bias_lbl.place(x=50, y=470)

bias_checkbox_var = tk.StringVar()
bias_cb = ttk.Checkbutton(gui,
                          text='Bias',
                          variable=bias_checkbox_var,
                          onvalue=True,
                          offvalue=False)
bias_cb.place(x=360, y=475)

btn = Button(gui,
             text="Run",
             fg='black',
             width=15,
             font=("Times New Roman", 14),
             command = func)  # command=lambda: [data_entry_error(), func()])
# btn.bind('<Button-1>',get_hidden_layers)
# btn.bind('<Button-1>', get_num_neurons)
# btn.bind('<Button-1>',get_eta)
# btn.bind('<Button-1>',get_epochs)
btn.place(x=200, y=545)

gui.mainloop()