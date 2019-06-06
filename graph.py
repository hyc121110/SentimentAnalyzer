import tkinter as tk
import tkinter.messagebox as msg
import pickle
from nltk.corpus import stopwords
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


    

def run(input_s):
    total_out = ""
    stop_words=set(stopwords.words('english'))

    model1 = pickle.load(open("models/model_nb.pk1", "rb"))
    model2 = pickle.load(open("models/model_lsvc.pk1", "rb"))
    model3 = pickle.load(open("models/model_lr.pk1", "rb"))
    tfidf_vec = pickle.load(open("models/tfidf_vec.pk1", "rb"))

    # Try with user input
    #print("Please enter a review of a company: ", end='')
    sentence = input_s
    sentence = [word for word in sentence.split() if word.lower() not in stop_words]
    sentence = " ".join(sentence)
    sent = tfidf_vec.transform([sentence])

    pred1 = model1.predict_proba(sent)
    total_out += ("\nNaive Bayes model is " + '{0:.2f}'.format(pred1[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred1[0][1]*100) + "% sure that this sentence is positive.\n")
    pred2 = model2.predict_proba(sent)
    total_out += ("Linear SVC model is " + '{0:.2f}'.format(pred2[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred2[0][1]*100) + "% sure that this sentence is positive.\n")
    pred3 = model3.predict_proba(sent)
    total_out += ("Logistic Regression model is " + '{0:.2f}'.format(pred3[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred3[0][1]*100) + "% sure that this sentence is positive.\n")

    # average the score
    score = (pred1[0][0] + pred2[0][0] + pred3[0][0]) / 3
    total_out +=("Averaging the three models: the new model is " + '{0:.2f}'.format(score*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format((1-score)*100) + "% sure that this sentence is positive.\n")

    if score > 0.8:
        total_out += ("\nThis is a negative review!")
    elif score < 0.2:
        total_out += ("\nThis is a positive review!")
    else:
        total_out += ("\nThis is a neutral review!")    
    return total_out, score    

class create_display(tk.Tk):
    def __init__(self, tasks=None):
        super().__init__()

        if not tasks:
            self.tasks = []
        else:
            self.tasks = tasks

        self.tasks_c = tk.Canvas(self)

        self.tasks_f = tk.Frame(self.tasks_c)
        self.text_frame = tk.Frame(self)

        self.scrollbar = tk.Scrollbar(self.tasks_c, orient="vertical", command=self.tasks_c.yview)

        self.tasks_c.configure(yscrollcommand=self.scrollbar.set)

        self.title("review of a company")
        self.geometry("1000x1200")

        self.task_create = tk.Text(self.text_frame, height=4, bg="white", fg="black")

        self.tasks_c.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas_frame = self.tasks_c.create_window((0, 0), window=self.tasks_f, anchor="n")

        self.task_create.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.task_create.focus_set()

        todo1 = tk.Label(self.tasks_f, text="--- Please enter a review of a company : ---", bg="lightgrey", fg="black", pady=10)

        self.tasks.append(todo1)

        for task in self.tasks:
            task.pack(side=tk.TOP, fill=tk.X)

        self.bind("<Return>", self.add_task)
        self.bind("<Configure>", self.on_frame_configure)
        self.bind_all("<MouseWheel>", self.mouse_scroll)
        self.bind_all("<Button-4>", self.mouse_scroll)
        self.bind_all("<Button-5>", self.mouse_scroll)
        self.tasks_c.bind("<Configure>", self.task_width)

        self.colour_schemes = [{"bg": "lightcyan", "fg": "dark slate gray"}, {"bg": "grey", "fg": "white"}]

    def add_task(self, event=None):
        def on_closing():
            root.destroy()
        task_text = self.task_create.get(1.0,tk.END).strip()

        if len(task_text) > 0:
            new_task = tk.Label(self.tasks_f, text=task_text, pady=10)

            self.set_task_colour(len(self.tasks), new_task)
            new_task.pack(side=tk.TOP, fill=tk.X)

            self.tasks.append(new_task)

        task_text_new, score_n = run(task_text)
        if len(task_text_new) > 0:
            new_task = tk.Label(self.tasks_f, text=task_text_new, pady=10)

            self.set_task_colour(len(self.tasks), new_task)
            new_task.pack(side=tk.TOP, fill=tk.X)

            self.tasks.append(new_task)
        self.task_create.delete(1.0, tk.END)
        if len(task_text_new) > 0:
            root = tk.Tk()
            x1 = score_n
            x2 = 1 - score_n
 
            figure2 = Figure(figsize=(5,4), dpi=100) 
            subplot2 = figure2.add_subplot(111) 
            labels2 = 'negative', 'positive'
            pieSizes = [float(x1),float(x2)]
            explode2 = (0, 0.1)  
            subplot2.pie(pieSizes, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True, startangle=90) 
            subplot2.axis('equal')  
            pie2 = FigureCanvasTkAgg(figure2, root) 
            pie2.get_tk_widget().pack()
            root.protocol("WM_DELETE_WINDOW", on_closing)
        
        

    
    def recolour_tasks(self):
        for index, task in enumerate(self.tasks):
            self.set_task_colour(index, task)

    def set_task_colour(self, position, task):
        _, task_style_choice = divmod(position, 2)

        my_scheme_choice = self.colour_schemes[task_style_choice]

        task.configure(bg=my_scheme_choice["bg"])
        task.configure(fg=my_scheme_choice["fg"])

    def on_frame_configure(self, event=None):
        self.tasks_c.configure(scrollregion=self.tasks_c.bbox("all"))

    def task_width(self, event):
        canvas_width = event.width
        self.tasks_c.itemconfig(self.canvas_frame, width = canvas_width)

    def mouse_scroll(self, event):
        if event.delta:
            self.tasks_c.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            if event.num == 5:
                move = 1
            else:
                move = -1
            self.tasks_c.yview_scroll(move, "units")

if __name__ == "__main__":
    display = create_display()
    display.mainloop()