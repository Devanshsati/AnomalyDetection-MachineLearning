import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder , StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    df['index'] = pd.to_datetime(df['timestamp'])
    df.index = df['index']
    del df['index']
    del df['timestamp']
    csvreader = csv.reader(df, delimiter=",")
    for row in csvreader:
        if "sensor_15" in row[0]:
            df['sensor_15'].nunique() 
            df.drop(['sensor_15'], axis=1, inplace = True)
    df['machine_status'].unique()#'NORMAL', 'BROKEN', 'RECOVERING'
    df['machine_status'].value_counts()
    le = LabelEncoder()
    df['machine_status'] = le.fit_transform(df['machine_status'])
    df['machine_status'].value_counts()
    # 1 - normal # 2 - recovering # 0 - broken
    return df

def preprocess_data_IsolationForest(df):
    def split_ttdata(data,test_ratio):
        np.random.seed(42)
        shuffled = np.random.permutation(len(data))
        test_set_size = int(len(data)*test_ratio)
        test_indices = shuffled[:test_set_size]
        train_indices = shuffled[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]
    train_data, test_data = split_ttdata(df,0.3)
    test_set_X, test_set_Y = split_ttdata(test_data,0.5)
    dt=test_set_X
    dy=test_set_Y
    return dt,dy

def train_logistic_regression(X_train,y_train):
    logit = LogisticRegression(max_iter=1000)
    mod = logit.fit(X_train,y_train)
    return mod

def train_isolation_forest(dt,arr):
    model = IsolationForest(n_estimators=100,max_samples="auto",contamination=float(0.2),max_features=0.5)
    return model.fit(dt[arr])

def predict_logistic_regression(model, X_test):
    return model.predict(X_test)

def predict_isolation_forest(model, arr, dy, dt):
    dt["anomailed_scores"] = model.decision_function(dt[arr])
    dt["anomaly"] = model.predict(dt[arr])
    pride = model.predict(dt[arr])
    pride2 = model.predict(dy[arr])
    return pride,pride2

def logistic_class_report(lrpredict,y_test):
    return classification_report(lrpredict, y_test, zero_division=1) 

def isolation_classification(dt):
    return pd.crosstab(dt["machine_status"],dt["anomaly"], rownames=['True'], colnames=['Predicted'], margins=True)

def iso_class_report(isolation_predictions1, isolation_predictions2):
    return classification_report(isolation_predictions1, isolation_predictions2, zero_division=1) 

def get_shape(df):
    df1 = df.copy()
    df1 = df[(df1.machine_status ==1)]
    df2 = df.copy()
    df2 = df[(df2.machine_status ==2)] 
    df3 = df.copy()
    df3 = df[(df3.machine_status ==0)]
    return df1.shape,df2.shape,df3.shape

def showgraph(df):
    plt.figure(figsize=(5,3)) 
    custom_palette = ["#334CFF", "#33FFC7", "#FF5733"]
    ax = sns.countplot(y=df['machine_status'], hue=df['machine_status'], palette=custom_palette, dodge=False)
    ax.set_facecolor("#DAF7A6") 
    plt.xlabel("Count") 
    plt.ylabel("Machine Status")
    plt.tight_layout() 
    return plt

def anomalyscore(df,filename):
    ascore=df
    m1 = IsolationForest(n_estimators=100,max_samples="auto",contamination=float(0.2),max_features=0.5)
    match filename:
        case "sensor2.csv": arr=["machine_status","sensor_00"]
        case "sensor3.csv": arr=["machine_status","sensor_22"]
        case "sensor1.csv": arr=["machine_status","sensor_00"]
    m1.fit(ascore[arr])
    ascore["anomailed_scores"] = m1.decision_function(ascore[arr])
    ascore["anomaly"] = m1.predict(ascore[arr])
    return ascore

def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        return file_path
    else:
        messagebox.showerror("Error", "No file selected.")
        return None

def display_output(output_text1, output_text2, output_text3, output_text4, describe, lr_classification, isolation_classify, graph, iso_classification_str,anomaly_score):
    output_window = Toplevel(root)
    output_window.geometry("1350x675")  # Set the geometry of the window
    output_window.config(bg="#FFF4D5")

    topFrame = Frame(output_window, height=350, width=1350, bg="#FFF4D5")
    topFrame.place(x=0,y=0)

    topLeftFrame = Frame(topFrame, bg="#FFF4D5") #height=350, width=310,
    topLeftFrame.place(x=0,y=0)

    topCenterFrame = Frame(topFrame, bg="#FFF4D5") # height=350, width=320,
    topCenterFrame.place(x=455,y=0)

    topRightFrame = Frame(topFrame,height=350, width=410, padx=10, pady=10, bg="#FFF4D5") # height=350, width=410,
    topRightFrame.place(x=990,y=0)

    bottomFrame = Frame(output_window, height=325, width=135, bg="#FFF4D5")
    bottomFrame.place(x=0,y=340)
    output_window.title("Model Results")

    Label(topRightFrame, text="Logistic Regression Analysis",width=30, bg="#FFF4D5", pady=5, fg="#000FFF", justify="center",font=('Roboto 14 bold')).pack(fill=X)
    Label(topRightFrame, text="Detailed Report in Text Box",width=30, bg="#FFF4D5", pady=5, justify="center",font=('Roboto 10')).pack(fill=X)
    Label(topRightFrame, text=output_text1,justify="center",width=30, bg="#FFF4D5", pady=5, font=('Roboto 10')).pack(fill=X)
    Label(topRightFrame, text="Isolation Forest Analysis",width=30, bg="#FFF4D5", pady=5, fg="#000FFF",justify="center",font=('Roboto 14 bold')).pack(fill=X)
    Label(topRightFrame, text="Detailed Report in Text Box",width=30, bg="#FFF4D5", pady=5, justify="center",font=('Roboto 10')).pack(fill=X)
    Label(topRightFrame, text=output_text2,justify="center",width=30, bg="#FFF4D5", pady=5, font=('Roboto 10')).pack(fill=X)
    Label(topRightFrame, text="Results Of Test Data",fg="#000FFF",width=30, bg="#FFF4D5", pady=5, justify="center",font=('Roboto 14 bold')).pack(fill=X) 
    Label(topRightFrame, text=output_text3,justify="center",width=30, bg="#FFF4D5", pady=5,  font=('Roboto 12')).pack(fill=X)
    Label(topRightFrame, text=output_text4,fg="#FF2700",width=30, bg="#FFF4D5", pady=1,  justify="center",font=('Roboto 12')).pack(fill=X)

    # Text Widgets for Logistic Regression and Isolation Forest Analysis
    text_widget1 = Text(bottomFrame, height=20, width=167, bg="#F3FFEC")
    text_widget1.insert(END, "\n\nDescribe Data\n\n")
    text_widget1.insert(END, describe)
    text_widget1.insert(END, "\n\n\nAnomaly Report\n\n")
    text_widget1.insert(END, anomaly_score)
    text_widget1.pack(side=TOP, padx=10, pady=10)

    text_widget2 = Text(topLeftFrame, height=22, width=55, bg="#F3FFEC")
    text_widget2.insert(END, "Isolation Forest Cross Tabulation\n\n")
    text_widget2.insert(END, isolation_classify)
    text_widget2.insert(END, "\n\nLogistic Regression Classification Report\n\n")
    text_widget2.insert(END, lr_classification)
    text_widget2.pack(side=TOP, padx=10, pady=10)

    # Graph for Machine Status Count
    graph_frame = Frame(topCenterFrame)
    graph_frame.pack(side=TOP, padx=10, pady=10)
    graph_label = Label(graph_frame, text="Machine Status Count", font=('Roboto', 12, 'bold'))
    graph_label.pack()
    canvas = FigureCanvasTkAgg(graph, master=graph_frame)
    canvas.get_tk_widget().pack()

    output_window.mainloop()

def run_model():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    import ntpath
    filename = ntpath.basename(file_path)
    if file_path:
        df = preprocess_data(file_path) 
        # Split data for Logistic Regression model
        df = df.ffill()
        X = df.drop(['machine_status'], axis=1)
        y = df['machine_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression model And Predict Score
        describe = df.describe().apply(lambda x: x.map('{:,.2f}'.format))
        anomaly_score=anomalyscore(df,filename)
        logit_model = train_logistic_regression(X_train_scaled, y_train)
        lr_predictions = predict_logistic_regression(logit_model, X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        lr_classification_str = logistic_class_report(lr_predictions, y_test)
        graph = showgraph(df)

        # Train Isolation Forest model
        s1=["sensor_00","sensor_01","sensor_02","sensor_03","sensor_04","sensor_05","sensor_06","sensor_07","sensor_08","sensor_09","sensor_10","sensor_11","sensor_12","sensor_13","sensor_14","sensor_16","sensor_17","sensor_18","sensor_19","sensor_20","sensor_21","sensor_22","sensor_23","sensor_24","sensor_25","sensor_26","sensor_27","sensor_28","sensor_29","sensor_30","sensor_31","sensor_32","sensor_33","sensor_34","sensor_35","sensor_36","sensor_37","sensor_38","sensor_39","sensor_40","sensor_41","sensor_42","sensor_43","sensor_44","sensor_45","sensor_46","sensor_47","sensor_48","sensor_49","sensor_50","sensor_51"]
        s2=["sensor_22","sensor_23","sensor_24","sensor_25","sensor_26","sensor_27","sensor_28","sensor_29","sensor_30","sensor_31","sensor_32","sensor_33","sensor_34","sensor_35","sensor_36","sensor_37","sensor_38","sensor_39","sensor_40","sensor_41","sensor_42","sensor_43","sensor_44","sensor_45","sensor_46","sensor_47","sensor_48","sensor_49","sensor_50","sensor_51"]
        s3=["sensor_00","sensor_01","sensor_02","sensor_03","sensor_04","sensor_05","sensor_06","sensor_07","sensor_08","sensor_09","sensor_10","sensor_11","sensor_12","sensor_13","sensor_14","sensor_16","sensor_17","sensor_18","sensor_19","sensor_20","sensor_21"]
        match filename:
            case "sensor2.csv": arr=s3
            case "sensor3.csv": arr=s2
            case "sensor1.csv": arr=s1
        dt, dy = preprocess_data_IsolationForest(df)
        isolation_model = train_isolation_forest(dt, arr)
        isolation_predictions1, isolation_predictions2 = predict_isolation_forest(isolation_model, arr, dy, dt)
        isolation_accuracy = accuracy_score(isolation_predictions1, isolation_predictions2)
        isolation_classification_str = str(isolation_classification(dt))
        iso_classification_str = iso_class_report(isolation_predictions1, isolation_predictions2)
        lrshape1, lrshape2, lrshape3 = get_shape(df)

        # Show the graph for Machine Status Count
        graph = showgraph(df).gcf()

        # Concatenate model results into a single string
        output_text1 = (f"Logistic Regression Accuracy: {round(lr_accuracy*100,3)}")
        output_text2 = (f"Isolation Forest Accuracy: {round(isolation_accuracy*100,3)}")
        output_text3 = (f"Normal Data (1) : {lrshape1[0]}\nRecovering Data (2) : {lrshape2[0]}")
        output_text4 = (f"Broken or Anomaly Detected Data (0) : {lrshape3[0]}")

        # Display the results in a separate window using labels and graph
        display_output(output_text1, output_text2, output_text3, output_text4, describe, lr_classification_str, isolation_classification_str, graph, iso_classification_str,anomaly_score)


# Create GUI
root = Tk()
root.geometry('400x150')
root.title("ML Model GUI DevanshSati")

frame = Frame(root)
frame.pack(pady=20)

button = Button(frame, text="Load Data and Run Model", command=run_model, padx=10, pady=5, bg='#00B203', fg='black', relief=RAISED)
button.pack(side=LEFT, padx=10)

quit_button = Button(frame, text="Quit", command=quit, padx=10, pady=5, bg='#C70039', fg='white', relief=RAISED)
quit_button.pack(side=LEFT, padx=10)

root.config(bg='#f0f0f0')

title_label = Label(root, text="Anomaly Detection in IoT Sensor\n Machine Learning Model GUI", font=('Arial', 14))
title_label.pack()

desc_label = Label(root, text="Click 'Load Data and Run Model' to start.", font=('Arial', 10))
desc_label.pack()

root.mainloop()
