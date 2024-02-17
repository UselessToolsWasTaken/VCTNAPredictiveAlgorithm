import tkinter as tk
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import csv

# Under this part you will find teh algorithm proper. the code has been merged due to circular functions, just easier to
# to do it this way since the code itself is quite short

best_model = None

statsFile = "vct_stats.csv"  # I have a bunch of CSV's.
outcomeFile = "vct_outcome.csv"  # I'm not entirely sure what to do with them I think the outcome file gets appended with the predictions. I think.
historicalOutcomeFile = "vct_historical.csv"  # I think I'm going to stick to using one big dataset and then just try and shit out predictions


def writeDataCSV(Accuracy, Outcome_1, outcome_2, fileName):
    with open('C:\\Users\\evryt\\OneDrive\\Documents\\My Cheat Tables\\' + fileName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Accuracy, Outcome_1, outcome_2])


# Load the CSV data
data = pd.read_csv("C:\\Users\\evryt\\OneDrive\\Documents\\My Cheat Tables\\vct_data.csv",
                   header=None)


def trainingRun(learn_iterations):
    global best_model
    # Split the data into features (X) and target (y)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]

    scaler = MinMaxScaler()  # Minmaxing the lovely data
    X_scaled = scaler.fit_transform(X)  # Scaling said data

    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient
        'kernel': ['rbf', 'poly', 'sigmoid']  # Kernel type
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=10, scoring='accuracy')

    # Perform grid search
    grid_search.fit(X_scaled, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best cross-validation score
    best_score = grid_search.best_score_
    print("Best Cross-Validation Score:", best_score)

    best_params = grid_search.best_params_

    # Initialize SVM model with best hyperparameters
    best_model = SVC(**best_params, class_weight='balanced')

    # Number of iterations

    for i in range(learn_iterations):
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the SVM model
        best_model.fit(X_train_scaled, y_train)

        # Making predictions on the testing set
        y_pred = best_model.predict(X_test_scaled)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Iteration {i + 1}: Model Accuracy: {accuracy}, F1 Score: {f1}")
        rounded_accuracy = round(accuracy, 2)
        writeDataCSV(f"Iteration {i + 1}", rounded_accuracy, f1, "stats.csv")


def predict_match(team1_name, team2_name, team1_stats_param, team2_stats_param):
    global best_model
    global data

    if best_model is None:
        print("Model could not have been trained")
        return None
    team_stats_diff = team1_stats_param - team2_stats_param
    prediction = best_model.predict([team_stats_diff])  # Calculates the prediction based on the best/optimal
                                                        # test model. Above you can see that it tests for the best
                                                        # solution possible.
    print(prediction)
    if prediction == 1:
        return f"{team2_name} wins"
    else:
        return f"{team1_name} wins"


winner_text = ""


def calculate_stats(t1_name, t2_name):
    global winner_text
    team1_name = t1_name  # team1_name is now using the user input from the UI instead of directly plugged string values
    team2_name = t2_name  # same with team2_name, it's now using the second UI input from the user as string
    team1_stats = data[data[0] == team1_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 1
    team2_stats = data[data[0] == team2_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 2
    print("The winner is: ", predict_match(team1_name, team2_name, team1_stats, team2_stats))
    winner_text = str(predict_match(team1_name, team2_name, team1_stats, team2_stats))

# At this point we're building the UI, no more ML algorithm going on
# Below you will find the UI portion of the code
root = tk.Tk()
# Creating the window
root.title("VCT Predictor Algorithm")
root.geometry("250x400")
root.resizable(width=False, height=False)

# All required frames for this application
padding_top = tk.Frame(root)
padding_bottom = tk.Frame(root)

entry_frame = tk.Frame(root)
result_frame = tk.Frame(root)
generator_button_frame = tk.Frame(root)

# Team name entry frames, they will feed the algorithm the appropriate names
team1_label = tk.Label(entry_frame, text="Enter 1st Team Name")
team1_entry = tk.Entry(entry_frame, width=25)

team2_label = tk.Label(entry_frame, text="Enter 2nd Team Name")
team2_entry = tk.Entry(entry_frame, width=25)

iteration_label = tk.Label(result_frame, text="Set the amount of iterations to run")
iteration_count = tk.Entry(result_frame, width=25)

result_Label = tk.Label(result_frame, text="The winner is:")
result_dialog = tk.Label(result_frame, text="winner result here")

# The button proper, it is the heart of this code <3

padding_top.grid(row=0, column=0, sticky="nsew")  # Padding on top

# Entry frame set below the padding
entry_frame.grid(row=1, column=0, sticky="nsew", padx=50, pady=25)
# Team 1 Entry and label set within the entry frame
team1_label.grid(row=0, column=0, sticky="nsew")
team1_entry.grid(row=1, column=0, sticky="nsew")

# Team 2 Entry and Label set within the entry frame
team2_label.grid(row=3, column=0, sticky="nsew")
team2_entry.grid(row=4, column=0, sticky="nsew")

# Set result frame
result_frame.grid(row=2, column=0, sticky="nsew", padx=35, pady=15)
iteration_label.grid(row=0, column=0, sticky="nsew")
iteration_count.grid(row=1, column=0, sticky="nsew")
result_Label.grid(row=2, column=0, sticky="nsew")
result_dialog.grid(row=3, column=0, sticky="nsew", pady=15)

team_one_name = ""
team_two_name = ""
iteration_number = ""


def on_button_click():
    global team_one_name
    global team_two_name
    global iteration_number
    team_one_name = team1_entry.get()
    team_two_name = team2_entry.get()
    try:
        iteration_number = int(iteration_count.get())
    except ValueError:
        iteration_number = 100
    trainingRun(iteration_number)
    calculate_stats(team_one_name, team_two_name)
    result_dialog.config(text=winner_text)


generator_button = tk.Button(generator_button_frame, text="Generate Text", width=35, height=7, command=on_button_click)

generator_button_frame.grid(row=3, column=0, sticky="nsew", pady=25)
generator_button.grid(row=0, column=0, sticky="nsew")

root.mainloop()
