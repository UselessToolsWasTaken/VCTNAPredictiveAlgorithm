import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import csv
import main as mn

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

    team_stats_diff = team1_stats_param - team2_stats_param
    prediction = best_model.predict([team_stats_diff])
    print(prediction)
    if prediction == 1:
        return f"{team2_name} wins"
    else:
        return f"{team1_name} wins"


winner_text = ""


def calculate_stats(t1_name, t2_name):
    global winner_text
    team1_name = t1_name  # Replace with the name of the first team
    team2_name = t2_name  # Replace with the name of the second team
    team1_stats = data[data[0] == team1_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 1
    team2_stats = data[data[0] == team2_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 2
    print("The winner is: ", predict_match(team1_name, team2_name, team1_stats, team2_stats))
    winner_text = str(predict_match(team1_name, team2_name, team1_stats, team2_stats))
