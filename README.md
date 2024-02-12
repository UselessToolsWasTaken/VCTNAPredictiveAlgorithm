Current stats of the algorithm (According to metrics I get from in-code calculations)

Accuracy Score: 88%
F1 Score: 0.87

# Anger Management Program

This a Predictive Algorithm i wrote with the help of ChatGPT and a friend who is studying Data Science.

Here are some explanations to help you use this thing.

##To start: 

You need to either have PyCharm with Python 3.12 or later installed to use this. If you want this in the form of a .exe file, you need to compile it on your own as Python exe's throw false positives left and right.

## The Code Explained and how you can Modify it to fit your needs

The script is written in such a way that it analyzes a CSV file from top to bottom, left to right. So Row by row. The structure of the CSV is as follows, you kinda have to follow this for the algorithm to not shit itself.

NAMESPACE (Name to assign to the features) | FEATURE 1 | FEATURE 2 | FEATURE ... (You can add up to 10) | CLASS (Positive/Negative or 1/0)

Ex: SENTINELS | 1.19 | 125 | 3 | 5 | 1 

|Namespace|Feature 1|Feature 2|Feature 3|Class|
|--------:|--------:|--------:|--------:|-----|
|SENTINELS|     1.19|      125|        3|    1|

It's important to understand this data structure to use this algorithm.

### Important to remember

This is a comparing algorithm. So in ONE CSV file, you need TWO sets of data, For example: Stats for LOUD and SENTINELS for the past year. The algorithm then compares those two sets and decides the (Winner).

# Script explanation

```Python

statsFile = "vct_stats.csv"                         # I have a bunch of CSV's.
outcomeFile = "vct_outcome.csv"                     # I'm not entirely sure what to do with them I think the outcome file gets appended with the predictions. I think.
historicalOutcomeFile = "vct_historical.csv"        # I think I'm going to stick to using one big dataset and then just try and shit out predictions


def writeDataCSV(Accuracy, Outcome, fileName):
    with open('C:\\Users\\evryt\\OneDrive\\Documents\\My Cheat Tables\\' + fileName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Accuracy, Outcome])

```

The above is the first part of the code. The first three are file names I use(Or don't, not in this instance of the program). After that ```def writeDataCSV()``` is what I use to write the outcome into a CSV file to keep track of it later.

---

Next is the complicated part:

```Python

data = pd.read_csv("C:\\Users\\evryt\\OneDrive\\Documents\\My Cheat Tables\\vct_data.csv",
                   header=None)

# Split the data into features (X) and target (y)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

scaler = MinMaxScaler()  # Minmaxing the lovely data
X_scaled = scaler.fit_transform(X)  # Scaling said data

```

There are some comments but let me explain. 

```data = ...``` is used to define the path for the CSV you want to use. I've included the one I use in the files. You can create your own based off of that as you go. 

Next. We specify the Data range with ```X = data.iloc[:, 1:-1]```. This means the following ':' means all rows. After that we say '1:-1', this piece of code means 1st column to the last one, without including the LAST one... so, minus ONE. After that we define where the class is located. This one's simple enough ```y = data.iloc[:, -1]``` If you look above you should already understand this. ':' is for all rows and the '-1' means LAST COLUMN (whatever that column is), unlike 1:-1(which is a range of columns)

The MinMaxing! I'll get to that later but we're using the SVM model to predict stuff here. Without giving you a lecture, it performs well if the range difference between data is not big. ex: 1 and 1 000 000. It likes 0 and 1 basically.

```scaler = MinMaxScaler()  # Minmaxing the lovely data``` Here we initialise the Scaler, after that we plug in X ```X_scaled = scaler.fit_transform(X)  # Scaling said data``` to it which as a range of numbers. It then takes those numbers and scales them down to a number between 0 and 1.

1 = The HIGHEST number in the feature set
0 = Lowest number in the Feature set 
BUT ONLY FOR THAT SPECIFIC FEATURE (Column)

---

For the next part.I will not go into much detail as it would take hours to write a text about this.

```Python

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

```

ChatGPT: Grid search involves defining a grid of hyperparameters and exhaustively searching through all possible combinations to find the optimal set of hyperparameters.

Basically, this whole section of code is running the model through multiple different combinations of itself. Since you can select different kernels, scaling, weights of data etc it's good to do that. In here``` grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=10, scoring='accuracy')```
We have a parameter called ```cv=10```. This little dude is the magic. This is the Cross-Validation number, basically it runs the data against itself X amount of times, in this case 10, also called `folds`. After that, it averages all the folds into a number which is a % in accuracy.

It's basically  helping your ML make a good determination. Important note: Higher isn't better, neither is lower. 

In my examples I had LOUD vs SENTINELS. when the CV was set to 5, the winner was LOUD however on ANY OTHER Amount of folds, 3, 4, 6, 10, 15 even 50(Don't do that it takes literally an hour to compute), SENTINELS won. What does that mean? You can infer that CV = 5 favors LOUDS data but to know which one is correct, you need to apply some of your own brain to it.

After all is done it selects the best parameters to fit the data and give you t he most accurate prediction possible.

---

```Python

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
    print(f"Iteration {i + 1}: Model Accuracy: {accuracy}")
    rounded_accuracy = round(accuracy, 2)
    writeDataCSV(f"Iteration {i + 1}", rounded_accuracy, statsFile)

```

This is the learning part. I iterate it over a certain amount. I'm not going to go over every line for this, just know that it takes a test sample and learning sample with the split define by ```test_size=0.5``` so it's 50% in my code. It then shits out the accuracy of the whole system in the console.
Remember not to iterate a few thousand times, unless you have an insanely big dataset. The more times you iterate the higher the chances of Noise being introduce into the system, ultimately breaking it.

---

```Python

def predict_match(team1_stats_param, team2_stats_param):
    global best_model
    global data

    team_stats_diff = team1_stats_param - team2_stats_param
    prediction = best_model.predict([team_stats_diff])
    print(prediction)
    if prediction == 1:
        return f"{team2_name} wins"
    else:
        return f"{team1_name} wins"

# Yes, I run 10 predictions. I want to see if it'll shit out some other stuff I didn't expect
for i in range(num_iterations):
    # Example usage:
    team1_name = "LOUD"  # Replace with the name of the first team
    team2_name = "100T"  # Replace with the name of the second team
    team1_stats = data[data[0] == team1_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 1
    team2_stats = data[data[0] == team2_name].iloc[:, 1:-1].mean()  # Extract and average the stats for Team 2

    print("The winner is: ", predict_match(team1_stats, team2_stats))
    rounded_accuracy = round(accuracy, 2)

    writeDataCSV(rounded_accuracy, predict_match(team1_stats, team2_stats),
                 outcomeFile)  # Appends outcome lines with Accuracy to CSV
```

These two pieces kinda go hand in hand. Basically ```predict_match``` is the actual comparison. It takes the NAME that you give it, compares it with Column 1 to see if it exists, if yes, then it does the magic and shits out 1 or 0. If prediction is 1, team 2 wins, if not, team 1 wins.

In the above case it's LOUD vs 100T but in ```team_name``` 1 and 2 you just put a string from the CSV.


This is it folks. The CSV will be available here as well.
