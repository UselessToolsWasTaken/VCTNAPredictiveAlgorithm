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

