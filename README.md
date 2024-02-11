# Anger Management Program

This a Predictive Algorithm i wrote with the help of ChatGPT and a friend who is studying Data Science.

Here are some explanations to help you use this thing.

##To start: 

You need to either have PyCharm with Python 3.12 or later installed to use this. If you want this in the form of a .exe file, you need to compile it on your own as Python exe's throw false positives left and right.

## The Code Explained and how you can Modify it to fit your needs

The script is written in such a way that it analyzes a CSV file from top to bottom, left to right. So Row by row. The structure of the CSV is as follows, you kinda have to follow this for the algorithm to not shit itself.

NAMESPACE (Name to assign to the features) | FEATURE 1 | FEATURE 2 | FEATURE ... (You can add up to 10) | CLASS (Positive/Negative or 1/0)

Ex: SENTINELS | 1.19 | 125 | 3 | 5 | 1 

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

Next. We specify the Data range with ```X = data.iloc[:, 1:-1]```. This means the following ':' means all rows. After that we say '1:-1', this piece of code means 1st column to the last one, without including the LAST one... so, minus ONE

after that we define where the class is located. This one's simple enough ```y = data.iloc[:, -1]``` If you look above you should already understand this. ':' is for all rows and the '-1' means LAST COLUMN (whatever that column is), unlike 1:-1(which is a range of columns)
