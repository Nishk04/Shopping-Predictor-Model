import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data - write: python shopping.py shopping.csv")

    # Load data from spreadsheet and split into train and test sets
    datafile = sys.argv[1]
    print(datafile)
    evidence, labels = load_data(datafile)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # First train model with 40% of dataset
    model = train_model(X_train, y_train)
    # Test model
    predictions = model.predict(X_test)
    sensitivity, specificity, actual = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Total True Out of All: {100 * actual:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        # Skips the header row
        next(reader)
        for row in reader:
            # Accessing the labels 
            if row[-1] == "TRUE":
                labels.append(1)
            else: 
                labels.append(0)
            # Add all data to evidence
            evidence.append([
                int(row[0]),
                float(row[1]),
                int(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                # Gives first occurence of the value to search for
                months.index(row[10]),
                int(row[11]),
                int(row[12]),
                int(row[13]),
                int(row[14]),
                1 if row[15] == "Returning_Visitor" else 0,
                1 if row[16] == "TRUE" else 0,
            ])
    return evidence, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    ''' 
    Increasing the n_neighbors results in a weird behavior. For example, making it for example 100
    results in a very low true positive rate while almost perfect true negative rate. Why does it 
    fit better for the negative values only, instead of fitting well for both? 
    '''
    neighborModel = KNeighborsClassifier(n_neighbors=1)
    neighborModel.fit(evidence, labels)
    return neighborModel

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity). 

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_pos = 0
    total_pos = 0
    true_neg = 0
    total_neg = 0
    # Checks if the models predictions were correct
    for index in range(len(labels)):
        if labels[index] == 1:
            total_pos +=1
            if predictions[index] == 1:
                true_pos += 1
        else:
            total_neg += 1
            if predictions[index] == 0:
                true_neg += 1
    # Finds the accuracy of our model
    sensitivity = true_pos/total_pos
    specificity = true_neg/total_neg
    actual = (true_neg + true_pos)/(total_pos + total_neg)

    return sensitivity, specificity, actual

if __name__ == "__main__":
    main()
