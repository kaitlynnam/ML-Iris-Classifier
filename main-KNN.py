from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

def main():
    """Iris Classifier using k-Nearest Neighbors (KNN).

    This script loads the Iris dataset, trains a KNN model, and lets the user
    either test model accuracy or predict a flower's species based on custom input.
    """
    
    # Load the Iris dataset
    iris = datasets.load_iris()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

    # Initialize the KNN model (k=1)
    k=1
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Main interaction loop: lets the user choose actions repeatedly until quitting
    while True:
        
        # Ask the user what they want to do
        path = input(
            "\nWhat do you want to do?\n\n"
            "  (a) Test and tune the model's accuracy\n"
            "  (b) Make a prediction with a custom iris\n"
            "  (q) Quit\n\n"
            "Enter your choice (a/b/q): "
        ).strip().lower()

        
        if path == "a":

            # Allow users to tune the model
            try:
                while True:

                    # Ask the user for a new neighbor number
                    k = int(input("\nChoose number of neighbors for training (1 to 112): "))

                    if not 1 <= k <= 112:
                        print("\nPlease enter a number between 1 and 112.")
                        continue

                    # Reinitialize the KNN model with new neighbor number
                    knn = KNeighborsClassifier(n_neighbors=k)

                    # Retrain the model
                    knn.fit(X_train, y_train)

                    # Shows user the model's accuracy
                    accuracy = knn.score(X_test, y_test) * 100
                    print(f"The model is {accuracy:.2f}% accurate.")

                    # Ask user if they want to retrain the model
                    retrain = input("Do you want to train the model again? (y/n): ").strip().lower()
                    if retrain != 'y':
                        break

            except ValueError:
                print("\nInvalid input.")
            except Exception as e:
                print(f"\nUnexpected error: {e}")

            

        elif path == "b": 
            try:

                # Get flower measurements from user input
                sepal_length = float(input("\nEnter sepal length (cm): "))
                sepal_width = float(input("Enter sepal width (cm): "))
                petal_length = float(input("Enter petal length (cm): "))
                petal_width = float(input("Enter petal width (cm): "))

                # Make a prediction
                new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                prediction = knn.predict(new_flower)

                species = iris.target_names # ['setosa', 'versicolor', 'virginica']
                print(f"\nYour iris is a: {species[prediction[0]].capitalize()}!\n")

            # Handle invalid user input (e.g., letters instead of numbers)
            except ValueError:
                print("\nInvalid input. Please enter numbers only.")
            except Exception as e:
                print(f"\nUnexpected error: {e}")

        elif path == "q":
            # Exit the program gracefully
            print("\nThanks for using the Iris Classifier!\n")
            break

        # Handle invalid user input (inputs aside from 'a' or 'b')
        else:
            print("Invalid input. Please enter 'a' or 'b'.")

        # Ask if the user wants to continue or exit after each operation
        cont = input("Do you want to try again? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nThanks for using the Iris Classifier!\n")
            break
        

if __name__ == "__main__":
    main()