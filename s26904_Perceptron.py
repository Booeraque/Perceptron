import numpy as np


class Perceptron:
    def __init__(self, inputNum, learning_Rate=0.01):
        self.weights = np.random.uniform(0.1, 0.5, inputNum)
        self.threshold = np.random.uniform(0.1, 0.5)
        self.learningRate = learning_Rate

    def predict(self, inputs):
        overall = np.dot(inputs, self.weights) + self.threshold
        return 1 if overall > 0 else 0

    def train(self, trainingData, testData, epochs):
        testAccuracy = 0
        for epoch in range(epochs):
            print("\nTraining epoch number - ", str(epoch + 1), "...")
            np.random.shuffle(trainingData)
            correctPredictions = 0
            for inputs, label in trainingData:
                predictedLabel = self.predict(inputs)
                if predictedLabel == label:
                    correctPredictions += 1
                error = label - predictedLabel
                if testAccuracy != 1.00:
                    self.weights += self.learningRate * error * inputs
                    self.threshold += self.learningRate * error

            testAccuracy = self.test(testData)
            print(f"Epoch {epoch + 1}, Test Accuracy: {testAccuracy:.2f}")

    def test(self, testData):
        correctPredictions = 0
        for inputs, label in testData:
            predictedLabel = self.predict(inputs)
            if predictedLabel == label:
                correctPredictions += 1
        return correctPredictions / len(testData)


def parse_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split(',')
            inputs = [float(x) if x.replace('.', '', 1).isdigit() else x for x in parts[:-1]]
            label = 1 if parts[-1] == '1' or parts[-1].lower() == 'iris-virginica' else 0
            data.append((np.array(inputs), label))
    return data


def init_perceptron():
    training_file_path = input("Enter path to file with training data: ")
    test_file_path = input("Enter path to file with test data: ")
    learning_rate = float(input("Enter learning rate: "))
    num_epochs = int(input("Enter number of epochs: "))

    training_data = parse_data(training_file_path)
    test_data = parse_data(test_file_path)
    input_num = len(training_data[0][0])

    perceptron = Perceptron(input_num, learning_rate)
    perceptron.train(training_data, test_data, num_epochs)
    return perceptron


def display_and_loop_menu(perceptron):
    while True:
        print("\n============ MENU ============")
        print("Choose one of the following:")
        print("1. Enter an observation and predict it's label")
        print("2. Display the trained weights and threshold")
        print("3. Exit the program")

        choice = input("\nEnter your choice: ")
        if choice == '1':
            observation = input("\nEnter new observation (comma-separated): ")
            observation = observation.strip().split(',')
            features = [float(x) if x.replace('.', '', 1).isdigit() else x for x in observation]
            prediction = perceptron.predict(features)
            print("Predicted class:", prediction)
            print("\n(Iris-virginica is considered as 1, Iris-versicolor is considered as 0)")
            continue
        if choice == '2':
            print("\nThe trained weights are:", perceptron.weights, "\nThe threshold is: ", perceptron.threshold)
        if choice == '3':
            print("\nExiting the program...")
            break


if __name__ == "__main__":
    perceptron_first = init_perceptron()
    display_and_loop_menu(perceptron_first)
