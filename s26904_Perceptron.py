import numpy as np


class Perceptron:
    def __init__(self, inputNum, learning_Rate=0.01):
        self.weights = np.random.uniform(0.1, 0.5, inputNum)
        self.threshold = np.random.uniform(0.1, 0.5)
        self.learningRate = learning_Rate
        self.label_map = []  # Initialize an empty dictionary for labels

    def parse_data(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                parts = line.split(',')
                inputs = [float(x) if x.replace('.', '', 1).isdigit() else x for x in parts[:-1]]
                label = parts[-1]  # Assume the label is always the last element in each line
                if label not in self.label_map:
                    self.label_map.append(label)
                data.append((np.array(inputs), label))
        return data

    def predict(self, inputs):
        overall = np.dot(inputs, self.weights) + self.threshold
        prediction = 1 if overall > 0 else 0
        return prediction

    def train(self, trainingData, testData, epochs):
        testAccuracy = 0
        for epoch in range(epochs):
            np.random.shuffle(trainingData)
            print("\nTraining epoch number - ", str(epoch + 1), "...")
            for inputs, label in trainingData:
                predicted_int = self.predict(inputs)
                label_int = 1 if self.label_map[0] == label else 0
                error = label_int - predicted_int
                if testAccuracy != 1.00:
                    self.weights += self.learningRate * error * inputs
                    self.threshold += self.learningRate * error

            testAccuracy = self.test(testData)
            print(f"Epoch {epoch + 1}, Test Accuracy: {testAccuracy:.2f}")

    def test(self, testData):
        correctPredictions = 0
        for inputs, label in testData:
            predictedLabel = self.predict(inputs)
            label_int = 1 if self.label_map[0] == label else 0
            if predictedLabel == label_int:
                correctPredictions += 1
        return correctPredictions / len(testData)


def compute_input_num(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            inputs = [float(x) if x.replace('.', '', 1).isdigit() else x for x in parts[:-1]]
            return len(inputs)


def init_perceptron():
    # training_file_path = "./DataForPerceptron/iris_perceptron/training.txt"
    # test_file_path = "./DataForPerceptron/iris_perceptron/test.txt"
    training_file_path = input("Enter path to file with training data: ")
    test_file_path = input("Enter path to file with test data: ")
    learning_rate = float(input("Enter learning rate: "))
    num_epochs = int(input("Enter number of epochs: "))
    input_num = compute_input_num(test_file_path)

    perceptron = Perceptron(input_num, learning_rate)
    training_data = perceptron.parse_data(training_file_path)
    test_data = perceptron.parse_data(test_file_path)
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
            prediction_int = perceptron.predict(features)
            prediction = perceptron.label_map[prediction_int]
            print("Predicted class:", prediction)
            continue
        if choice == '2':
            print("\nThe trained weights are:", perceptron.weights, "\nThe threshold is: ", perceptron.threshold)
        if choice == '3':
            print("\nExiting the program...")
            break


if __name__ == "__main__":
    perceptron_first = init_perceptron()
    display_and_loop_menu(perceptron_first)
