import csv
import random
from datetime import datetime, timedelta
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DataProcessor:
    @staticmethod
    def load_dataset(file_path):
        data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                number, alphabet_original, alphabet_squared = row
                data.append((int(number), alphabet_original, alphabet_squared))
        return data

    @staticmethod
    def generate_features(number):
        return [
            number,  # original number
            len(str(number)),  # number of digits
            number % 2,  # even or odd
            sum(int(digit) for digit in str(number)),  # sum of digits
        ]

class Encryptor:
    @staticmethod
    def func1(number):
        return str(number)

    @staticmethod
    def func2(number):
        return str(number ** 2)

    @staticmethod
    def number_to_alphabet(number):
        mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', '6': 'F', '7': 'G', '8': 'H', '9': 'I', '0': 'Z'}
        return ''.join(mapping[digit] for digit in number)

    def encrypt_with_func1(self, number):
        transformed = self.func1(number)
        return self.number_to_alphabet(transformed)

    def encrypt_with_func2(self, number):
        transformed = self.func2(number)
        return self.number_to_alphabet(transformed)

class ModelTrainer:
    def __init__(self, data):
        self.features = []
        self.labels = []
        self.process_data(data)
        self.model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)

    def process_data(self, data):
        for number, alphabet_original, alphabet_squared in data:
            feature_vector = DataProcessor.generate_features(number)
            self.features.append(feature_vector)
            self.labels.append(1 if random.random() > 0.5 else 2)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model accuracy: {accuracy * 100:.2f}%')

    def save_model(self, file_path):
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

class FunctionSelector:
    def __init__(self, model):
        self.model = model
        self.encryptor = Encryptor()
        self.last_switch_time = datetime.now()

    def select_best_function(self, number):
        feature_vector = DataProcessor.generate_features(number)
        prediction = self.model.predict([feature_vector])[0]
        return self.encryptor.encrypt_with_func1 if prediction == 1 else self.encryptor.encrypt_with_func2

    def adaptive_function_selector(self):
        current_time = datetime.now()
        if current_time - self.last_switch_time >= timedelta(hours=2):
            self.last_switch_time = current_time
            return random.choice([self.encryptor.encrypt_with_func1, self.encryptor.encrypt_with_func2])
        else:
            return self.select_best_function(random.randint(0, 999))

    def log_function_usage(self, func_name, performance, usage_log):
        usage_log.append((func_name, performance))

if __name__ == "__main__":
    dataset_path = r"C:\Users\linga\Downloads\hackathon demo\dataset.csv"
    model_path = 'model.pkl'
    
    # Load and process data
    data = DataProcessor.load_dataset(dataset_path)
    
    # Train and save model
    trainer = ModelTrainer(data)
    trainer.train_model()
    trainer.save_model(model_path)
    
    # Load the trained model
    trainer.load_model(model_path)
    
    # Select and use encryption function
    selector = FunctionSelector(trainer.model)
    number = random.randint(0, 999)
    selected_function = selector.adaptive_function_selector()
    encrypted_value = selected_function(number)
    print(f'Number: {number}, Encrypted Value: {encrypted_value}')
    
    # Logging
    usage_log = []
    selector.log_function_usage(selected_function.__name__, random.random(), usage_log)
    for log in usage_log:
        print(f'Function: {log[0]}, Performance: {log[1]}')
