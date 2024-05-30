import time
import random
import csv
from datetime import datetime, timedelta
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Crypto.Util import number
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from semiprime import generate_primes, generate_semiprimes
from model import DataProcessor, Encryptor, ModelTrainer, FunctionSelector

# Load semiprimes from file
def load_semiprimes(file_path):
    semiprimes = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                semiprimes.add(int(line.strip()))
    except Exception as e:
        print(f"Error loading semiprimes: {e}")
    return list(semiprimes)

# Generate RSA Keys
def generate_rsa_keys(semiprimes):
    p = random.choice(semiprimes)
    q = random.choice(semiprimes)
    while p == q:
        q = random.choice(semiprimes)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    d = number.inverse(e, phi)
    return RSA.construct((n, e, d))

# Encrypt message with RSA
def rsa_encrypt(message, public_key):
    try:
        cipher = PKCS1_OAEP.new(public_key)
        return cipher.encrypt(message.encode())
    except Exception as e:
        print(f"Error during RSA encryption: {e}")
        return None

# Decrypt message with RSA
def rsa_decrypt(ciphertext, private_key):
    try:
        cipher = PKCS1_OAEP.new(private_key)
        return cipher.decrypt(ciphertext).decode()
    except Exception as e:
        print(f"Error during RSA decryption: {e}")
        return None

# Inverse functions
def inverse_func1(encrypted_value):
    return int(encrypted_value)

def inverse_func2(encrypted_value):
    return int(encrypted_value)  # Inverse function for function 2 depends on the specific mapping

# Reverse dynamic encryption before RSA decryption
def reverse_dynamic_encryption(encrypted_value, selected_function):
    if selected_function == encrypt_with_func1:
        return inverse_func1(encrypted_value)
    elif selected_function == encrypt_with_func2:
        return inverse_func2(encrypted_value)

# Decrypt message using RSA
def rsa_decrypt_and_reverse(ciphertext, private_key, selected_function):
    decrypted_message = rsa_decrypt(ciphertext, private_key)
    if decrypted_message is None:
        print("Decryption failed.")
        return None
    return reverse_dynamic_encryption(int.from_bytes(decrypted_message.encode(), 'big'), selected_function)

# Main function
def main():
    # File paths
    dataset_path = r"C:\Users\linga\Downloads\hackathon demo\dataset.csv"
    model_path = r"C:\Users\linga\Downloads\hackathon demo\model.pkl"
    semiprimes_path = r"C:\Users\linga\Downloads\hackathon demo\semiprime.txt"
    
    # Load semiprimes
    semiprimes = load_semiprimes(semiprimes_path)
    
    # Generate RSA keys
    rsa_key = generate_rsa_keys(semiprimes)
    public_key = rsa_key.publickey()
    private_key = rsa_key

    # Message to encrypt
    message = "Hello, this is a secure message."
    print(f"Original Message: {message}")
    
    # Encrypt message using RSA
    ciphertext = rsa_encrypt(message, public_key)
    if ciphertext is None:
        print("Encryption failed.")
        return
    print(f"RSA Encrypted Message: {ciphertext}")

    # Initialize model and function selector
    # Load and process data
    try:
        data = DataProcessor.load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Train and save model
    trainer = ModelTrainer(data)
    try:
        trainer.train_model()
        trainer.save_model(model_path)
    except Exception as e:
        print(f"Error training or saving model: {e}")
        return
    
    # Load the trained model
    try:
        trainer.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Select and use encryption function
    selector = FunctionSelector(trainer.model)
    selected_function = selector.adaptive_function_selector()
    dynamically_encrypted_message = selected_function(int.from_bytes(ciphertext, 'big'))
    print(f"Dynamically Encrypted Message: {dynamically_encrypted_message}")

    # Simulate sending and receiving the encrypted message
    received_ciphertext = dynamically_encrypted_message  #
    # Reverse dynamic encryption before RSA decryption
    if selected_function == encrypt_with_func1:
        reversed_dynamic_encryption = inverse_func1(received_ciphertext)
    elif selected_function == encrypt_with_func2:
        reversed_dynamic_encryption = inverse_func2(received_ciphertext)

    # Convert reversed dynamic encryption to bytes for RSA decryption
    reversed_dynamic_encryption_bytes = reversed_dynamic_encryption.to_bytes((reversed_dynamic_encryption.bit_length() + 7) // 8, 'big')

    # Decrypt message using RSA and reverse dynamic encryption
    decrypted_message = rsa_decrypt_and_reverse(reversed_dynamic_encryption_bytes, private_key, selected_function)
    if decrypted_message is None:
        print("Decryption failed.")
        return
    print(f"Decrypted Message: {decrypted_message}")

if __name__ == "__main__":
    main()
