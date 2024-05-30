import time
import itertools

# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# Function to generate primes using the sieving technique
def generate_primes(time_limit):
    primes = [3]  # Start with the first known odd prime
    start_time = time.time()

    # Initialize set A
    set_A = set()

    # Starting with n = 0
    n = 0

    while True:
        for p in primes:
            # Check if n fits the exemption rule for the prime p
            if (n - ((p - 3) // 2)) % p == 0:
                break
        else:
            set_A.add(n)
            candidate_prime = 2 * n + 3
            if is_prime(candidate_prime):
                primes.append(candidate_prime)
                print(f"Found prime: {candidate_prime}")

        # Check time limit
        if time.time() - start_time >= time_limit:
            break

        # Increment n
        n += 1
        if n % 1000 == 0:
            print(f"Current n: {n}, Time elapsed: {time.time() - start_time:.2f}s")
    
    print(f"Total primes generated: {len(primes)}")
    return primes

# Function to generate semiprimes by multiplying primes
def generate_semiprimes(primes):
    semiprimes = set()
    start_time = time.time()

    # Generate semiprimes by taking products of two elements from primes
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            semiprime = primes[i] * primes[j]
            semiprimes.add(semiprime)

    print(f"Time taken to generate semiprimes: {time.time() - start_time:.2f}s")
    return semiprimes

# Function to write semiprimes to file and display total count
def write_semiprimes(semiprimes):
    try:
        with open("semiprime.txt", "w") as file:
            for semiprime in sorted(semiprimes):
                file.write(str(semiprime) + "\n")
        print(f"Total semiprimes generated: {len(semiprimes)}")
        print("Semiprimes written to 'semiprime.txt'")
    except Exception as e:
        print("Error writing semiprimes to file:", e)

# Main function
def main():
    # Define time limit for prime generation (in seconds)
    prime_generation_time_limit = 10
    
    # Generate primes using the sieving technique
    primes = generate_primes(prime_generation_time_limit)
    
    # Generate semiprimes by multiplying generated primes
    semiprimes = generate_semiprimes(primes)
    
    # Write semiprimes to file and display total count
    write_semiprimes(semiprimes)

if __name__ == "__main__":
    main()
