

def mix(a, secret):
    return a ^ secret

def prune(secret):
    temp = secret % 16777216
    return temp

def next_secret(secret):
    temp = secret * 64
    secret = mix(temp, secret)
    secret = prune(secret)
    
    temp = secret // 32
    secret = mix(temp, secret)
    secret = prune(secret)
    
    temp = secret * 2048
    secret = mix(temp, secret)
    secret = prune(secret)
    
    return secret

def parse_input():
    with open('inputs/day22_input.txt', 'r') as file:
        initial_secrets = []
        for line in file:
            initial_secrets.append(int(line.strip()))
    return initial_secrets


def sum_of_nth_secret(n, buyers_initial_secrets):
    buyers_secrets = {}
    for secret in buyers_initial_secrets:
        original_secret = secret
        for i in range(n):
            secret = next_secret(secret)
        buyers_secrets[original_secret] = secret

    return sum(buyers_secrets.values())

initial_secrets = parse_input()
# initial_secrets = [1, 10, 100, 2024]
print(sum_of_nth_secret(2000, initial_secrets))
