from itertools import accumulate
import itertools
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

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

def price_from_secret(secret):
    return int(str(secret)[-1:])

def get_prices_and_changes(initial_secret, n=2000):
    secrets = list(accumulate(range(n), lambda s, _: next_secret(s), initial=initial_secret))
    prices = [price_from_secret(s) for s in secrets]
    diffs = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    diffs.insert(0, None)
    seq_payouts = {}
    # 5th number, when we have 4 diffs, is the first time we can trigger a buy
    for i in range(5, n):
        # sequences are 4 diffs longs
        key = tuple(diffs[i-3:i+1])
        if key not in seq_payouts:
            seq_payouts[key] = prices[i]
    assert len(diffs) == len(prices)
    # print(len(diffs), len(prices), len(seq_payouts))
    return prices, diffs, seq_payouts


def get_prices_and_changes_df(initial_secret, n=2000):
    # Generate all secrets first
    secrets = [initial_secret]
    for _ in range(n-1):
        secrets.append(next_secret(secrets[-1]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'secret': secrets,
        'price': [price_from_secret(s) for s in secrets],
    })
    
    # Calculate price changes
    df['price_change'] = df['price'].diff()
    
    # Create sequences (only where we have 4 consecutive changes)
    sequences = []
    for i in range(len(df)):
        if i < 4:  # First 4 rows can't have a complete sequence
            sequences.append(None)
        else:
            # Get the last 4 changes
            seq = tuple(df['price_change'].iloc[i-3:i+1].tolist())
            sequences.append(seq)
    
    df['sequence'] = sequences
    
    return df


def sum_of_nth_secret(n, buyers_initial_secrets):
    buyers_secrets = {}
    for secret in buyers_initial_secrets:
        original_secret = secret
        for i in range(n):
            secret = next_secret(secret)
        buyers_secrets[original_secret] = secret

    return sum(buyers_secrets.values())

def find_sublist(main_list, sublist):
    '''returns index of sublist in main_list, or None if not found'''
    if not sublist or len(sublist) > len(main_list):
        return None
        
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i:i + len(sublist)] == sublist:
            return i
            
    return None


initial_secrets = parse_input()
# initial_secrets = initial_secrets[:10]
# initial_secrets = [1, 10, 100, 2024]
# initial_secrets = [1, 2, 3, 2024]
# initial_secrets = [123]
# print(sum_of_nth_secret(2000, initial_secrets))


# df = get_prices_and_changes_df(initial_secrets[0])
# print("First few rows of the DataFrame:")
# print(df.head(10))


# for secret in initial_secrets:
#     a,b,c = get_prices_and_changes(secret)
#     print(find_sublist(b, (1, -3, 5, 1)))
#     print(c[(1, -3, 5, 1)])


seq_payouts = defaultdict(int)
num_payouts = defaultdict(int)
for initial_secret in tqdm(initial_secrets, desc='Initial secret'):
    a, b, c = get_prices_and_changes(initial_secret)
    for seq, payout in c.items():
        if seq == (1, -3, 5, 1):
            print(f"Sequence {seq} pays out {payout} bananas on secret {initial_secret}")
        seq_payouts[seq] += payout
        num_payouts[seq] += 1

max_payout = max(seq_payouts.values())
print(f"Max payout: {max_payout} using sequence {max(seq_payouts, key=seq_payouts.get)}")
print(f"Number of monkeys that triggered a sale with the top sequence: {num_payouts[max(seq_payouts, key=seq_payouts.get)]}")

print(f"Top 10 sequences and their payouts:")
for seq, payout in sorted(seq_payouts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Sequence {seq} pays out {payout} bananas")

