import sys
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

YEAR = 2024
SESSION_COOKIE = os.getenv('AOC_SESSION')

def download_input(day: int) -> None:
    """Download the input file for the specified day"""
    
    if not SESSION_COOKIE:
        print("Error: AOC_SESSION environment variable not set")
        print("Please create a .env file with your session cookie")
        sys.exit(1)
    
    # Create inputs directory if it doesn't exist
    Path("inputs").mkdir(exist_ok=True)
    
    # Construct the URL and filename
    url = f"https://adventofcode.com/{YEAR}/day/{day}/input"
    filename = f"inputs/day{day}_input.txt"
    
    # Set up the session cookie
    cookies = {'session': SESSION_COOKIE}
    
    try:
        # Download the input
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(response.text.rstrip('\n'))
        
        print(f"Successfully downloaded input for day {day} to {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading input for day {day}: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python download.py <day>")
        print("Example: python download.py 3")
        sys.exit(1)
    
    try:
        day = int(sys.argv[1])
        if day < 1 or day > 25:
            raise ValueError("Day must be between 1 and 25")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    download_input(day)

if __name__ == "__main__":
    main()