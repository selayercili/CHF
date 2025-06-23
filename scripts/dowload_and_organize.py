"""
TO-DO
1) Only download if the data hasnt been downloaded yet
2) Do a train test split if it hasnt been organized yet

Things to consider:
- Make sure that the code is written within functions
- Maintain good coding practices
- Produce some plots for the data 
  - how is it structured 
  - what is the distribution of the different pieces of data 

- Have the plotting functions be inside of src/
- Only having this function make calls to files and functions within the src/ folder
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data import download_dataset, split_data 

def main():


    #downloads the dataset from kaggle
    print("Starting data download...")
    download_success = download_dataset()
    
    if download_success:
        print("\n=== Data Splitting ===")
        try:
            train_path, test_path = split_data()
            print(f"\nTrain: {train_path}\nTest: {test_path}")
        except Exception as e:
            print(f"\n× Error: {str(e)}")
    else:
        print("Aborting due to download failure")


    

if __name__ == "__main__":
    main()

