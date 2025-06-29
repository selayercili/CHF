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
from src.plotting import create_eda_plots 

def main():
    print("=== Data Download ===")
    download_success = download_dataset()
    
    if download_success:
        print("\n=== Data Splitting ===")
        train_path, test_path = split_data()
        
        print("\n=== Data Visualization ===")
        create_eda_plots() 
        
        print("\n Pipeline complete! Ready for modeling.")
    else:
        print("\n Aborting due to download failure")



    

if __name__ == "__main__":
    main()

