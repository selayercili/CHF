import kagglehub

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

# Download latest version
path = kagglehub.dataset_download("saurabhshahane/predicting-heat-flux")
