# Midas Touchpad Project

The code loads or trains two models, then asks to select a model and a random sample from the test data it covers.
It then passes the data point through the model, makes a prediction and outputs the predicted character on the highlighted typespace.

**To test: **
1. run the script
2. pick a sample
3. quickly switch to a window that has a typespace
4. select that typespace
5. watch the magic happen.

For the Milestone Submission:
- Download dependencies (into a venv)
- Run "python3 classifier.py"

We have included the .pkl files for the math and Greek classifier models; they will be loaded and allow for a database image to be predicted upon. 
If you would like to see the workflow for the model(s) being built and trained/tested, delete the .pkl files and run "python classifier.py" to execute this.

We came across an issue in the torch library where 'mps' wasn't recognized as a device. The directory of the trouble is provided, and removing the ( , device=mps ) part of the return of the trouble function fixed the issue.
