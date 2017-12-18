# My Project's README

This is the code for my directed study, fall 17. This initial code works to take in a text file of labelled tweets, and outputs an arff\_file with the parameters specified in config.

Install/Setup:

1. Create virtual environment for dependencies/libraries. I would recommend using
   anaconda.
```
   command line instruction: conda create --name myenv python=2.7
   source activate myenv 
```
2. Install dependencies: 
```
   pip install pyyaml
   conda install pandas 
   pip install preprocessor 
   pip install nltk
   pip install sklearn
   conda install scipy
   pip install twython
   pip install liac-arff 
   pip install scipy
   pip install numpy
   conda install matplotlib
   pip install ggplot
   pip install ipython
   sudo apt-get install default-jre
```
3. To Run:
```
   cd ~/
   git clone https://nathancanterbury@bitbucket.org/nathancanterbury/directedstudyfall17.git
   cd ~/directedstudyfall17/
   python source/app.py tweets/labelledtweets.txt arff_files/test.arff
```

Acknowledgements:
(To be refined later, I just want to keep track as I go)
Original Labelled Tweet source
Carnegie Mellon Twiiter POS Tagger

Positive & Negative Opinion Words:
;   Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
;       Proceedings of the ACM SIGKDD International Conference on Knowledge
;       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
;       Washington, USA,
;   Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing
;       and Comparing Opinions on the Web." Proceedings of the 14th
;       International World Wide Web conference (WWW-2005), May 10-14,
;       2005, Chiba, Japan.

Deleted line 768 in labelled tweets because there was no tweet attached, only sentiment rating. Might have to parse
for troublesome data like this and remove. Should still be present in original labelled tweets

There is a blank line in original training set, line 1533, and a sentiment tag under it. Delete line 1533 and the
sentiment tag on line 1534. It is under the tweet that starts with '9 days'.


These are instructions for running the entire project. 
 From this point these are being written from the point of having completed 
 weka svm and scilearn svm, and having exported weka svm model. We now begin
 dealing with the 40 million tweets. 

 First remove all tweets that mention both obama and romney 
 We will use the awk command for this. Enter the following command in the terminal.

awk '! (/romney/ && /obama/)' election_data.txt > new_election_data.txt

Then there is an issue where there are occassionally lines in the election data where there 
are random spaces on multiple lines, and lines of text that do not seem to belong to any tweets.
We will use the following awk command to remove all lines that do not begin with a timestamp.

awk '(/2012/)' new_election_data.txt > tmpfile.txt && mv tmpfile.txt new_election_data.txt 

Now we can run the arff file generator on the new_election_data.txt file. 
Navigate to the arff_generator directory 
cd directedstudyfall17 cd arff_generator

Three arguments are passed in on the command line,
they are [tweet file] [arff file to be written to] [is this labelled data]
You can enter the following command to run.

python source/app.py ~/directedstudyfall17/new_election_data.txt arff_files/election_data.arff False

This will write the data from the tweets in the new_election_data file to the election_data.arff file.
Then open the weka explorer. Click the explorer tab. Click on the preprocess tab, and click on open file.
Open the arff file you just generated. Click the Edit button, right click on the word "sentiment" in the
top left corner. Select "attribute as class" on the popup menu. Click OK.

Then we need to remove all attributes but the ones we choose earlier through attribute selection.
Click the button that says all, deselect "negative_words", "mention_romney", "vader_polarity", 
"T" and "sentiment". Click remove. Then click save in the top right, and overwrite the previous
election_data.arff file with the new modified one you just made. 

Then click on classify. In the result list, right click and select load model. Select the SVM model
you previously saved. Under test option select supplied test set, and click Set. Select the arff 
file you just saved. Click more options and deselect all options selected. Click choose next to 
Output predictions, and select CSV. This is where the prediction data will be written out to. Click 
on the white bar that says CSV, and change the value of "outputFile" to be the csv file the results 
should be written to. 




















