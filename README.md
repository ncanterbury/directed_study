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
   python source/app.py tweets/labelledtweets.txt arff_files/test.arff True
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
