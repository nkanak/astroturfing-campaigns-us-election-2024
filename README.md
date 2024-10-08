# Detecting fake news campaigns for the 2024 United States presidential election

### Summary
* This repository contains code for detecting fake news and astroturfing campaigns using Twitter data. It is an improved version of the code written for the paper titled ["Detection of fake news campaigns using graph convolutional networks"](https://www.sciencedirect.com/science/article/pii/S2667096822000477).

### Dataset
The dataset can be generated using the [FakeNewNet repository](https://github.com/KaiDMML/FakeNewsNet). In a nutsell, the code requires information about the users, their followers and labels for the main tweets (e.g. fake news or real news). Labels for the retweets are not needed. Any other dataset that provides similar information can also be used.

#### Dataset location
Unzip all data into the `raw_data` folder. It should contain the following directory 
structure. 

```
politifact
user_followers
user_profiles
```


### Setup environment
Developed and tested in **Python3.9**

```sh
python3.9 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt

cd raw_data
curl -LO https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

### Run code
Take a look at the **run.sh** file to understand how the entire process works. Reading the main part of the related paper and the slides of the presentation can also be of help.

```sh
./run.sh
```


### Run code for ollama
* Run the jupyter notebook named `prompt_engineering_analysis.ipynb`