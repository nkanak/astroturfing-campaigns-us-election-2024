#!/usr/bin/env python

#
# Preprocess the FakeNews dataset. Create a folder which contains one file 
# per tweet. Inside each tweet we have its list of retweets.
#

import argparse
import os
import logging
import utils
import random
import copy
import logging
import re
import random
import networkx as nx

import models
import json
from datetime import datetime 

USER_PROFILES_PATH = "../raw_data/user_profiles"
USER_FOLLOWERS_PATH = "../raw_data/user_followers"
MISSING_USER_PROFILES_PATH = "missing_user_profiles"

def _lookup_RT(text):
    match = re.search(r'RT\s@((\w){1,15}):', text)
    if match: 
        return match.group(1)
    return None


def _find_retweet_source(retweet, previous_retweets):
    """Given a retweet and all previous retweers estimate from which 
    retweet it originated.
    """
    # Check if there is only one retweet previously. If so, then assign to it.
    if len(previous_retweets) == 1:
        return previous_retweets[0]

    user = retweet.user
    rt_username = _lookup_RT(retweet.text)

    # Find a tweet from the RT user
    if rt_username is not None:
        candidates = []
        for rt in previous_retweets:
            if rt.user.screenname == rt_username: 
                candidates.append(rt)

        if len(candidates) != 0:
            return max(candidates, key= lambda k: rt.user.popularity)

    # Check if we follow some of the previous users that retweeted
    candidates = []
    for rt in previous_retweets:
        if user.id in rt.user.followers:
            candidates.append(rt)

    if len(candidates) != 0:
        return max(candidates, key= lambda k: rt.user.popularity)

    # Assign to most popular based on popularity
    weights = [rt.user.popularity for rt in previous_retweets]

    # If all the tweets have zero popularity, then pick one at random.
    if sum(weights) == 0:
        return random.choice(previous_retweets)

    # Else assign to most popular.
    return random.choices(previous_retweets, weights=weights, k=1)[0]


def load_user_from_disk(user_id):
    #print("Looking for user {}".format(user_id))
    user = models.User(user_id)

    user_filename = "{}/{}.json".format(USER_PROFILES_PATH, user_id)
    if not os.path.exists(user_filename): 
        with open("{}/{}.json".format(MISSING_USER_PROFILES_PATH, user_id), "wt") as json_file:
            json.dump({ "id": user_id }, json_file)
        return user

    # load user from file
    with open(user_filename) as json_file:
        user_dict = json.load(json_file)

        # failed when downloading
        if "done" in user_dict and user_dict["done"] == "ERROR": 
            return user

        #print(user_dict)
        if str(user_dict["id"]) != user_id:
            raise ValueError(
                "Invalid userid {} in json files".format(str(user_dict["id"]))
            )

        for key in [
            "followers_count",
            "listed_count",
            "favourites_count",
            "statuses_count",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value == 0:
                setattr(user, key, user_dict.get(key, 0))

        for key in [
            "verified",
            "protected",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value is False:
                setattr(user, key, user_dict.get(key, False))

        if user.following_count is None or user.following_count == 0:
            user.following_count = user_dict.get("friends_count", 0)

        if user.description is None:
            user.description = user_dict.get("description", None)

    user_followers_filename = "{}/{}.json".format(USER_FOLLOWERS_PATH, user_id)
    if not os.path.exists(user_followers_filename): 
        return user

    # load user followers from file
    with open("{}/{}.json".format(USER_FOLLOWERS_PATH, user_id)) as json_file:
        try:
            followers_dict = json.load(json_file)
            for follower_id in followers_dict.get("followers", []):
                user.followers.add(str(follower_id))
        except json.JSONDecodeError:
            logging.warning(f"There was an issue while trying to retrieve followers for {USER_FOLLOWERS_PATH}/{user_id}.json. Followers remains an empty set.")

    return user


def create_tweet(tweet_dict, real):
    tweet = models.Tweet(str(tweet_dict["id"]))
    tweet.real = real
    tweet.created_at = datetime.strptime(
            tweet_dict["created_at"], "%a %b %d %H:%M:%S %z %Y"
        )
    tweet.text = tweet_dict["text"]
    
    return tweet

def get_user_id(tweet_dict):
    if "user" in tweet_dict:
        user_id = str(tweet_dict["user"]["id"])
    elif "userid" in tweet_dict:
        user_id = str(tweet_dict["userid"])
    else:
        raise ValueError("Failed to parse user in tweet: {}".format(tweet_dict))
    return user_id

def create_tree(tweet_dict, min_retweets): 
    real = tweet_dict["label"] == "real"
    tweet = create_tweet(tweet_dict, real=real)
    tweet.user = load_user_from_disk(get_user_id(tweet_dict))

    for retweet_dict in tweet_dict["retweets"]:
        retweet = create_tweet(retweet_dict, real=real)
        retweet.retweet_of = tweet
        tweet.retweeted_by.append(retweet)
        retweet.user = load_user_from_disk(get_user_id(retweet_dict))

    if len(tweet.retweeted_by) < min_retweets:
        return None

    retweets = sorted(
        tweet.retweeted_by, key=lambda t: t.created_at, reverse=True
    )

    tree = nx.DiGraph()
    tree.add_node(tweet)

    tree.nodes[tweet]["delay"] = 0

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        tree.add_node(cur)

        cur_retweet_of = _find_retweet_source(cur, previous)
        tree.add_edge(cur, cur_retweet_of)

        tree.nodes[cur]['delay'] = abs((cur.created_at-cur_retweet_of.created_at).total_seconds())

        previous.append(cur)

    if tweet.real: 
        tree.graph['label'] = "real"
    else:
        tree.graph['label'] = "fake"

    return tree


def postprocess_tree(tree):
    """Given a tree convert vertices to integers and compute features.
    """
    p_tree = nx.DiGraph()

    vid = 0
    tweet_to_id = {}
    for tweet in tree.nodes(data=True):
        p_tree.add_node(vid)

        p_tree.nodes[vid]['user_id'] = tweet[0].user.id
        p_tree.nodes[vid]['delay'] = tree.nodes[tweet[0]]['delay']
        p_tree.nodes[vid]['followers_count'] = max(len(tweet[0].user.followers), tweet[0].user.followers_count)
        p_tree.nodes[vid]['following_count'] =  max(len(tweet[0].user.following), tweet[0].user.following_count)

        for key in ['verified', 'protected', 'favourites_count', 'listed_count', 'statuses_count']:
            p_tree.nodes[vid][key] = int(getattr(tweet[0].user, key))

        #if tweet.user.embedding is not None:
        #    p_tree.vertex_attrs[vid]['user_profile_embedding'] = tweet.user.embedding

        tweet_to_id[tweet[0]] = vid
        vid += 1

    for source, target in tree.edges:
        p_tree.add_edge(tweet_to_id[source], tweet_to_id[target])

    p_tree.graph['label'] = tree.graph['label']

    return p_tree


def tree_to_dict(tree):
    out = {}
    out['label'] = tree.graph['label']

    nodes = []
    for v in tree.nodes(data=True):
        node = {}
        node['id'] = v[0]
        for k, v in tree.nodes[v[0]].items():
            node[k] = v
        nodes.append(node)
    edges = []
    for source, target in tree.edges:
        edge = {}
        edge['source'] = source
        edge['target'] = target
        edges.append(edge)
    out['nodes'] = nodes
    out['edges'] = edges

    return out


def run(args):

    random.seed(31)

    logging.info("Creating trees")
    os.makedirs(args.trees_path, exist_ok=True)
    os.makedirs(MISSING_USER_PROFILES_PATH, exist_ok=True)
    

    count = 0
    for fentry in os.scandir(args.tweets):
        tweet_path = fentry.path
        with open(tweet_path) as json_file:
            tweet_dict = json.load(json_file)
            tree = create_tree(tweet_dict, min_retweets=args.min_retweets)
            if tree is not None:
                if count % 25 == 0: 
                    logging.info("{}".format(count))
                tree = postprocess_tree(tree)
                tree_path = os.path.join(args.trees_path, "trees-{}.json".format(count))
                with open(tree_path, 'w') as tree_file:
                    json.dump(tree_to_dict(tree), tree_file)
                    #tree_file.write(tree_to_json(tree))
                count += 1        


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python create_trees.py")
    parser.add_argument(
        "--tweets",
        help="Tweets directory",
        dest="tweets",
        type=str,
        default="tweets"
    )

    parser.add_argument(
        "--min-retweets",
        help="Keep only the tweets that have a number of retweets greater or equal than this value",
        dest="min_retweets",
        type=int,
        default=8
    )

    parser.add_argument(
        "--trees-path",
        help="The location where the generated trees will be stored.",
        dest="trees_path",
        type=str,
        default="produced_data/trees"
    )

    args = parser.parse_args()
    run(args)
