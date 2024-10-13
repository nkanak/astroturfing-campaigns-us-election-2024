import json
import os
from collections import defaultdict
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

RESULT_FILE = 'result_zero_shot_adjaceny.json'

def extract_text_and_graph(data):
    text = data['nodes'][0]['tweet_text'].split(' ', 2)[-1]
    id_mapping = {node['id']: str(i+1) for i, node in enumerate(data['nodes'])}
    graph = [f"({id_mapping[edge['source']]}->{id_mapping[edge['target']]})" for edge in data['edges']]
    return text, ", ".join(graph)

def check_astroturfing(text, graph, examples):
    system_content = f'''You are an intelligent classifier capable of identifying political fake news campaigns. Your task is to report a political fake news campaign.
        A coordinated political fake news campaign is an organized effort by a group, organization, or government to deliberately create and spread false or misleading information with the intention of influencing public opinion or political outcomes.
        These campaigns are driven by specific political motives, such as discrediting opponents, swaying voter preferences, or promoting particular ideologies.
        They rely on various forms of media, particularly social networks, to disseminate the false narratives widely and effectively.
        The information is deliberately deceptive, often using fabricated or manipulated content, and the goal is to undermine trust in democratic processes, shape public debates, or achieve electoral advantages.
        You may also analyze the social interactions among users who purposely tweet or retweet a piece of information to identify coordinated fake news campaigns, but a retweet action is not always malicious. 
        I will give you information about the tweet and re-tweet actions in a social network in a tree-based graph format. I will also give you the text of the main tweet.
        For every node of the tree-based I will give you some information regarding its user. 
        'fake' denotes that you believe the given input is an organized political fake news campaign. 'real' means that you believe the given input is not an organized political fake news campaign, but a simple interaction of organic users. 
        I will also provide you with some examples with their labels. You must take into consideration the examples as well. You can find the examples below:
        # Start of examples #
        {examples}
        # End of examples #
        Please analyze the following text and its associated tree-based graph.
        Then, determine if it shows signs of being part of a political fake news campaign.
        The output must be a single 'fake' or 'real'. Do not explain your result.
    '''

    user_content = f'''Text to analyze: "{text}"
    Tree-based graph: "{graph}"
    '''

    
    #response = ollama.chat(model='llama3.1:70b', messages=[
    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'system',
            'content': system_content
        },
        {
            'role': 'user',
            'content': user_content
        },
    ],
    options={
        "temperature": 0.8
    })
    return response['message']['content']

def load_results():
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_result(results, filename, prediction):
    results[filename] = prediction
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def get_examples(directory) -> str:
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    examples = ""
    for i, filename in enumerate(json_files):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
        text, graph = extract_text_and_graph(data)
        label = data["label"]
        #examples += f'Example {i+1} text:"{text}"\nExample {i+1} graph of retweet actions: {graph}\nExample {i+1} label:{label}.\n'
        examples += f'Example {i+1} text:"{text}"\nExample {i+1} label:{label}.\n'
    return examples


def process_files(folder_path):
    results = load_results()
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    examples = get_examples("dataset_examples")
    
    with tqdm(total=len(json_files), desc="Processing files") as pbar:
        for filename in json_files:
            if filename in results:
                pbar.update(1)
                continue
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            text, graph = extract_text_and_graph(data)
            prediction = check_astroturfing(text, graph, examples)
            print(file_path)
            print(text)
            print(graph)
            print(prediction)
            print(data["label"])
            print('################')
            #prediction = prediction.split("<response>")[1].split("</response>")[0]
            
            save_result(results, filename, prediction)
            
            pbar.update(1)
    
    return results

def calculate_metrics(folder_path, results):
    true_labels = []
    predicted_labels = []
    
    for filename, prediction in results.items():
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        true_labels.append(1 if data['label'] == 'fake' else 0)
        predicted_labels.append(1 if 'fake' in prediction.lower() else 0)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1, roc_auc

def main():
    folder_path = './dataset1/test'  # Adjust this to your test folder path
    #folder_path = './dataset_small'  # Adjust this to your test folder path
    results = process_files(folder_path)
    
    # Calculate metrics
    accuracy, precision, recall, f1, roc_auc = calculate_metrics(folder_path, results)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    main()