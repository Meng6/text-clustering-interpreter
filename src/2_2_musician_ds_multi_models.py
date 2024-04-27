import ollama, random
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from func_timeout import func_timeout, FunctionTimedOut

############################################################
LLM_MODEL = 'llama3' # 'llama3' OR 'mistral'
NUM_SAMPLES_PER_CLUSTER = 1 # pick one from {1, 3, 6, 9}
NUM_CLUSTERS = 24
REPETITIONS = 50
############################################################

def get_llms_response(dataset, num_clusters, num_samples_per_cluster, llm_model):
    cid, prompt, true = 1, "Below is the clustering results of user intentions in music domain:\n\n", []
    for cluster in random.sample(intents, num_clusters):
        true.append(cluster)
        curr_intent = data[data['intent_label'] == cluster]
        samples = curr_intent.sample(min(num_samples_per_cluster, len(curr_intent)))
        prompt = prompt + "Cluster {cid}: {samples}\n".format(cid=cid, samples=samples['input'].tolist())
        cid += 1
    prompt1 = prompt + "You are a musician. Please use one or two words to interpret each cluster."
    prompt2 = prompt + "You are a data scientist. Please use one or two words to interpret each cluster."

    response1 = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt1}], keep_alive=0)['message']['content']
    response2 = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt2}], keep_alive=0)['message']['content']

    prompt3 = """{prompt}

A musician interpreted these clusters as follows:
{response1}

A data scientist interpreted these clusters as follows:
{response2}

Based on the clustering results and their interpretations, please summarize the interpretations and provide your response in the format of "interpretation 1\ninterpretation 2\n..." directly. No extra sentences are included. The following responses should start from cluster 1 and end with cluster 24. Here are the interpreted clusters separated by a new line (do not repeat or paraphrase this sentence, one line per cluster):
""".format(prompt=prompt, response1=response1, response2=response2)
    
    response = ollama.chat(model=llm_model, messages=[{'role': 'user', 'content': prompt3}], keep_alive=0)['message']['content']

    return (response, true)

def parse_response(response):
    pred_ = [x.strip() for x in response.split('\n')]
    if len(pred_) >= 24:
        if pred_[1] == '':
            pred_ = pred_[2:]
        if pred_[-2] == '':
            pred_ = pred_[:-2]
    pred = [item for item in pred_ if item != '']
    return pred

model = INSTRUCTOR('hkunlp/instructor-large')
ordered_true = ['remove from playlist music', 'repeat all music', 'skip track music', 'repeat all off music', 'rewind music', 'play media', 'loop music', 'like music', 'previous track music', 'add to playlist music', 'play music', 'fast forward music', 'stop music', 'question music', 'pause music', 'resume music', 'dislike music', 'get track info music', 'create playlist music', 'get lyrics music', 'delete playlist music', 'start shuffle music', 'set default provider music', 'replay music']

count, res = 0, pd.DataFrame()
data = pd.read_csv("data/mtop.csv")
data = data[data['domain_label'] == 'music'] # 24 intentions
intents = data['intent_label'].unique().tolist()

while count < REPETITIONS:
    print("Iteration: ", count)

    try:
        (response, true) = func_timeout(90, get_llms_response, args=(intents, NUM_CLUSTERS, NUM_SAMPLES_PER_CLUSTER, LLM_MODEL))
    except FunctionTimedOut:
        continue
    pred = parse_response(response)

    if len(pred) == 24:
        instruction = "Represent the Intent sentence: "
        # reorder pred based on ordered_true
        try:
            ordered_pred = []
            for true_label in ordered_true:
                idx = true.index(true_label)
                ordered_pred.append(pred[idx])
            count = count + 1

            sentences_pred = [[instruction, x] for x in ordered_pred]
            sentences_true = [[instruction, x] for x in ordered_true]
            embeddings_pred = model.encode(sentences_pred)
            embeddings_true = model.encode(sentences_true)
            similarities = cosine_similarity(embeddings_pred,embeddings_true)

            if count == 1:
                rows = []
                for i in range(24):
                    rows.append([str(i+1), ordered_true[i], similarities[i][i]])
                res = pd.DataFrame(columns=["cluster", "true", count], data=rows)
            else:
                res[count] = similarities.diagonal()
            res.to_csv("output/v3_musician_ds/details/metrics_{x}_{y}.csv".format(x=NUM_CLUSTERS, y=NUM_SAMPLES_PER_CLUSTER), index=False)
        except:
            print("Skip this iteration.")

next_row_idx = len(res)
res.loc[next_row_idx, ["cluster"]+[x+1 for x in range(count)]] = ["avg"] + [res[x+1].mean() for x in range(count)]
res.loc[next_row_idx, "true"] = res.loc[next_row_idx, [x+1 for x in range(count)]].mean()
res.to_csv("output/musician_ds_multimodels_{llm}_metrics_{x}_{y}.csv".format(llm=LLM_MODEL, x=NUM_CLUSTERS, y=NUM_SAMPLES_PER_CLUSTER), index=False)
