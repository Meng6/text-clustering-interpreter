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

def get_llms_response(dataset, num_clusters):
    cid, prompt, true = 1, "Below is the clustering results of user intentions in music domain:\n\n", []
    for cluster in random.sample(intents, NUM_CLUSTERS):
        true.append(cluster)
        curr_intent = data[data['intent_label'] == cluster]
        samples = curr_intent.sample(min(NUM_SAMPLES_PER_CLUSTER, len(curr_intent)))
        prompt = prompt + "Cluster {cid}: {samples}\n".format(cid=cid, samples=samples['input'].tolist())
        cid += 1
    
    msg_l = [{'role': 'user', 'content': prompt + """\nPlease use one or two words to interpret each cluster without explanation.\n"""}]
    res_l = ollama.chat(model='llama3', messages=msg_l, keep_alive=0)['message']['content']

    msg_m = [{'role': 'user', 'content': """{prompt}
Someone interpreted these clusters as follows:
{res_l}.
Based on the clustering results and the above interpretations, please summarize the interpretations and provide your response in the format of "interpretation 1\ninterpretation 2\n..." directly. No extra sentences are included. The following responses should start from cluster 1 and end with cluster 24. Here are the interpreted clusters separated by a new line (do not repeat or paraphrase this sentence, one line per cluster):""".format(prompt=prompt, res_l=res_l)}]
    # Based on the clustering results and the above interpretations, please use one or two words to interpret each cluster without explanation and provide your response in the formate of [cluster number: interpretation of the cluster].format(prompt=prompt, res_l=res_l)}]
    res_m = ollama.chat(model='mistral', messages=msg_m, keep_alive=0)['message']['content']
    print(res_m)
    return (res_m, true)

model = INSTRUCTOR('hkunlp/instructor-large')
ordered_true = ['remove from playlist music', 'repeat all music', 'skip track music', 'repeat all off music', 'rewind music', 'play media', 'loop music', 'like music', 'previous track music', 'add to playlist music', 'play music', 'fast forward music', 'stop music', 'question music', 'pause music', 'resume music', 'dislike music', 'get track info music', 'create playlist music', 'get lyrics music', 'delete playlist music', 'start shuffle music', 'set default provider music', 'replay music']

count, res = 0, pd.DataFrame()
data = pd.read_csv("mtop_large_merged.csv")
data = data[data['domain_label'] == 'music'] # 24 intentions
intents = data['intent_label'].unique().tolist()

while count < REPETITIONS:
    
    print("Iteration: {count}".format(count=count))

    try:
        (res_m, true) = func_timeout(90, get_llms_response, args=(intents, NUM_CLUSTERS))
    except FunctionTimedOut:
        continue

    pred = [x.strip() for x in res_m.split("\n")]
    print(pred)
    if len(pred) >= 24:
        if pred[1] == '':
            pred = pred[2:]
        if pred[-2] == '':
            pred = pred[:-2]
        if '24' in pred[-1]:
            pred = pred[-24:]
        print('revised',pred)

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
                print('res', res)
            else:
                res[count] = similarities.diagonal()
                print('else res', res)
        except:
            print("Skip this iteration.")
    else:
        print('less/greater than 24 clusters')
result = res[res.columns[-50:]].mean(axis = 1)
result.to_csv("output_/metrics/metrics_{0}.csv".format(NUM_SAMPLES_PER_CLUSTER), index = False)
