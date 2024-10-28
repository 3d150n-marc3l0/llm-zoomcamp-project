




def calculate_hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

    
def calculate_mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


def evaluate(
    ground_truth, 
    search_function
):
    relevance_total = []
    #for q in tqdm(ground_truth):
    for q in ground_truth:
        #print(q)
        doc_id = q['id']
        results = search_function(q)
        #for d in results:
        #    print(d)
        #    break
        #relevance = [d['id'] == doc_id for d in results]
        doc_ids = [d['id'] for d in results]
        #print(doc_id, q['id'], doc_ids)
        relevance = [doc_id.startswith(d['doc_id']) for d in results]
        relevance_total.append(relevance)

    #for d in relevance_total:
    #    print(d)
    return {
        'hit_rate': calculate_hit_rate(relevance_total),
        'mrr': calculate_mrr(relevance_total),
    }