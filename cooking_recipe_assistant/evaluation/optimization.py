from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tqdm import tqdm
import numpy as np
from .retrievers import evaluate


def calculate_mrr(
    df, 
    es, 
    index, 
    es_with_boost, 
    boosts, 
    embedding_model):

    ground_truth = df.to_dict(orient='records')
    if embedding_model:
        results = evaluate(
            ground_truth, 
            lambda q: es_with_boost(es, index, q['question'], embedding_model, 'text_vector', boosts))
    else:
        results = evaluate(
            ground_truth, 
            lambda q: es_with_boost(es, index, q['question'], boosts))
    hit_rate = results['hit_rate']
    mrr = results['mrr']
    return mrr

def objective(
    params, 
    df, 
    es, 
    index, 
    es_with_boost, 
    embeddings
):
    """Función objetivo para Hyperopt."""
    boosts = {
        'meals': params['meals'],
        'title': params['title'],
        'ingredients': params['ingredients'],
        'summary': params['summary'],
        'text': params['text'],
        'tips': params['tips'],
    }
    if embeddings:
        boosts['vector_boost'] = params['vector_boost']
    
    #hit_rate, mrr = calculate_hit_rate_mrr(df, es, index, boosts)
    mrr = calculate_mrr(df, es, index, es_with_boost, boosts, embeddings)
    
    # Pérdida combinada
    loss = 1 - mrr  # Convertir MRR a pérdida para optimización
    return {'loss': loss, 'status': STATUS_OK, 'mrr': mrr}



def define_search_space():
    """Define el espacio de búsqueda para los boosts."""
    return {
        'meals': hp.uniform('meals', 0.0, 5.0),
        'title': hp.uniform('title', 0.0, 5.0),
        'ingredients': hp.uniform('ingredients', 0.0, 5.0),
        'summary': hp.uniform('summary', 0.0, 5.0),
        'text': hp.uniform('text', 0.0, 5.0),
        'tips': hp.uniform('tips', 0.0, 5.0),
        'vector_boost': hp.uniform('vector_boost', 0.1, 0.9)
    }


def run_hyperopt(
    df, es, 
    index, 
    es_with_boost, 
    embeddings=None,
    max_evals=50
):
    """Ejecuta Hyperopt para optimizar los boosts."""
    space = define_search_space()
    #if embeddings:
    #    space['vector_boost'] = hp.uniform('vector_boost', 0.1, 0.9)

    trials = Trials()

    best = fmin(
        fn=lambda params: objective(params, df, es, index, es_with_boost, embeddings),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    print("Mejores parámetros encontrados:")
    print(best)

    best_boosts = {
        'meals': best['meals'],
        'title': best['title'],
        'ingredients': best['ingredients'],
        'summary': best['summary'],
        'text': best['text'],
        'tips': best['tips'],
    }
    if embeddings:
        best_boosts['vector_boost'] = best['vector_boost']

    print("Boosts optimizados:")
    print(best_boosts)

    # Encontrar el ensayo con el menor loss (es decir, el mayor MRR)
    best_trial = min(trials.results, key=lambda x: x['loss'])
    best_mrr = best_trial['mrr']

    print(f"El mejor valor de MRR es: {best_mrr}") 

    return best_boosts, best_mrr
