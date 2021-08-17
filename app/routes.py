import faulthandler
faulthandler.enable()

import logging
import numpy as np
import operator
import gc
import torch
import pandas as pd
from flask import Blueprint, jsonify, request
from pycarol import Carol, Storage
from sentence_transformers import SentenceTransformer, util
from webargs import fields, ValidationError
from webargs.flaskparser import parser
import re
import ftfy
from unidecode import unidecode
from time import time
from pycarol.apps import Apps

# Logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

login = Carol()
storage = Storage(login)
#_settings = Apps(login).get_settings()

server_bp = Blueprint('main', __name__)

def transformSentences(m, custom_stopwords):
    # Ensure the parameter type as string
    mproc0 = str(m)
    
    # Set all messages to a standard encoding
    mproc1 = ftfy.fix_encoding(mproc0)
    
    # Replaces accentuation from chars. Ex.: "férias" becomes "ferias" 
    mproc2 = unidecode(mproc1)

    login = Carol()
    _settings = Apps(login).get_settings()
    preproc_mode = _settings.get('preproc_mode')
    
    if preproc_mode == "advanced":
        # Removes special chars from the sentence. Ex.: 
        #  - before: "MP - SIGAEST - MATA330/MATA331 - HELP CTGNOCAD"
        #  - after:  "MP   SIGAEST   MATA330 MATA331   HELP CTGNOCAD"
        mproc3 = re.sub('[^0-9a-zA-Z]', " ", mproc2)
        
        # Sets capital to lower case maintaining full upper case tokens and remove portuguese stop words.
        #  - before: "MP   MEU RH   Horario ou Data registrado errado em solicitacoes do MEU RH"
        #  - after:  "MP MEU RH horario data registrado errado solicitacoes MEU RH"
        mproc4 = " ".join([t.lower() for t in mproc3.split() if t not in custom_stopwords])
        
        return mproc4

    else:
        return mproc2

def update_embeddings():
    global df

    login = Carol()
    _settings = Apps(login).get_settings()
    kb_parameter = _settings.get('knowledge_base')

    kb_list = [ kb.lstrip().rstrip() for kb in kb_parameter.split(",") ]
    
    kb_datasets = []

    for kb in kb_list:
        # Get dataframe from Carol storage
        logger.info(f'Loading documents from {kb}.')
        try:
            df_tmp = storage.load(kb, format='pickle', cache=False)
            df_tmp["database"] = kb

            kb_datasets.append(df_tmp)
        except Exception as e:
            logger.warn(f'Failed to load {kb}. Discarding.')

    # Update dataframe after it has been loaded from Carol storage
    df = pd.concat(kb_datasets, ignore_index=False)
    return kb_list


def findMatches(title, query_tokens, scale):
    title_tokens = title.split()
    
    matches=0
    for qt in query_tokens:
        if qt in title_tokens: 
            matches += 1
            
    return scale * (matches/len(query_tokens))

def keywordSearch(kb, q, nresults=1, threshold=None, fields=None):
    if fields:
        title_kb = kb[kb["sentence_source"].isin(fields)].copy()
    else:
        title_kb = kb.copy()
    
    # Checks if any article contains the exact query as substring
    substr_df = title_kb[title_kb["sentence"].str.contains(q)].copy()
    
    if len(substr_df) >= 1:
        substr_df["score"] = 1.0
    
    # If the query is not a substring of the title...
    else:
        # ... try moken matches individualy
        query_tokens = q.split()
        title_kb["score"] = title_kb["sentence"].apply(lambda s: findMatches(s, query_tokens, scale=0.9))
        title_kb.sort_values(by="score", ascending=False, inplace=True)
        substr_df = title_kb.copy()
    
    if threshold:
        substr_df = substr_df[substr_df["score"] >= threshold].copy()
    else:
        substr_df =  substr_df.head(nresults)

    # Discount score by its length difference to query
    substr_df["s_length"] = [ len(s) for s in substr_df["sentence"].values ]
    query_length = len(q)
    max_length = substr_df["s_length"].max()
    substr_df["score"] = substr_df["s_length"].apply(lambda x: max(0.80, 1 - (x - query_length)/max_length))
    substr_df.drop(columns=["s_length"], inplace=True)
    
    return substr_df

def reverseKeywordSearch(kb, q, fields=None):
    if fields:
        title_kb = kb[kb["sentence_source"].isin(fields)].copy()
    else:
        title_kb = kb.copy()

    # Checks if any title or question is exactly contained on the query
    substr_df = title_kb[title_kb["sentence"].apply(lambda x: True if x in q else False)].copy()
    if len(substr_df) >= 1: substr_df["score"] = 1.0

    # Discount score by its length difference to query
    substr_df["s_length"] = [ len(s) for s in substr_df["sentence"].values ]
    query_length = len(q)
    substr_df["score"] = substr_df["s_length"].apply(lambda x: max(0.75, 1 - (query_length - x)/query_length))
    substr_df.drop(columns=["s_length"], inplace=True)

    substr_df["type_of_search"] = "rkeyword"

    return substr_df

def hasCode(txt):
    # The first pattern captures de definition + numbers, such as "erro d2233".
    pattern1 = re.compile(r'(?:rotina|rejeicao|registro|error|erro|evento)\s[a-z]*[0-9.]{3,}')
    
    # The second pattern, an special case observed on rejections, captures de numbers + definition, such as "934 rejeicao".
    pattern2 = re.compile(r'[0-9]{3,}[\s]+(?:rejeicao)')

    # The third pattern, an special case observed on nts, captures de numbers + date, such as "nt 22 07 2020".
    pattern3 = re.compile(r'(\s)(?:nt)\s[0-9 ]{2,10}')

    # The fourth pattern, an special case observed on mps, captures de numbers + definition, such as "mp 927 2020" or "mp 936".
    pattern4 = re.compile(r'(?:mp)\s[0-9\s]{3,}')

    if pattern1.match(txt) or pattern2.match(txt) or pattern3.match(txt) or pattern4.match(txt): return True
    else: return False

def get_similar_questions(model, sentence_embeddings_df, query, threshold=None, k=None, validation=False):
    global keywordsearch_flag

    # Reading stopwords to be removed
    #logger.debug('Loading list of stopwords.')
    with open('/app/cfg/stopwords.txt') as f:
        custom_stopwords = f.read().splitlines()

    query = transformSentences(query, custom_stopwords)

    if keywordsearch_flag:
        logger.info(f'Trying keyword search on \"{query}\".')
        keywordResults = keywordSearch(kb=sentence_embeddings_df, q=query, threshold=1)

    else: 
        keywordResults = pd.DataFrame(columns=sentence_embeddings_df.columns)

    logger.info(f'Trying semantic search on \"{query}\".')
    semanticResults = sentence_embeddings_df.copy()
    query_vec = model.encode([query], convert_to_tensor=True)
    torch_l = [torch.from_numpy(v) for v in semanticResults['sentence_embedding'].values]
    sentence_embeddings = torch.stack(torch_l, dim=0)
    score = util.pytorch_cos_sim(query_vec, sentence_embeddings)
    semanticResults['score'] = list(score.cpu().detach().numpy()[-1,:])

    logger.debug(f'Merging {keywordResults.shape[0]} semantic to {semanticResults.shape[0]} keyword results.')
    if (len(keywordResults) > 0) and (keywordResults["score"].mean() == 1.0) and hasCode(query):
        # if there is any exact match on keywords, ignore semantic search
        results = keywordResults.copy()
    else:
        # if keyword search didn't succeed returns a mix between semantic and keyword search results
        results = pd.concat([keywordResults, semanticResults], ignore_index=True)

    if (len(results) < 1):
        # Tries reverse keyword search as the last resource
        results = reverseKeywordSearch(kb=sentence_embeddings_df, q=query)
        logger.info(f'Reverse keyword search applied as no result has been found so far. Retrieved articles: \"{results.shape[0]}\".')

    # Applying threshold if defined
    if threshold and not results.empty:

        # If there is a general threshold sets it to start with. 
        # Particular thresholds for each column will be set after the general
        if "all" in threshold:

            logger.info(f'Using general threshold {threshold["all"]} for all columns without custom threshold.')
            results["custom_threshold"] = int(threshold["all"])
            del threshold['all']
            
        else:
            results["custom_threshold"] = None

        # Setting the particular thresholds (per column)
        for c in threshold.keys():

            logger.info(f'Using {threshold[c]} threshold for {c}.')
            results.loc[results["sentence_source"] == c, "custom_threshold"] = int(threshold[c])

        results = results[results["score"] >= results["custom_threshold"] / 100].copy()
        articlesAfterThreshold = results["id"].nunique()
        logger.info(f'Total of {articlesAfterThreshold} articles satisfying the threshold.')

    results.drop(columns=['sentence_embedding'], inplace=True)
    results.sort_values(by=['score'], inplace=True, ascending=False)
    if not validation: results.drop_duplicates(subset=['id'], inplace=True)
    total_matches = results.shape[0]
    if k: results = results.head(k)

    return results, total_matches


# Initialize variables
df = None
model = None
keywordsearch_flag = False

logger.info('App started. Please, make sure you load the model and knowledge base before you start.')

# Get files from Carol storage
#update_embeddings()

@server_bp.route('/', methods=['GET'])
def ping():
    return jsonify("App is running. Available routes are: \
    \\r\\n     - /query: Send a request to /query for document searching.\
    \\r\\n     - /validate: Validate the similarity between a given query and expected documents.\
    \\r\\n     - /update_embeddings: Update the document embeddings as defined on settings.\
    \\r\\n     - /load_model: Loads NLP model defined on settings.\
    \\r\\n     - /switch_keywordsearch: Enable / disable keyword search.")

# Alows to enable/ disable keyword search on run time
@server_bp.route('/switch_keywordsearch', methods=['GET'])
def switch_keywordsearch():
    global keywordsearch_flag

    keywordsearch_flag = not keywordsearch_flag
    logger.info(f'Keyword search switch set to: {keywordsearch_flag}.')
    return jsonify(f'Keyword search switch set to: {keywordsearch_flag}.')

@server_bp.route('/load_model', methods=['GET'])
def load_model():
    global model

    login = Carol()
    _settings = Apps(login).get_settings()
    model_storage_file = _settings.get('model_storage_file')
    model_sentencetransformers = _settings.get('model_sentencetransformers')

    try:
        gpu = torch.cuda.is_available()
    except Exception as e:
        gpu = False

    if model_storage_file:
        name = model_storage_file
        logger.info(f'Loading model {name}. Using GPU: {gpu}.')
        storage = Storage(login)
        model = storage.load(model_storage_file, format='pickle', cache=False)

        if gpu: 
            model.to(torch.device('cuda'))
            model._target_device = torch.device('cuda')
        else: 
            model.to(torch.device('cpu'))
            model._target_device = torch.device('cpu')
        
    else:
        name = model_sentencetransformers
        logger.info(f'Loading model {name}. Using GPU: {gpu}.')
        model = SentenceTransformer(model_sentencetransformers)

    return jsonify(f'Model {name} loaded.')

@server_bp.route('/update_embeddings', methods=['GET'])
def update_embeddings_route():
    kb_list = update_embeddings()
    return jsonify(f'Embeddings are updated. The following knowledgebases have been loaded from the storage: {kb_list}.')

# Route to be used for validation purposes. The user can send
# a query and expected results, the response will be the top
# matches on the provided article(s) and their respective sco
# res.
@server_bp.route('/validate', methods=['POST'])
def validate():

    query_arg = {
        "query": fields.Str(required=True, 
            description='Query to be searched in the documents.'),
        "expected_ids": fields.List(fields.Str(), required=True, description='List of expected articles to compare to query.')
    }

    args = parser.parse(query_arg, request)
    query = args['query']
    expected_ids = args['expected_ids']
    expected_ids = [int(i) for i in expected_ids]

    logger.info(f'Validating query \"{query}\" against target IDS {expected_ids}.')

    df_tmp = df[df["id"].isin(expected_ids)].copy()
    if len(df_tmp) > 0:
        logger.info(f'Validating query {query} against the following articles: {expected_ids}')
        results_df, total_matches = get_similar_questions(model=model, 
                                                          sentence_embeddings_df=df_tmp, 
                                                          query=query,
                                                          validation=True)

        records_dict_tmp = results_df.to_dict('records')
        records_dict = sorted(records_dict_tmp, key=operator.itemgetter('score'), reverse=True)
    else:
        records_dict, total_matches = ([], 0)

    logger.info(f'Returning validation results.')
    return jsonify({'total_matches': total_matches, 'topk_results': records_dict})
    
@server_bp.route('/query', methods=['POST'])
def query():
    t0 = time()

    if model is None:
        logger.warn(f'It looks like the NLP model has not been loaded yet. This operation usually takes up to 1 minute.')
        load_model()

    if df is None:
        logger.warn(f'It looks like the knowledge base has not been loaded yet. This operation usually takes up to 1 minute.')
        update_embeddings()

    query_arg = {
        "query": fields.Str(required=True, 
            description='Query to be searched in the documents.'),
        "threshold": fields.Int(required=False, missing=55, description='Documents with scores below this threshold \
            are not considered similar to the query. Default: 55.'),
        "k": fields.Int(required=False, missing=5, description='Number of similar documents to be return. \
            Default: 5.'),
        "filters": fields.List(fields.Dict(keys=fields.Str(), values=fields.Raw(), required=False), required=False, missing=None, validate=validate_filter, description='List of dictionaries \
            in which the filter_field means the name of the field and the filter_value the value used for filtering the documents.'),
        "response_columns": fields.List(fields.Str(), required=False, missing=None, validate=validate_response_columns, description='List of columns \
            from the documents base that should be returned in the response.'),
        "threshold_custom": fields.Dict(keys=fields.Str(), values=fields.Raw(), required=False, validate=validate_threshold_custom, description='Dictionary \
            in which the key is the source from the document in which the sentences has been taking and the values is the the threshold to be considered for that group of sentences.')
    }
    args = parser.parse(query_arg, request)
    query = args['query']
    threshold = args['threshold']
    k = args['k']
    filters = args['filters']
    response_columns = args['response_columns']
    threshold_custom = args.get('threshold_custom')
    df_tmp = df.copy()

    # If there's a custom threshold defined it overcomes the general threshold
    if threshold_custom not in ("", None):
        logger.info('Consolidating thresholds.')
        # If there's no "all" key on custom threshold, sets it to the general threshold provided (or default)
        if "all" not in threshold_custom:
            threshold_custom["all"] = threshold
        threshold = threshold_custom
    elif threshold not in ("", None):
        threshold = {"all":threshold}
    else:
        threshold = None

    logger.info(f'Running filter on {len(df_tmp)} articles.')

    # Expceting filters to be passed as a list of dicts as in the example below:
    #     [{'filter_field': 'modulo', 'filter_value': "ARQUIVO SIGAFIS"}] 
    if filters in ("", None): filters = []
    for filter in filters:
        if df_tmp.empty:
            break
        
        filter_field, filter_value = (filter.get('filter_field'), filter.get('filter_value'))
        logger.info(f'Applying filter \"{filter_field}\" == \"{filter_value}\".')

        filter_field_type = df_tmp.iloc[0][filter_field]

        if isinstance(filter_field_type, list) and isinstance(filter_value, list) and filter_value:
            tmp_dfs = []
            for value in filter_value:
                value = value.lower()
                tmp_df = df_tmp[([any(value == v.lower() for v in values) for values in df_tmp[filter_field]])]
                if not tmp_df.empty:
                    tmp_dfs.append(tmp_df)
            if tmp_dfs:
                final_df = pd.concat(tmp_dfs)
                df_tmp = final_df

        elif isinstance(filter_field_type, str) and isinstance(filter_value, list):
            df_tmp = df_tmp[df_tmp[filter_field].isin(filter_value)]

        else:
            df_tmp = df_tmp[df_tmp[filter_field] == filter_value]

    if not df_tmp.empty:
        logger.info(f'Total records satisfying the filter: {len(df_tmp)}.')

    else:
        logger.warn(f'No results returned from filter.')
        return jsonify({'total_matches': 0, 'topk_results': []})

    results_df, total_matches = get_similar_questions(model=model, 
                                                      sentence_embeddings_df=df_tmp, 
                                                      query=query, 
                                                      threshold=threshold, 
                                                      k=k)

    if len(results_df) < 1:
        logger.warn(f'Unable to find any similar article for the threshold {threshold}.')
        return jsonify({'total_matches': 0, 'topk_results': []})

    if response_columns:
        results_df = results_df[list(set(response_columns + ['score', 'sentence_source']))]

    records_dict_tmp = results_df.to_dict('records')
    records_dict = sorted(records_dict_tmp, key=operator.itemgetter('score'), reverse=True)

    return jsonify({'total_matches': total_matches, 'topk_results': records_dict})

@server_bp.errorhandler(422)
@server_bp.errorhandler(400)
def handle_error(err):
    headers = err.data.get("headers", None)
    messages = err.data.get("messages", ["Invalid request."])
    messages = messages.get('json', messages)
    if headers:
        return jsonify(messages), err.code, headers
    else:
        return jsonify(messages), err.code

def validate_filter(val):
    filter_columns = []
    filters = list(val)
    for filter in filters:
        filter_field = filter.get('filter_field')
        if filter_field:
            filter_columns.append(filter_field)
        else:
            raise ValidationError("The key 'filter_field' must be filled when you are using filters.") 
    if filters and any(c not in df.columns for c in filter_columns):
        raise ValidationError("One or more columns that you are trying to filter does not exist in the documents base.")


def validate_response_columns(val):
    response_columns = list(val)
    if response_columns and any(c not in df.columns for c in response_columns):
        raise ValidationError("One or more columns that you are trying to return does not exist in the documents base.")


def validate_threshold_custom(val):
    if 'sentence_source' not in df.columns:
        raise ValidationError("The sentence_source column does not exist in the documents base so it will not be possible to filter the custom threshold.")
    sentence_source_values = list(df['sentence_source'].unique()) + ["all"]
    sentence_source_filter_values = val.keys()
    if sentence_source_values and any(s not in sentence_source_values for s in sentence_source_filter_values):
        raise ValidationError("One or more values on custom_threshold does not exist in the knowledge base.")