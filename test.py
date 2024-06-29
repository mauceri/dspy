import os
import openai
import dspy
from typing import Optional, Union
from ragatouille import RAGPretrainedModel
from typing import Optional
import dspy
from dsp.utils import dotdict
from dspy.teleprompt import BootstrapFewShot
#from dspy.teleprompt import MIPRO
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import functools
from dspy import context

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

class DotDict(dict):
    """Un dictionnaire qui permet l'accès via des attributs, c'est un patch d'un bug DSPy
        Cf. https://github.com/stanfordnlp/dspy/issues/166"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

class RAGatouilleRM(dspy.Retrieve):
    def __init__(
        self,
        index_root: str,
        index_name: str, 
        k: int = 3,
    ):
        self.index_root = index_root
        self.index_name = index_name
        self.k = k
        self.rag = RAGPretrainedModel.from_index(index_root+index_name)
        
    def forward(self, query_or_queries:str, k:Optional[int]=None) -> dspy.Prediction:
        if k is None:
            k = self.k
        raw_response = self.rag.search(query_or_queries,k=k)
        #print(raw_response)
        response = [DotDict({"long_text":item["content"]}) for item in raw_response]#
        #response = dspy.Prediction(passages=[dotdict({"long_text": item["content"]}) for item in raw_response])
        #Cf. bug https://github.com/stanfordnlp/dspy/issues/166
        #return dspy.Prediction(
        #    passages=response
        #)
        return response



class Anyscale(dspy.HFModel):
    def __init__(self, model, url,**kwargs):
        #super().__init__(model=model, is_client=True)
        self.model = model
        self.url = url
        self.kwargs=kwargs
        self.initial_instruction = "Vous exprimez seulement en français"
        print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {os.getenv("ANYSCALE_API_KEY")}")
        self.client = openai.OpenAI(
            base_url = self.url,
            api_key = os.getenv("ANYSCALE_API_KEY"),
        )
        # print(self.kwargs)
        
    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        print(f"*************** {kwargs}")
        print(f"########### {prompt}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            #temperature=0.7,
            **kwargs
        )
        #print(response.model_dump_json())
        try:
            # completions = json_response["generated_text"]
            #print(response)
            completions = response.choices

            response = {"prompt": prompt, "choices": [c.message.content for c in completions]}
            print(response)
            return response
        except Exception:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


class SimpleQR(dspy.Signature):
    """Répondez aux questions par des textes courts et circonstanciés. Vous vous exprimez en français"""

    question = dspy.InputField()
    reponse = dspy.OutputField(desc="Le plus souvent entre 10 et 25 mots")


from dspy.primitives.assertions import assert_transform_module, backtrack_handler

def validate_query_distinction_local(previous_queries, query):
    """check if query is distinct from previous queries"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True

def isfr(txt):
    lang = ""
    try:
        lang = detect(txt)
    except:
        print(f"Erreur détection langue")
        return False        
        # Charger le modèle spaCy correspondant à la langue détectée
        # Note : Assurez-vous d'avoir téléchargé les modèles correspondants pour chaque langue supportée
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%% {lang}")
    if lang != "fr":
        return False
    else:
        return True

def is_too_long(txt,max):
    return len(txt.split(' ')) > max

class GenerationDeReponse(dspy.Signature):
    """Répondez aux questions par des textes courts et circonstanciés. Vous vous exprimez en français"""
    #contexte:list[str] = dspy.InputField(desc="Peut contenir des faits pertinents",format=list)
    contexte:list[str]= dspy.InputField(desc=["Peut contenir des faits pertinents]"],format=list)
    question = dspy.InputField()
    reponse = dspy.OutputField(desc="La réponse doit compter moins de 20 mots")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.recherche = dspy.Retrieve(k=num_passages)
        self.generation_reponse = dspy.ChainOfThought(GenerationDeReponse)
    
    def forward(self, question):
        contexte = self.recherche(question).passages
        prediction = self.generation_reponse(contexte=contexte, question=question)
        
        #print(f"{'*'*50}contexte={contexte}")
        return dspy.Prediction(contexte=contexte, reponse=prediction.reponse)

class RAGEvaluation(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.recherche = dspy.Retrieve(k=num_passages)
        self.generation_reponse = dspy.ChainOfThought(GenerationDeReponse)
    
    def forward(self, question):
        contexte = self.recherche(question).passages
        prediction = self.generation_reponse(contexte=contexte, question=question)
        dspy.Suggest(isfr(prediction.reponse),"Le texte doit être rédigé en français")
        dspy.Suggest(is_too_long(prediction.reponse,20),"Le texte ne doit pas dépasser 20 mots")
        
        #print(f"{'*'*50}contexte={contexte}")
        return dspy.Prediction(contexte=contexte, reponse=prediction.reponse)
        
def main():
    import openai
    import dspy

    raga = RAGatouilleRM(index_root="./.ragatouille/colbert/indexes/",index_name="minicorpus",k=5)
    #turbo = dspy.OpenAI(model='gpt-3.5-turbo')
    #os.environ["OPENAI_API_KEY"] = openai.api_key

    #anyscale = Anyscale(model='mistralai/Mixtral-8x7B-Instruct-v0.1',url="https://api.endpoints.anyscale.com/v1",temperature=0.7,stream=False)
    anyscale = dspy.Anyscale(model='mistralai/Mixtral-8x7B-Instruct-v0.1')
    #anyscale = dspy.Anyscale(model='meta-llama/Llama-2-70b-chat-hf')   
    mixtral = dspy.OpenAI(model=os.getenv('ANYSCALE_API_BASE'), model_type="chat",api_key=os.getenv('ANYSCALE_API_KEY'))
    dspy.settings.configure(lm=anyscale, rm=raga)
    # Define the predictor.
    predicteur = dspy.Predict(SimpleQR)

    # Call the predictor on a particular input.
    pred = predicteur(question="Qui est Baldur ?")

    # Print the input and the prediction.
    print(f"Réponse: {pred.reponse}")

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()