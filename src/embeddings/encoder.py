from sentence_transformers import SentenceTransformer
import os

FILE_DIR = os.getcwd()

if not (FILE_DIR.endswith("bio-med-rag")):
    raise ValueError("Please run this script from the bio-med-rag directory")

class Encoder():
    def __init__(self, texts, batch_size, model_name, device):
        self.texts= texts
        self.batch_size= batch_size
        self.model_name= model_name
        self.device= device
        self.model = self.get_embed_model()


    def get_embed_model(self):
        embed_model= SentenceTransformer(
        self.model_name,
        device= self.device)
        return embed_model

    def encode(self):
        return self.model.encode(self.texts, 
        show_progress_bar= True, 
        batch_size= self.batch_size,
        convert_to_numpy= True,
        normalize_embeddings=True)

    

    





    
    