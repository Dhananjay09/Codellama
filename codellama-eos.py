
from kserve import Model, ModelServer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from typing import Dict

class MetaLLMA2Model(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.ready = False
       self.tokenizer = None
       self.model_id = 'codellama/CodeLlama-7b-hf'
       self.model_local_path = '/mnt/models'
       self.load()

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_local_path,
                                                          trust_remote_code=True,
                                                          device_map='auto')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path)
        self.pipeline = pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        print("payload:", payload)
        inputs = payload["instances"]
        source_text = inputs[0]["text"]

        sequences = self.pipeline(source_text,
                                  do_sample=True,
                                  top_k=10,
                                  top_p=0.95,
                                  num_return_sequences=1,
                                  eos_token_id=self.tokenizer.eos_token_id,
                                  max_length=200,
                                  )

        results = []
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
            results.append(seq['generated_text'])
        
        return {"predictions": results}

if __name__ == "__main__":
    model = MetaLLMA2Model("codellama/CodeLlama-7b-hf")
    ModelServer().start([model])
