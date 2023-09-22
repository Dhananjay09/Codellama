
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

    def get_generator(self, params, source_text):
        generator_input = {
            'eos_token_id':self.tokenizer.eos_token_id,
        }
        if params:
            for key,value in params.items():
                    generator_input[key] = value

        answer = self.pipeline(source_text, **generator_input)
        return answer
    

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        print("payload:", payload)
        inputs = payload["instances"]

        results = []

        for input in inputs:
            source_text = input.pop("text")
            sequences = self.get_generator(params=input, source_text=source_text)
            result = []

            for seq in sequences:
                print(f"Result: {seq['generated_text']}")
                result.append(seq['generated_text'])

            results.append(result)
        
        return {"predictions": results}

if __name__ == "__main__":
    model = MetaLLMA2Model("codellama-eos")
    ModelServer().start([model])
