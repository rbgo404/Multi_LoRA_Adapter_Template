from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from peft import PeftModel


class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cuda")
        self.model = PeftModel.from_pretrained(base_model,"CATIE-AQ/mistral7B-FR-InstructNLP-LoRA", adapter_name="french")
        self.model.load_adapter("Liu-Xiang/mistral7bit-lora-sql", adapter_name="sql")
        self.model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")
        self.model.load_adapter("uukuguy/Mistral-7B-OpenOrca-lora", adapter_name="orca")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        adapter_name = inputs.pop("adapter_name")

        print(self.model.active_adapter)
        if (self.model.active_adapter) != adapter_name:
            self.model.set_adapter(adapter_name)

        temperature = inputs.get("temperature",0.7)
        repetition_penalty = float(inputs.get("repetition_penalty",1.18))
        max_new_tokens = inputs.get("max_new_tokens",128)

        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        result = self.tokenizer.decode(self.model.generate(**model_input,temperature=temperature, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)[0], skip_special_tokens=True)

        return {'generated_result': result}

    def finalize(self):
        self.model = None
