import os
import logging
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


logging.set_verbosity(logging.CRITICAL)

def load_CodeLlama_llm():
    model_name_or_path = os.getenv("MPSD_CODE_LLAMA",None) or "../models/CodeLlama-7B-Instruct-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None)

    return model

def tokenise(prompt):
    prompt_template=f'''[INST] Write bash code to solve the following coding problem that obeys the constraints. Please wrap your code answer using ```. write oneline bash codes as much as possible:
    {prompt}
    [/INST]
    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)

def main():

    # load the model
    # set the prompt
    # query the prompt from user input
    # parse the results
    # print the results
    pass


if __name__ == '__main__':
    main()