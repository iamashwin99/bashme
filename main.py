import os
import logging
import argparse
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


logging.set_verbosity(logging.CRITICAL)

def load_CodeLlama_llm():


    pass

def tokenise(prompt):
    pass

def main():

    # load the model
    model_name_or_path = os.getenv("MPSD_CODE_LLAMA",None) or "../models/CodeLlama-7B-Instruct-GPTQ"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None)
    # set the prompt
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="prompt to generate code for")
    args = parser.parse_args()
    prompt = args.prompt

    # construct complete prompt
    prompt_template=f'''[INST] Write bash code to solve the following coding problem that obeys the constraints. Please wrap your code answer using ```. write single line bash codes as much as possible. Keep the explanation very short.:
    {prompt}
    [/INST]
    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    print( tokenizer.decode(output[0]).split('[/INST]')[-1] )

    # parse the results
    # print the results


if __name__ == '__main__':
    main()