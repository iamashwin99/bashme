import os
import logging
import argparse
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
from rich.syntax import Syntax
# import cProfile

bashme_logo = """
 __        __              ___
|__)  /\  /__` |__|  |\/| |__
|__) /~~\ .__/ |  |  |  | |___
-----------------------------------------------------
"""

logging.set_verbosity(logging.CRITICAL)


def load_CodeLlama_llm():
    pass


def tokenise(prompt):
    pass


def main():
    # set the prompt
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="prompt to generate code for")
    parser.add_argument(
        "--explain", action="store_true", help="include explanation in output"
    )

    args = parser.parse_args()
    prompt = args.prompt
    include_explanation = args.explain

    # load the model
    # TODO try to use all gpus if possible
    # import torch
    # print(f"Available no of gpus:{torch.cuda.device_count()}"")
    model_name_or_path = (
        os.getenv("MPSD_CODE_LLAMA", None) or "../models/CodeLlama-7B-Instruct-GPTQ"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
    )

    detail_prompt = (
        "Keep the explanation very short."
        if include_explanation
        else "No explanation required, write only code."
    )

    # construct complete prompt
    prompt_template = f"""[INST] Write bash code to solve the following coding problem that obeys the constraints. Please wrap your code answer using ```. write single line bash codes as much as possible. If multiple steps are required split them. {detail_prompt} Use markdown syntax:
    {prompt}
    [/INST]
    """

    # method 1
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
    output = model.generate(inputs=input_ids, max_new_tokens=512)
    # output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    raw_output = tokenizer.decode(output[0])
    # print(f"{raw_output=}")

    # # method 2
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     # temperature=0.7,
    #     # top_p=0.95,
    #     repetition_penalty=1.15
    # )
    # raw_output = pipe(prompt_template)[0]['generated_text']
    raw_output = raw_output.replace("</s>", "")
    raw_output = raw_output.split("[/INST]")[-1]
    # rprint(raw_output)
    code = "$ " + raw_output.split("```bash")[1].split("```")[0].strip()
    explanation = raw_output.split("```bash")[1].split("```")[1]
    explanation += (
        "\n \n Pass `--explain` to include explanation " if not include_explanation else ""
    )

    code_syntax = Syntax(code, "Arduino", theme="github-dark", line_numbers=True)

    console = Console()
    print(bashme_logo)
    console.print(code_syntax)
    console.print(Markdown(explanation))

    # parse the results
    # print the results


if __name__ == "__main__":
    # cProfile.run('main()')
    main()
