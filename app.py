import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './assistant-1_8b'
os.system(f'git clone https://code.openxlab.org.cn/mingyanglee/assistant-1_8b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, device_map='auto', torch_dtype=torch.float16) # .cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.8,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="mingyanglee_assistant 1.8B",
                description="""
mingyanglee\'s assistant based on InternLM2 1.8B mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
