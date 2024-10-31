import os
import torch

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(".")]

pipe = pipeline('text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,
               #repetition_penalty=2.0,
                eos_token_id=terminators,)

hf_llm = HuggingFacePipeline(pipeline=pipe)

# build prompt template for simple question-answering
template = """Вопрос: {question}.

Ответ:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = prompt | hf_llm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class UserQuery(BaseModel):
    userQ: str

@app.post("/generate")
async def generate_response(query: UserQuery):
    response = llm_chain.invoke(query.userQ)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
