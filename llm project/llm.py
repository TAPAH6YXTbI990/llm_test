import streamlit as st

from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

st.title('Test os llm (ai-forever/rugpt3small_based_on_gpt2)')

hf_api_key = st.sidebar.text_input('HuggingFace key')

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

def generate_response(input_text):
  st.info(llm_chain.invoke(input_text))

with st.form('my_form'):
  text = st.text_area('Введите текст:', 'Сколько пальцев на руке?')
  submitted = st.form_submit_button('Ввод')
  if not hf_api_key.startswith('hf_'):
    st.warning('Please enter your HuggingFace key!', icon='⚠')
  if submitted and hf_api_key.startswith('hf_'):
    generate_response(text)
