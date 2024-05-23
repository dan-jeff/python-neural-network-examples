import torch
from transformers import pipeline
import os
import re
from pathlib import Path
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os.path as path

#Langchain modules
from langchain import document_loaders as dl
from langchain import embeddings
from langchain import text_splitter as ts
from langchain import vectorstores as vs
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import PromptTemplate
from operator import itemgetter
#Torch + transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
#Other useful modules
import re
import time

# RAG
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.directory import DirectoryLoader

# Demo URL: huggingface.co/spaces/HuggingFaceH4/zephyr-chat
# RAG URL: https://medium.com/aimonks/exploring-offline-rag-with-langchain-zephyr-7b-beta-and-decilm-7b-c0626e09ee1f

#A document about quantum computing
#document_path ="/home/dan/Documents/git/moq-helpers/"
document_path = '/mnt/504A68D84A68BBFA/My Files/Documents/University'

def get_documents(rootdir: str):
    loader = DirectoryLoader(rootdir, "**/*.docx",True)
    documents = loader.load()
    return documents

#we set default chunk size of 500 characters with an overlap of 20 characters
def split_doc(document_path, chunk_size=500, chunk_overlap=20):
    #loader = dl.TextLoader(document_path) # Other loaders available: PyPDFLoader
    document = get_documents(document_path)#loader.load()
    text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_splitted = text_splitter.split_documents(documents=document)
    return document_splitted

if not path.exists('sentence-transformers'):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    #Save the model locally
    model.save('sentence-transformers')
    #release memory (RAM + cache)
    del model
    torch.cuda.empty_cache()

def load_embedding_model():
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model_instance = embeddings.HuggingFaceEmbeddings(
        #Foldername where the model was stored
        model_name="sentence-transformers",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding_model_instance

#Instantiate the embedding model
embedding_model_instance = load_embedding_model()

def create_db(document_splitted, embedding_model_instance):

    model_vectorstore = vs.FAISS
    db=None
    try:
        content = []
        metadata = []
        for d in document_splitted:
            content.append(d.page_content)
            metadata.append({'source': d.metadata})
        db=model_vectorstore.from_texts(content, embedding_model_instance, metadata)
    except Exception as error:
        print(error)
    return db


db = None
print('Building DB')
if True:#not path.exists('db.index'):
    #Split the document and print the different chunks
    document_splitted = split_doc(document_path)
    #for doc in document_splitted:
    #  print(doc)
    db = create_db(document_splitted, embedding_model_instance)
    #store the db locally for future use
    #db.save_local('db.index')

# 4 bit using bitsandbytes
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

print('Building Zephyr')
if not path.exists('zephyr-7b-beta-tokenizer'):
    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map="auto")
    
    #tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    #model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model.save_pretrained('zephyr-7b-beta-model', max_shard_size="1000MB")
    tokenizer.save_pretrained('zephyr-7b-beta-tokenizer')
    del model
    del tokenizer
    torch.cuda.empty_cache()

print('Loading Models')
tokenizer = AutoTokenizer.from_pretrained("zephyr-7b-beta-tokenizer")
#model = AutoModelForCausalLM.from_pretrained("zephyr-7b-beta-model", low_cpu_mem_usage=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("zephyr-7b-beta-model", quantization_config=nf4_config, device_map="auto")
pipe = pipeline(task="text-generation", model=model,tokenizer=tokenizer, max_new_tokens=1000)#removing , device="cuda:0"
llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.2})# 0 for truth (no creativity)

query = "What are the functional requirements in NET302?"
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 6, 'score_threshold': 0.01})
retrieved_docs = retriever.get_relevant_documents(query)

template = """Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#First chain to query the LLM
rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

#Second chain to postprocess the answer
rag_chain_with_source = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

t0=time.time()
resp = rag_chain_with_source.invoke(query)
if len(resp['documents'])==0:
  print('No documents found')
else:
  stripped_resp = re.sub(r"\n+$", " ", resp['answer'])
  print(stripped_resp)
  print('Sources',resp['documents'])
  print('Response time:', time.time()-t0)
  
exit()





torch.cuda.set_per_process_memory_fraction(0.99)  # Adjust the fraction as needed
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def read_file_as_string(filename):
    """
    Reads all lines from a file and returns them as a string.
    
    Args:
      filename: str -- File name to be read.
    
    Returns:
      str -- Content of file as a string.
    """
    with open(filename, 'r') as fileobj:
        return '\n'.join(fileobj.readlines())
    
def get_concatenated_text(path):
    """
    Concatenates all text from given path and its subdirectories into a single string.
    
    Args:
      path: str -- Directory path to search files recursively.
    
    Returns:
      str -- String containing all text found in files under the provided path.
    """
    result = ""
    for dirpath, _, filenames in os.walk(path):
        # Skip current and parent directories
        if dirpath == path or dirpath == os.path.dirname(path):
            continue
        
        for filename in filenames:
            fullname = os.path.join(dirpath, filename)
            
            if (os.path.splitext(filename)[1][1:] != '.cs'):
                continue
            
            #if (fullname == '/home/dan/Documents/git/moq-helpers/README.md'):
            #    continue
            
            with open(fullname, 'r') as f:
                content = f.read()
                
                # Remove trailing whitespace characters
                content = re.sub('[\s]+$', '', content)
            
                result += content
    
    return result

# Traditional pipe, uses higher memory than 4bit alternative
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

# 4 bit using bitsandbytes
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'HuggingFaceH4/zephyr-7b-beta'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
context = get_concatenated_text('/home/dan/Documents/git/moq-helpers/FunctionalDev.MoqHelpers/FunctionalDev.MoqHelpers')
#question = f'''Does the following readme markdown file accurately represent what the library does and what changes would you make?
#{read_file_as_string('/home/dan/Documents/git/moq-helpers/README.md')}
#'''

def get_message(question, context):
    return [
    {
        "role": "user", 
        "content": f'''
Answer the following question using the provided context, and if the answer isn't
contained in the context, say "I don't know:"
Context: {context} # Insert text to be searched for an answer
Question: {question}
Answer:
'''
    }
    ]

def execute(messages, temperature = 0.7):
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')#, add_generation_prompt=True)
    ids = prompt.to('cuda')
    outputs = model.generate(ids, max_new_tokens=128, temperature=temperature, top_k=50, top_p=0.95)
    print(tokenizer.decode(outputs[0]))
    #return outputs[0]["generated_text"]

execute(get_message('What does the library do as described by the readme given as context?', read_file_as_string('/home/dan/Documents/git/moq-helpers/README.md')))
#execute(get_message('Why is the sky blue?'));

#i = 0
#while i < 5:
#    i+=1
    
    
    

#question: ''



#messages = [
#    {
#        "role": "system",
#        "content": "You are a friendly chatbot who always responds in the style of a pirate",
#    },
#    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#]








## messages = [
#    {
#        "role": "user", 
#        "content": f'''
#Answer the following question using the provided context, and if the answer isn't
#contained in the context, say "I don't know:
#Context: {context}
#Question: {questio