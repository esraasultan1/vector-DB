from pinecone.core.openapi.control.models import Embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

#get api-key from pinecone
pc = Pinecone(api_key="pcsk_46bj5t_3Vk7n9qmesN2xqDdTNg959wALkGeJXrdWM4FNKU4hJmv8mUnPnY1395ibKSBokc")


#define a dataset
data=[
    
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

#convert the row data into embedding vector
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

print(embeddings)


#create an index
index_name="my-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
    
    
    #wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
      
  #get my index to store embedding vector
    index = pc.Index("my-index")
    
    # get my index to store the vector embeddings
index = pc.Index("my-index")

# Prepare the records for upsert
records = []
for d, e in zip(data, embeddings):
    records.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="my-namespace"
)

  
# Define  query
query = "Tell me about the tech company known as Apple."

# Convert the query into emdedding vector
query_embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

# Search the index for the three most similar vectors
results = index.query(
    namespace="my-namespace",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)















    

