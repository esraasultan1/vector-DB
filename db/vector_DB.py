from pinecone.core.openapi.control.models import Embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel 
from transformers import BlipProcessor, BlipForConditionalGeneration


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

#print(embeddings)


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

#print(results)




 #######################image#############################
#Load CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#define image data
imagedata=[
    
    {"id":"img1","path":r"C:\Users\zigzag\Desktop\z1.jpg","description":"A photo of a cat"},
    {"id":"img2","path":r"C:\Users\zigzag\Desktop\z2.jpg","description":"A photo of a cat"},
    {"id":"img3","path":r"C:\Users\zigzag\Desktop\d1.jpg","description":"A photo of a dog"},
    {"id":"img4","path":r"C:\Users\zigzag\Desktop\d2.jpg","description":"A photo of a dog"}
    ]

 #Convert image data into embeddings
embeddinggs = []
for item in imagedata:
    image = Image.open(item['path'])  
    inputs = processor(images=image, return_tensors="pt", do_center_crop=True, do_resize=True)
    
    #Generate image embedding
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)  
    
     #praper image for upsert
    embeddinggs.append({
        "id": item["id"],
        "values": image_embedding[0].cpu().numpy().tolist(),
        "metadata": {"description": item["description"], "path": item["path"]}
    })

#create an index
index_name="my-image-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
    
    #wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
    
    # get my index to store the vector embeddings
indexx = pc.Index("my-image-index")

#updert image embeddings into the index
indexx.upsert(vectors=embeddinggs, namespace="image-namespace")

#print(f"Uploaded {len(embeddinggs)} image embeddings to Pinecone.")

# Define a query 
query_image_path = r"C:\Users\zigzag\Desktop\d3.jpg"
query_image = Image.open(query_image_path)
query_inputs = processor(images=query_image, return_tensors="pt", do_center_crop=True, do_resize=True)

with torch.no_grad():
    query_embedding = model.get_image_features(**query_inputs)

#Search for the most similar images in Pinecone
resultss = indexx.query(
    namespace="image-namespace",
    vector=query_embedding[0].cpu().numpy().tolist(),
    top_k=2,
    include_values=False,
    include_metadata=True
)

print("Search Results:")
for match in resultss["matches"]:
    print(f"ID: {match['id']}, Description: {match['metadata']['description']}")
    

   ##########add description###################
   
# Load the BLIP model and processor for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the CLIP model and processor for image embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


# Define a dataset of images
dataa = [
    {"id": "imgg1", "path":r"C:\Users\zigzag\Desktop\space1.jpg" },
    {"id": "imgg2", "path":r"C:\Users\zigzag\Desktop\f1.jpg"},
    {"id": "imgg3", "path":r"C:\Users\zigzag\Desktop\car1.jpg" },
]

# Process images  embeddings + descriptions
vectors = [] 
for item in dataa:
    # Load the image
    image = Image.open(item["path"]).convert("RGB")
    
     # Generate description for the image using BLIP
    blip_inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = blip_model.generate(**blip_inputs)
        description = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    
    # Generate the image embedding using CLIP
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**clip_inputs)
        image_embedding = image_embedding[0].cpu().numpy()  # Convert to NumPy array

    # Add the embedding and metadata to the vectors list
    vectors.append({
        "id": item["id"],  # Unique identifier for each image
        "values": image_embedding.tolist(),  # Embedding vector
        "metadata": {"description": description}  # Auto-generated metadata
    })

#Print the results
for vector in vectors:
    print(f"ID: {vector['id']}")
    print(f"Description: {vector['metadata']['description']}")
    print(f"Embedding (first 5 values): {vector['values'][:5]}")


   

    












    


















    

