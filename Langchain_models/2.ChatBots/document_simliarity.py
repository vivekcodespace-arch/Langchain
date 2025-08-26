from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The capital of India is New Delhi.",
    "Cricket is the most popular sport in India.",
    "Artificial Intelligence is transforming industries worldwide.",
    "The Great Wall of China is a UNESCO World Heritage site.",
    "Mahatma Gandhi led India’s non-violent freedom movement.",
    "Python is a widely used programming language for AI and data science.",
    "The Eiffel Tower is one of the most famous landmarks in Paris.",
    "Electric vehicles are becoming more popular due to sustainability.",
    "Water is essential for all forms of life on Earth.",
    "The Internet has revolutionized how people communicate and learn.",
    "Mount Everest is the highest peak in the world.",
    "The Taj Mahal was built by Shah Jahan in memory of his wife.",
    "Amazon Rainforest is known as the lungs of the Earth.",
    "SpaceX is working on reusable rockets for space travel.",
    "The human brain contains around 86 billion neurons."
]
doc_embedding = embeddings.embed_documents(documents)

query = "Who built Taj Mahal?"

query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)
best_doc_index = scores[0].argmax()

print(f"Query: {query}")
print(f"Best matched document: {documents[best_doc_index]}")

