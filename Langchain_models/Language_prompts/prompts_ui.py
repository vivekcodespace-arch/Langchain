from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="facebook/opt-350m", device=-1)

generator = load_model()

response = generator('How are you doing',
                     max_new_tokens = 10,
                     )

print(response[0]['generated_text'])