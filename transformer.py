import streamlit as st
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from safetensors.torch import load_file
import torch
import os

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model_path = '/Users/rashmibaghel/learning_om/output/checkpoint-3'  # Replace with your actual path
    config_path = os.path.join(model_path, 'config.json')
    model_weights_path = os.path.join(model_path, 'model.safetensors')

    # Load Configuration
    config = GPT2Config.from_json_file(config_path)

    # Instantiate Model
    model = GPT2LMHeadModel(config)

    # Load Model Weights
    try:
        model.load_state_dict(load_file(model_weights_path), strict=False)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# --- Web App Interface ---
st.title("Text Generation with Your Fine-Tuned GPT-2 Model")

prompt = st.text_area("Enter your prompt here:", "Thinking about the future of")

# Generation Parameters (you can expose more if you like)
temperature = st.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
num_return_sequences = st.slider("Number of Generations:", min_value=1, max_value=5, value=3, step=1)
max_length = st.slider("Max Length:", min_value=50, max_value=300, value=200, step=50)

if st.button("Generate Text"):
    if prompt and model is not None and tokenizer is not None:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )

        st.subheader("Generated Text:")
        for i, sequence in enumerate(output_sequences):
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
            st.write(f"**Generation {i+1}:**")
            st.write(generated_text)
            st.markdown("---")
    else:
        st.warning("Please enter a prompt and ensure the model and tokenizer are loaded.")