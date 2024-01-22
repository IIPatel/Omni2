import streamlit as st
from clarifai.client.model import Model
from clarifai.client.input import Inputs
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
import os


def encode_image(uploaded_file):
    # Read the file from the UploadedFile object
    file_bytes = uploaded_file.getvalue()
    # Encode the file bytes to base64
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    return encoded_image

#Function to analyze the O&M issue and provide a solution
def analyze_om_issue(base64_image, user_description):
    prompt = f"Analyze this image from the O&M industry, which includes the following description: '{user_description}'. Provide a detailed, practical solution to the issue depicted and described:"
   # prompt = f"Analyze the following image from an Operations and Maintenance industry, considering the user's description of the issue: '{user_description}'. Provide a detailed, professional solution, including safety precautions and step-by-step instructions:"
    inference_params = dict(temperature=0.5, image_base64=base64_image)
    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-vision"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw

def handle_ongoing_conversation(base64_image, conversation_history):
    prompt = f"Given the entire conversation history below regarding an O&M issue, along with the corresponding image, provide a relevant and context-aware response to the latest query:\n\n{conversation_history}"
    inference_params = dict(temperature=0.7, image_base64=base64_image)
    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-vision"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw
    

def generate_image(user_description):
    prompt = f"You are a professional scenery artist. Based on the below user's description and content, create a scenery without living beings: {user_description}"
    inference_params = dict(quality="standard", size="1024x1024")
    model_prediction = Model(
        f"https://clarifai.com/openai/dall-e/models/dall-e-3"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open("generated_image.png", "wb") as f:
        f.write(output_base64)
    return "generated_image.png"

def text_to_speech(input_text):
    inference_params = dict(voice="alloy", speed=1.0)
    model_prediction = Model(
        "https://clarifai.com/openai/tts/models/openai-tts-1"
    ).predict_by_bytes(
        input_text.encode(), input_type="text", inference_params=inference_params
    )
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    return audio_base64

def main():
    st.set_page_config(page_title="OMNI: AI O&M Assistant", layout="wide")
    st.title("OMNI: AI Operations and Maintenance Assistant")
    
    with st.sidebar:
        st.header("Controls")
        user_pat = st.text_input("Enter your Clarifai Personal Access Token:", type="password", help="Your Clarifai access token is required for processing")
        if user_pat:
            os.environ['CLARIFAI_PAT'] = user_pat
            clarifai_pat = os.getenv("CLARIFAI_PAT")
            
        uploaded_image = st.file_uploader("Upload an image related to your O&M issue", type=["png", "jpg", "jpeg"], help="Upload the image related to your O&M issue")
        om_issue_description = st.text_area("Describe your O&M Issue", height=100, help="Provide a detailed description of the issue for accurate analysis")
        
        analyze_btn = st.button("Analyze Issue", help="Click to analyze the uploaded image and issue description")
   
    if uploaded_image:
         # Display a preview of the uploaded image
         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
   
    if analyze_btn and uploaded_image and om_issue_description:
        with st.spinner("Analyzing O&M issue..."):
            base64_image = encode_image(uploaded_image)
            solution_text = analyze_om_issue(base64_image, om_issue_description)
            st.write(solution_text)
            st.success("Analysis generated!")
            if st.button("Convert Text to Audio"):
                    audio_base64 = text_to_speech(solution_text)
                    st.audio(audio_base64, format="audio/mp3")
                    st.success("Audio generated!")
            
            
      
             
          
if __name__ == "__main__":
    main()
