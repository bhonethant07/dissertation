# 🧠 Mental Health Companion: From Model Evaluation to Empathetic AI

This project is an end-to-end NLP pipeline designed to provide a supportive, empathetic AI companion. It covers the entire development lifecycle: from **benchmarking multiple Transformer architectures** to **fine-tuning a high-performance RoBERTa model** and deploying a **real-time Gradio chatbot** directly in Google Colab.

---

## 🚀 Project Overview

The **Mental Health Companion** analyzes user input through a multi-faceted lens to identify mental health states, sentiments, and emotions. By combining specialized classification models with a Large Language Model (LLM), it provides responses that are validating, compassionate, and contextually aware.

### Key Features
* **Mental Health Classification**: Detects states like Depression, Anxiety, Burnout, and Stress using a custom fine-tuned model.
* **Sentiment Analysis**: Gauges the overall tone (Positive, Neutral, Negative) of the user's message.
* **Emotion Detection**: Identifies and ranks specific emotions (e.g., Fear, Joy, Sadness).
* **Empathetic Synthesis**: Uses `Zephyr-7b-beta` to craft human-like responses based on the analysis results.
* **Interactive Web UI**: A clean interface built with Gradio for seamless interaction.

---

## 📊 Phase 1: Model Comparison & Benchmarking

To find the best engine for our companion, we evaluated five Transformer models on a custom mental health dataset.

### Performance & Efficiency Metrics
| Model                  | Post-Train F1 | Post-F1 (macro) | Accuracy | Inference Time (ms) | Model Size (MB) |
|:-----------------------|:--------------|:----------------|:---------|:--------------------|:----------------|
| **ModernBERT-base** | **0.9493** | **0.9491** | **94.9%**| 13.06               | 439.11          |
| roberta-base           | 0.9388        | 0.9384          | 93.8%    | 12.87               | 475.98          |
| bert-base-uncased      | 0.9374        | 0.9370          | 93.7%    | 11.88               | 438.16          |
| distilroberta-base     | 0.9329        | 0.9328          | 93.3%    | 7.15                | 316.03          |
| distilbert-base-uncased| 0.9234        | 0.9229          | 92.3%    | 5.86                | 250.77          |



**Key Insight**: While ModernBERT offered peak accuracy, **RoBERTa-base** was selected for its robust performance-to-efficiency ratio in deployment environments.

---

## 🛠 Phase 2: Training Methodology

The core classifier was developed using `roberta-base` with the following specialized techniques:
1.  **Selective Layer Freezing**: We froze the lower 70% of encoder layers to preserve foundational language understanding while allowing the upper layers to specialize in mental health nuances.
2.  **Optimized Tokenization**: Texts were processed with a `max_length` of 309, determined by the 95th percentile of our dataset.
3.  **Training Parameters**: Utilized mixed-precision (`fp16`) and stratified sampling for class balance.

---

## 🤖 Phase 3: The Chatbot Implementation

The final application orchestrates three separate NLP pipelines to provide a "smart" response:



1.  **User Input**: Message received via Gradio UI.
2.  **Multi-Model Analysis**: 
    * **Classification**: Fine-tuned RoBERTa predicts the mental health state.
    * **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest` determines the tone.
    * **Emotions**: `SamLowe/roberta-base-go_emotions` extracts specific emotional triggers.
3.  **Empathetic Output**: The LLM (`Zephyr-7b-beta`) generates a validating and compassionate response.

---

## ⚙️ How to Run in Google Colab

This project is designed to be run entirely within **Google Colab**. Follow these steps:

### 1. Load the Notebooks
* Upload the provided `.ipynb` files to your Google Drive or open them directly in Google Colab.

### 2. Enable GPU Hardware
* In the top menu, go to **Runtime** > **Change runtime type**.
* Under "Hardware accelerator," select **T4 GPU** and click **Save**.

### 3. Setup API Keys
The generative response requires a Hugging Face token:
1. Create a token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
2. In Colab, click the **🔑 (Secrets)** icon in the left sidebar.
3. Add a new secret named `HF_TOKEN` and paste your token.
4. Toggle "Notebook access" to **ON**.

### 4. Upload the Model Folder
* Click the **Files (folder icon)** in the left sidebar.
* Upload your fine-tuned model directory.
* Ensure the folder is renamed to `model` and contains files like `model.safetensors` and `config.json`.

### 5. Execute Cells
* Run the cells in order by clicking the **Play (▶️)** button on the left side of each code block. 
* The final cell in `Chatbot.ipynb` will generate a public URL to open the chatbot interface.

---

## 📁 Project Structure
* `Model_Comparison.ipynb`: Evaluation and benchmarking results.
* `Training_2_Chatbot.ipynb`: The training pipeline and preprocessing logic.
* `Chatbot.ipynb`: The primary application and Gradio interface.

---

> **Disclaimer**: This AI is intended for supportive and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
