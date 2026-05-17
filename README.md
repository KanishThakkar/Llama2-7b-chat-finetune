# Llama 2 Fine-Tuning using QLoRA and LoRA

This project demonstrates parameter-efficient fine-tuning (PEFT) of a pretrained Llama 2 model using **QLoRA**, **LoRA adapters**, and the **Hugging Face ecosystem**.

The notebook covers the complete workflow from loading datasets to training and saving fine-tuned adapters while optimizing GPU memory usage.

---

## Features

* Fine-tuning pretrained **Llama 2**
* 4-bit quantization using **BitsAndBytes (QLoRA)**
* Parameter-efficient fine-tuning with **LoRA**
* Dataset preprocessing and tokenization
* Training using **SFTTrainer**
* Memory optimization for limited GPUs
* Saving LoRA adapters and optional model merging
* 
---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* TRL
* Datasets
* BitsAndBytes
* Accelerate

---

## Workflow

Install libraries
↓
Load dataset
↓
Configure 4-bit quantization
↓
Load pretrained Llama 2 model
↓
Load tokenizer
↓
Configure LoRA parameters
↓
Set training arguments
↓
Create SFTTrainer
↓
Train model
↓
Save LoRA adapters
↓
(Optional) Merge adapters with base model
↓
Save / Upload model

---

## Key Concepts Used

### QLoRA

Uses 4-bit quantization to reduce GPU memory usage during fine-tuning.

### LoRA (Low-Rank Adaptation)

Trains small adapter matrices instead of updating all model parameters.

### PEFT

Parameter-Efficient Fine-Tuning techniques used for efficient LLM adaptation.

### SFT (Supervised Fine-Tuning)

Fine-tuning the model on instruction-response datasets.

---

## Challenges Faced

During implementation I encountered and solved issues such as:

* CUDA Out Of Memory errors
* `bitsandbytes` installation issues
* TRL / Transformers version mismatches
* Hugging Face upload configuration
* GPU memory optimization

---

## Learning Outcome

Through this project I learned:

* LLM fine-tuning workflow
* LoRA and QLoRA concepts
* Quantization techniques
* Hugging Face ecosystem
* Memory-efficient training strategies
* Debugging LLM training pipelines

---

## Future Improvements

* Fine-tune on custom datasets
* Experiment with larger models
* Add inference notebook
* Deploy model using Hugging Face Spaces

---

## Author

Kanish Thakkar

