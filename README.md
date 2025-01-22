# LocalML Introduction
Local/Offline Machine Learning Resources

This repo is intended to be catalog of local, offline, and open-source machine learning (ML) tools and approaches, for enhancing community-centered connectivity and education, particularly in underconnected regions and low-resource contexts.

If your objective is machine learning/AI without internet, on a standard consumer laptop or desktop PC, or phone, there should be useful resources for you in this repository.

We will attempt to label any closed source tools as such.   

The shared Zotero Library for this project can be found here: https://www.zotero.org/groups/5718368/localml/library

This research project is generously funded by [APC](https://www.apc.org/). 

Contributions/issues/PRs are very welcome.

CC-BY-SA-4.0


---

# Table of Contents

- [LocalML Introduction](#localml-introduction)
- [Table of Contents](#table-of-contents)
- [Local Machine Learning Considerations](#local-machine-learning-considerations)
  - [Benefits of/Reasons for Doing Local/Offline ML](#benefits-ofreasons-for-doing-localoffline-ml)
  - [Challenges of Local/Offline ML](#challenges-of-localoffline-ml)
  - [Resource requirements](#resource-requirements)
  - [Strategies for Mitigating Resource Requirements:](#strategies-for-mitigating-resource-requirements)
  - [Text-Generation on Low End PCs](#text-generation-on-low-end-pcs)
    - [Clippings related to text-generation on low-end PCs.](#clippings-related-to-text-generation-on-low-end-pcs)
  - [Image Generation on Low-End PCs](#image-generation-on-low-end-pcs)
    - [Clippings related to image-generation on low-end PCs.](#clippings-related-to-image-generation-on-low-end-pcs)
  - [Speech on Low-End PCs](#speech-on-low-end-pcs)
  - [Additional Local Machine Learning Limitations](#additional-local-machine-learning-limitations)
- [Local Machine Learning General Resources](#local-machine-learning-general-resources)
    - [Hugging Face](#hugging-face)
  - [Leaderboards](#leaderboards)
    - [Livebench](#livebench)
  - [Models](#models)
  - [Wiki Articles](#wiki-articles)
  - [Reddit Groups](#reddit-groups)
  - [GitHub Awesome Lists](#github-awesome-lists)
  - [Prompt Engineering](#prompt-engineering)
  - [AI research](#ai-research)
- [Text Generation Tools and Platforms](#text-generation-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building)
  - [Running LLM on Android Devices](#running-llm-on-android-devices)
    - [MLC LLM](#mlc-llm)
    - [Maid - Mobile Artificial Intelligence Distribution](#maid---mobile-artificial-intelligence-distribution)
    - [ChatterUI](#chatterui)
    - [smolchat Android](#smolchat-android)
  - [Running LLM on Docker](#running-llm-on-docker)
  - [Open WebUI](#open-webui)
  - [Ollama](#ollama)
  - [OpenVINO](#openvino)
  - [PrivateGPT](#privategpt)
  - [Anything LLM](#anything-llm)
  - [Ollama](#ollama-1)
  - [GPT4All](#gpt4all)
  - [koboldcpp](#koboldcpp)
  - [vllm](#vllm)
  - [AnythingLLM](#anythingllm)
  - [jan](#jan)
  - [Llama.cpp](#llamacpp)
  - [llamafile](#llamafile)
  - [smol-tools](#smol-tools)
  - [SmolLM2](#smollm2)
  - [smol-course](#smol-course)
  - [gpt4free](#gpt4free)
  - [private-gpt](#private-gpt)
  - [Open WebUI](#open-webui-1)
  - [Lobe Chat](#lobe-chat)
  - [Text generation web UI](#text-generation-web-ui)
  - [localGPT](#localgpt)
- [Text - Translation Tools and Platforms](#text---translation-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-1)
  - [InkubaLM](#inkubalm)
- [Text - RAG Tools and Platforms](#text---rag-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-2)
  - [Sort](#sort)
  - [Datasets](#datasets)
  - [WikiChat](#wikichat)
  - [Android-Document-QA](#android-document-qa)
- [Coding Tools and Platforms](#coding-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-3)
  - [Continue](#continue)
  - [Qwen2.5](#qwen25)
  - [Claude 3.5 Sonnet](#claude-35-sonnet)
- [Image Generation Tools and Platforms](#image-generation-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-4)
  - [Fooocus](#fooocus)
  - [Generative AI for Krita](#generative-ai-for-krita)
  - [SD.Next](#sdnext)
  - [Stable Diffusion web UI](#stable-diffusion-web-ui)
  - [ComfyUI](#comfyui)
- [Audio Tools and Platforms](#audio-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-5)
  - [Whisper](#whisper)
      - [How to install and use Whisper offline (no internet required)](#how-to-install-and-use-whisper-offline-no-internet-required)
  - [local-talking-llm](#local-talking-llm)
  - [Music](#music)
    - [FluxMusic](#fluxmusic)
    - [OpenMusic](#openmusic)
- [Video Generation Tools and Platforms](#video-generation-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-6)
  - [Awesome Video Diffusion](#awesome-video-diffusion)
  - [CogVideo](#cogvideo)
  - [LTX-Video](#ltx-video)
- [CAD Generation Tools and Platforms](#cad-generation-tools-and-platforms)
  - [Why might this be useful for community networking and/or capacity building?](#why-might-this-be-useful-for-community-networking-andor-capacity-building-7)
  - [Trellis](#trellis)
  - [Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts](#text2cad-generating-sequential-cad-designs-from-beginner-to-expert-level-text-prompts)



---

# Local Machine Learning Considerations



##  Benefits of/Reasons for Doing Local/Offline ML

- Resilience: Offline machine learning tools are less vulnerable to internet outages and disruptions, ensuring continued functionality in unpredictable environments.
- Data Privacy:  Processing data locally reduces reliance on cloud services, enhancing privacy and data security
- Reduced costs: Eliminating the need for constant internet access and potential cloud computing fees can significantly reduce costs


## Challenges of Local/Offline ML

- Computational Resources:  Running complex machine learning models requires significant computational power. Devices in resource-limited settings may lack the necessary processing power and memory.
- Model Size: Large pre-trained models can be difficult to deploy on devices with limited storage capacity. Techniques like model compression and quantization are crucial but come with trade-offs in accuracy
- Data Acquisition and Preprocessing: Obtaining and preparing relevant data for training and deploying models offline can be challenging without reliable internet access for data collection and cleaning.
- Model Maintenance and Updates: Keeping models up-to-date with new data and improvements can be difficult without regular internet connectivity.


## Resource requirements

1. Computational Power (CPU/GPU)

   - Text Generation: Large language models (LLMs) have billions of parameters, often requiring significant processing power. However smaller, more efficient models exist, even though they may compromise on quality or capabilities. 
   - Image Generation:  This tends to be even more computationally demanding due to the complexity of image processing. However, image generation can still be done on lower-end GPUs.

2. Memory (RAM)

   - All these tasks require loading the model into memory, which can consume gigabytes of RAM. Larger models necessitate even more memory, potentially exceeding the capacity of devices with limited RAM.

3. Storage:

   - The model files themselves can be quite large, ranging from gigabytes to tens of gigabytes. Adequate storage space is crucial for storing both the model and any datasets used for training or fine-tuning.

4. Energy Consumption:

   - Running powerful models locally can consume considerable energy, which may be a concern in areas with limited access to electricity.



## Strategies for Mitigating Resource Requirements:

- Model Compression: Techniques like pruning and quantization can reduce model size and computational demands without sacrificing too much accuracy.
- Optimization for Specific Tasks: Fine-tuning pre-trained models for specific tasks can reduce the need for massive models and lower resource requirements.
- Run the ML tool on one or more PCs and enable other machines on the network to access it/them. In this scenario, laptops and mobile devices without a powerful GPU or CPU can use ML tools via a simple user interface, with a noisy/power-hungry machine placed elsewhere in the networkm, ideally with power backup. Resources on this here: https://github.com/Wakoma/LocalML/blob/main/resources/tools.md#running-llm-on-docker




## Text-Generation on Low End PCs


Low-end Machine Tips:
- TPS: or t/s, means Tokens per second. The higher the better. 1t/s is slow, and you'll be waiting for the output. 
  - Aim for more than 10 t/s if possible.
  - LLM tokens per second (TPS) measures how quickly a Large Language Model can process text, with each token representing a word or subword
  - You can use a larger model on low-end machine, but the speed at which the LLM spits out a response will be much slower.  This is still offline though. 
- Software: If you don't know where to start, install GPT4ALL (https://gpt4all.io/index.html) or kobold (https://github.com/LostRuins/koboldcpp).
- Models: try a 7B or 8B model. If it's too slow for you, try a 3B model. 
  - Look for models that offer quantized versions. Quantization reduces the precision of the model weights, making them smaller and faster to process. The model should say something like Q8 next to the name. 
- Low/No VRAM: With little to no RAM, make sure your CPU fan is working well and does not let the processor overheat. Running LLMs  can be very demanding.
  - 8GB RAM? This will work, but you may want to save up some money to get more RAM. RAM is relatively cheap these days and getting 16GB would enable you to run models that are twice as big.  (https://www.reddit.com/r/LocalLLaMA/comments/14q5n5c/comment/jqm3cpm/)
- Other tips:
  - Process text in batches rather than individually. This can improve efficiency by allowing the CPU/GPU to process multiple sequences simultaneously.
  - Have it running in the background.  Ask the question (click "generate") then go make yourself some tea.  With a low-end PC you'll get the same reponse from a given model, it will just take longer to arrive.
  

How to check how much RAM you have:
https://www.howtogeek.com/435644/how-to-see-how-much-ram-is-in-your-pc-and-its-speed/


When is internet access required?
- Download the app/service/client
- Download the model
- Connect for occasional updates (optional)




### Clippings related to text-generation on low-end PCs. 

"You can run 13B models with 16 GB RAM but they will be slow because of CPU inference. I'd stick to 3B and 7B if you want speed. Models with more B's (more parameters) will usually be more accurate and more coherent when following instructions but they will be much slower." 

"If you can get 32/64 GB of RAM what would be the best."

"On the PC side, get any laptop with a mobile Nvidia 3xxx or 4xxx GPU, with the most GPU VRAM that you can afford. You don't want to run CPU inference on regular system RAM because it will be a lot slower."

"RAM quantity/speed is more relevant than anything else. I mean you're not going to get good performance on a literal potato because compute does matter, but you'll get a much bigger speedup going from 3200mhz DDR4 to 6000mhz DDR5 than you will from a quad core to a 12+ core CPU."

https://www.reddit.com/r/LocalLLaMA/comments/18yz3ba/best_models_for_cpu_without_gpu/


"A small model like that (8B) will run on the CPU just fine, you don't need a GPU."

https://www.reddit.com/r/LocalLLaMA/comments/1cj5um2/best_model_for_low_end_pc/

"What if I have a miniPC with 32GB of RAM?"

"I'm able to run Wizard-Vicuna-13B-GGML 8bit at around 4 tokens/s and Wizard-Vicuna-30B-GGML 4bit at .9 tokens/s on a beelink 7735hs with 32 gig."

https://www.reddit.com/r/LocalLLaMA/comments/13wnuuo/whats_the_best_local_llm_for_low_to_medium_end/


"If you intend to work on a local LlaMa often, consider a cheap laptop that remotes into a beefy PC with lots of ram and a good GPU. You can upgrade your desktop frequently and you don’t have to worry about your laptop overheating or weighing a ton or any of that."

"8gb ram with a quad core CPU for good 7B inference
Thank you, I hate these entitled posts: "Is it my 16 core CPU with newest nvidia 24GB VRAM enough to run llm?"


"4GB RAM or 2GB GPU / You will be able to run only 3B models at 4-bit, but don't expect great performance from them as they need a lot of steering to get anything really meaningful out of them. Even most phones can run these models using something like MLC."

"8GB RAM or 4GB GPU / You should be able to run 7B models at 4-bit with alright speeds, if they are llama models then using exllama on GPU will get you some alright speeds, but running on CPU only can be alright depending on your CPU. Some higher end phones can run these models at okay speeds using MLC. (Might get out of memory errors, have not tested 7B on GPU with 4GB of RAM so not entirely sure, but under Linux you might be able to just fine, but windows could work too, just not sure about memory)."

"16GB RAM or 8GB GPU / Same as above for 13B models under 4-bit except for the phone part since a very high end phone could, but never seen one running a 13B model before, though it seems possible."

"On a desktop CPU it would be 10-12 t/s, for the notebook CPU I would assume about half of that."

"I'd also mention that, if you're going the CPU-only route, you'll need a processor that supports at least the AVX instruction set. Personally, I wouldn't try with anything that doesn't also support AVX2 but if you're looking for bare minimum, that'd be any Intel Sandy Bridge or later or AMD Bulldozer or later processors. AVX2 was introduced in Haswell and Excavator architectures respectively."

https://www.reddit.com/r/LocalLLaMA/comments/15s8crb/whats_the_bare_minimum_specs_needed_for_running_ai/

---

What’s the bare minimum specs needed for running ai?

"Amplifying what many others are saying, you can run many models on just a normal PC computer without a GPU. I've been benchmarking upwards of 50 different models on my PC which is only an i5-8400 with 32 Gig of RAM. Here's a list of models and the kind of performance I'm seeing."

https://www.reddit.com/media?url=https%3A%2F%2Fpreview.redd.it%2Fwhats-the-bare-minimum-specs-needed-for-running-ai-v0-p7phjivyjiib1.png%3Fwidth%3D1103%26format%3Dpng%26auto%3Dwebp%26s%3D4ef213132fe2b5b23326ae241ca3fb5defe57390

"For reference, I've just tried running a local llama.cpp (llamafile) on a MacBook Pro from 2011 (i5 2nd gen. Sandy Bridge, 16GB DDR3-MacOS12) and got about 3tps with Phi-3-mini-4k-instruct and 2tps using Meta-Llama-3-8B-Instruct."

"I ran a 7b q4 gmml on an old laptop with 8gb RAM yesterday. 1.5t/s. Just choose the right model and you will be OK. You can go for 3b ones if you really need it."

"You can run a 3B model on a 6Gb Android phone."

"8GB RAM, any CPU, no GPU can run a 4 bit quantised 7 billion weights llm at low to usable speeds."

https://www.reddit.com/r/LocalLLaMA/comments/15s8crb/whats_the_bare_minimum_specs_needed_for_running_ai/


"you don't even need a GPU to run local models. as long as you have enough system ram, you can run them off your CPU alone."

"you can experiment with different quantization levels to reduce the size and complexity of the models. Quantization is a technique that compresses the weights of the neural network by using fewer bits to represent them. This can speed up the inference time and lower the memory usage of the models. However, it can also affect the accuracy and quality of the outputs, so you have to find a balance that works for you."

https://www.reddit.com/r/LocalLLaMA/comments/14q5n5c/any_option_for_a_low_end_pc/


"I 1,000% recommend opening it up and replacing the thermal paste between your CPU and the cooler, if you haven’t done this already."

You’re going to be putting that CPU through its paces, and it’s probably still got the original paste on there and it’s 5 years old at this point, and you wanna make sure it runs as cool as possible.

Oh, and use Linux too. It takes much less RAM for itself. Windows, even Windows 7 will hog all this memory for itself."
 
https://www.reddit.com/r/LocalLLaMA/comments/14q5n5c/comment/jqlsook/

"You can run on CPU if speed isn't a concern or if you run small models. You need to make sure you have enough ram though"

"I run Ollama on my home PC, 32 Gig of Ram, and an i9 CPU . Just recently added voice-to-txt and txt-to-voice capabilities for better interaction. It's smooth with no GPU support."

https://www.reddit.com/r/ollama/comments/1da7lqg/cheapest_way_to_run_llm_server_at_home/


"I keep seeing posts for building to a specific budget but had a thought “How cheaply could a machine possibly be built?” Of course there will be a lower boundary for model size but what are your thoughts for the least expensive way to run an LLM with no internet connection?"

"Personally, I believe mlc LLM on an android phone is the highest value per dollar option since you can technically run a 7B model for around $50-100 on a used android phone with a cracked screen."

"It's a lot cheaper to get an answer in 10 minutes than in 10 seconds."

"You can get an off-lease dell or Lenovo SFF desktop with an older i5 and 16gb RAM for under $100. It'll run rings around any ARM SBC."

"Currently running Orca Mini 3B on a Raspberry Pi 400 and quite happy with it. Managed to get Piper TTS also running along with it. All in 4 GB RAM, 2k context."

"Grab yourself a Raspberry Pi 4 with 8 GB RAM, download and compile llama.cpp and there it goes: local LLM for under 100 USD. Takes 5 mins to set up and you can use quantized 7B models at ~0.5 to 1 token/s. Yes, it's slow, painfully slow, but it works.  For larger models, merge more Pis into a MPI cluster for more RAM, but don't expect reasonable performance (here's where you will switch your wording from "tokens per second" to "seconds per token")."

https://www.reddit.com/r/LocalLLaMA/comments/16eyxgw/absolute_cheapest_local_llm/


"Just CPU. It's slow but works. I will put the prompt in and come back in a few minutes to see how things are going."

Advice on Building a GPU PC for LLM with a $1,500 Budget

https://www.reddit.com/r/LocalLLaMA/comments/1drnbq7/advice_on_building_a_gpu_pc_for_llm_with_a_1500/

"I don't know why people are dumping on you for having modest hardware.
I was using a T560 with 8GB of RAM for a while for guanaco-7B.ggmlv3.q4_K_M.bin inference, and that worked fine.
Pick out a nice 4-bit quantized 7B model, and be happy.
If it's too slow, try 4-bit quantized Marx-3B, which is smaller and thus faster, and pretty good for its size."
https://www.reddit.com/r/LocalLLaMA/comments/16imcc0/recommend_a_local_llm_for_low_spec_laptop/



## Image Generation on Low-End PCs


- Image generation "tokens per second" isn't a standard metric like it is for text LLMs. Instead, image generation speed is measured in images generated per second or time taken to generate a single image.

- **CPU:** While the CPU plays a role in managing overall processes, its impact on image generation speed is less direct compared to GPUs. It handles tasks like loading data and preprocessing but the heavy lifting of image synthesis is done by the GPU.
- **RAM:** Sufficient RAM is crucial for holding the model weights and image data during processing. Low RAM can lead to slowdowns as the system swaps data between RAM and the slower hard drive.
- **VRAM:** This is the most critical factor. Image generation models, especially those based on diffusion models, are extremely memory intensive. A dedicated GPU with ample VRAM is recommended for decent performance. 
  - 4GB VRAM min is strongly recommended
- **Smaller Models**: Opt for smaller, less complex image generation models. There are many open-source alternatives to popular large models that require significantly less VRAM.
- **Lower Resolution**: Generating images at lower resolutions (e.g., 512x512 instead of 1024x1024) drastically reduces VRAM usage and processing time.
  - You can upscale smaller images later.
- **Reduce Sampling Steps**: Diffusion models generate images through multiple sampling steps. Decreasing the number of steps can speed up generation but might result in lower quality images.
- **Batch Processing**: If you need to generate multiple images, consider batch processing. This allows the GPU to work more efficiently.
- **Software** https://github.com/rupeshs/fastsdcpu
  - More in tools.md in this repo.


### Clippings related to image-generation on low-end PCs. 

You can run A1111 all on CPU if you add all these command line arguments to the webuser-ui.bat : --use-cpu all --precision full --no-half --skip-torch-cuda-test

You can also run ComfyUI purely on CPU, just start it using the run_cpu.bat no extra steps needed.

They will be very very slow but still work. Only way to know if they will work for your system is to try them.

https://www.reddit.com/r/StableDiffusion/comments/17bzn30/running_sd_locally_without_a_gpu/

"Sure, it'll just run on the CPU and be considerably slower. Might need at least 16GB of RAM."

"I use pc with no dedicated gpu, 16 gb of RAM. It takes around 4 minutes to render 512x512 picture, 25 steps."

https://www.reddit.com/r/StableDiffusion/comments/108hsd3/can_i_run_sd_without_a_dedicated_gpu/


"I wanted to try running it on my CPU only setup without GPU to see if there's anyway that I can run without GPU but still made it so I would love to learn from your tips

My setups includes:
▸ Quad-core CPU (Intel Core i7-1165G7)
▸ 64 GB RAM
They took me ~10-15 minutes per image (512x512 resolution) and ~8 GB RAM memory usage."
https://www.redditmedia.com/r/StableDiffusion/comments/1hiee9l/anyone_running_stable_diffusion_without_gpu/



## Speech on Low-End PCs

https://www.redditmedia.com/r/LocalLLaMA/comments/1fx9lo9/speech_to_text_on_laptop_without_api_calls/


## Additional Local Machine Learning Limitations

- Bias and Fairness: The training data used to create a language model may inadvertently reflect biases present in the training set, leading to potentially discriminatory or inappropriate outputs in generated text.
  - If a language model is trained on unfiltered or biased data, it may inadvertently generate hallucinations by incorporating misinformation into its outputs.
- Hallucinations refer to incorrect or fantastical responses that a model generates instead of real, accurate information. These errors can arise for several reasons and pose significant challenges. 
  - Understanding Hallucination in LLMs: Causes, Consequences, and Mitigation Strategies: https://medium.com/@gcentulani/understanding-hallucination-in-llms-causes-consequences-and-mitigation-strategies-b5e1d0268069
  - Does Your Model Hallucinate? Tips and Tricks on How to Measure and Reduce Hallucinations in LLMs: https://deepsense.ai/blog/does-your-model-hallucinate-tips-and-tricks-on-how-to-measure-and-reduce-hallucinations-in-llms/
  - To address these challenges related to hallucinations, researchers and developers are continuously working on improving model training methods, enhancing data quality, and developing techniques to detect and correct errors in generated content. 


---

# Local Machine Learning General Resources

LocalML Zotero Library: https://www.zotero.org/groups/5718368/localml/library


### Hugging Face 

The platform where the machine learning community collaborates on models, datasets, and applications.

- https://huggingface.co/docs

- https://github.com/huggingface

- https://github.com/huggingface/transformers



## Leaderboards

### Livebench

- https://livebench.ai/#/



## Models

- https://huggingface.co/models

- https://github.com/Troyanovsky/Local-LLM-Comparison-Colab-UI

- https://www.openml.org/


## Wiki Articles

- https://en.wikipedia.org/wiki/Generative_artificial_intelligence

## Reddit Groups
- https://www.reddit.com/r/LocalLLaMA/
- https://www.reddit.com/r/MachineLearning/
- https://www.reddit.com/r/MLQuestions/

## GitHub Awesome Lists

* https://github.com/janhq/awesome-local-ai

* https://github.com/vince-lam/awesome-local-llms

* https://github.com/mahseema/awesome-ai-tools

* https://github.com/steven2358/awesome-generative-ai


## Prompt Engineering

* https://www.promptingguide.ai/

* https://github.com/dair-ai/Prompt-Engineering-Guide

* https://github.com/f/awesome-chatgpt-prompts


## AI research 

* https://github.com/youssefHosni/Awesome-AI-Data-GitHub-Repos

* https://github.com/aimerou/awesome-ai-papers

* https://github.com/DefTruth/Awesome-LLM-Inference



# Text Generation Tools and Platforms

## Why might this be useful for community networking and/or capacity building?



## Running LLM on Android Devices

### MLC LLM

MLC LLM is a machine learning compiler and high-performance deployment engine for large language models. The mission of this project is to enable everyone to develop, optimize, and deploy AI models natively on everyone's platforms. 

* https://github.com/mlc-ai/mlc-llm

* https://www.xda-developers.com/run-local-llms-smartphone/

* https://www.reddit.com/r/LocalLLaMA/comments/16t0lbw/best_model_to_run_locally_on_an_android_phone/

* https://llm.mlc.ai/docs/deploy/android.html

* https://www.kdnuggets.com/install-run-llms-locally-android-phones

* https://beebom.com/how-run-llm-locally-mlc-chat-android-phones/




### Maid - Mobile Artificial Intelligence Distribution

Maid is a cross-platform free and an open-source application for interfacing with llama.cpp models locally, and remotely with Ollama, Mistral, Google Gemini and OpenAI models remotely. Maid supports sillytavern character cards to allow you to interact with all your favorite characters. Maid supports downloading a curated list of Models in-app directly from huggingface.

* https://github.com/Mobile-Artificial-Intelligence/maid



### ChatterUI
ChatterUI is a native mobile frontend for LLMs.

Run LLMs on device or connect to various commercial or open source APIs. ChatterUI aims to provide a mobile-friendly interface with fine-grained control over chat structuring.

* https://github.com/Vali-98/ChatterUI


### smolchat Android

Project Goals

    Provide a usable user interface to interact with local SLMs (small language models) locally, on-device
    Allow users to add/remove SLMs (GGUF models) and modify their system prompts or inference parameters (temperature, min-p)
    Allow users to create specific-downstream tasks quickly and use SLMs to generate responses
    Simple, easy to understand, extensible codebase


- https://github.com/shubham0204/SmolChat-Android


## Running LLM on Docker

Discussion: https://www.reddit.com/r/LocalLLaMA/comments/17v5r0p/a_fun_day_evaluating_llm_chat_guisservers_in/

## Open WebUI
- https://github.com/open-webui/open-webui

- https://docs.openwebui.com/

## Ollama
- https://hub.docker.com/r/ollama/ollama

## OpenVINO
- https://docs.openvino.ai/2024/index.html

## PrivateGPT
- https://github.com/zylon-ai/private-gpt



## Anything LLM
- - https://github.com/Mintplex-Labs/anything-llm



## Ollama

ollama.com

- https://github.com/ollama/ollama

Get up and running with Llama 3.2, Mistral, Gemma 2, and other large language models. 


Run LLMs locally without internet with Ollama
- https://medium.com/@pratikgtm/run-llms-locally-without-internet-with-ollama-1305ee83ceb7


## GPT4All

- nomic.ai/gpt4all

- https://github.com/nomic-ai/gpt4all

 GPT4All runs large language models (LLMs) privately on everyday desktops & laptops.

No API calls or GPUs required - you can just download the application and get started. 


## koboldcpp

KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models, inspired by the original KoboldAI. It's a single self-contained distributable from Concedo, that builds off llama.cpp, and adds a versatile KoboldAI API endpoint, additional format support, Stable Diffusion image generation, speech-to-text, backward compatibility, as well as a fancy UI with persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything KoboldAI and KoboldAI Lite have to offer.


- https://github.com/LostRuins/koboldcpp


## vllm

vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

    State-of-the-art serving throughput
    Efficient management of attention key and value memory with PagedAttention
    Continuous batching of incoming requests
    Fast model execution with CUDA/HIP graph
    Quantizations: GPTQ, AWQ, INT4, INT8, and FP8.
    Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    Speculative decoding
    Chunked prefill

- https://github.com/vllm-project/vllm


## AnythingLLM

AnythingLLM: The all-in-one AI app you were looking for.
Chat with your docs, use AI Agents, hyper-configurable, multi-user, & no frustrating set up required. 

- https://github.com/Mintplex-Labs/anything-llm

anythingllm.com


## jan

Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM) 

- https://github.com/janhq/jan

jan.ai/

## Llama.cpp

- https://github.com/ggerganov/llama.cpp

Inference of Meta's LLaMA model (and others) in pure C/C++

The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware - locally and in the cloud.

    Plain C/C++ implementation without any dependencies
    Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
    AVX, AVX2, AVX512 and AMX support for x86 architectures
    1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
    Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads MTT GPUs via MUSA)
    Vulkan and SYCL backend support
    CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

Since its inception, the project has improved significantly thanks to many contributions. It is the main playground for developing new features for the ggml library.

## llamafile

llamafile lets you distribute and run LLMs with a single file. 

- https://github.com/Mozilla-Ocho/llamafile



## smol-tools

A collection of lightweight AI-powered tools built with LLaMA.cpp and small language models. These tools are designed to run locally on your machine without requiring expensive GPU resources. They can also run offline, without any internet connection.

- https://github.com/huggingface/smollm/blob/main/smol_tools/README.md


## SmolLM2


SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.

- https://github.com/huggingface/smollm/tree/main


## smol-course

This is a practical course on aligning language models for your specific use case. It's a handy way to get started with aligning language models, because everything runs on most local machines. There are minimal GPU requirements and no paid services. The course is based on the SmolLM2 series of models, but you can transfer the skills you learn here to larger models or other small language models.

- https://github.com/huggingface/smol-course

Why Small Language Models?

While large language models have shown impressive capabilities, they often require significant computational resources and can be overkill for focused applications. Small language models offer several advantages for domain-specific applications:

    Efficiency: Require significantly less computational resources to train and deploy
    Customization: Easier to fine-tune and adapt to specific domains
    Control: Better understanding and control of model behavior
    Cost: Lower operational costs for training and inference
    Privacy: Can be run locally without sending data to external APIs
    Green Technology: Advocates efficient usage of resources with reduced carbon footprint
    Easier Academic Research Development: Provides an easy starter for academic research with cutting-edge LLMs with less logistical constraints



## gpt4free

The official gpt4free repository | various collection of powerful language models 

* https://g4f.ai

* https://github.com/xtekky/gpt4free



## private-gpt

PrivateGPT is a production-ready AI project that allows you to ask questions about your documents using the power of Large Language Models (LLMs), even in scenarios without an Internet connection. 100% private, no data leaves your execution environment at any point.

- https://github.com/zylon-ai/private-gpt

 privategpt.dev 

 ## Open WebUI

 Open WebUI is an extensible, feature-rich, and user-friendly self-hosted WebUI designed to operate entirely offline. It supports various LLM runners, including Ollama and OpenAI-compatible APIs. For more information, be sure to check out our Open WebUI Documentation.

 - https://github.com/open-webui/open-webui

  openwebui.com 


## Lobe Chat

An open-source, modern-design ChatGPT/LLMs UI/Framework.
Supports speech-synthesis, multi-modal, and extensible (function call) plugin system.

- https://github.com/lobehub/lobe-chat

 chat-preview.lobehub.com 

 ## Text generation web UI

A Gradio web UI for Large Language Models.

Its goal is to become the AUTOMATIC1111/stable-diffusion-webui of text generation.

- https://github.com/oobabooga/text-generation-webui



## localGPT

LocalGPT is an open-source initiative that allows you to converse with your documents without compromising your privacy. With everything running locally, you can be assured that no data ever leaves your computer. Dive into the world of secure, local document interactions with LocalGPT.

- https://github.com/PromtEngineer/localGPT




# Text - Translation Tools and Platforms

## Why might this be useful for community networking and/or capacity building?


## InkubaLM

Africa’s first multilingual AI large language model, aimed at supporting and enhancing low-resource African languages. The model is starting with Swahili, Yoruba, IsiXhosa, Hausa, and isiZulu. This model is named after the dung beetle for its efficient design and seeks to address the digital underrepresentation of these languages by providing tools for translation, transcription, and various natural language processing tasks. InkubaLM is designed to be robust yet compact, leveraging two datasets—Inkuba-Mono and Inkuba-Instruct—to pre-train and enhance the model’s capabilities across the five selected languages. 

Lelapa AI is committed to linguistic diversity and digital inclusivity by offering open access to the model and its resources. By providing tools and datasets that facilitate the development of digital solutions, Lelapa AI aims to empower African communities and ensure that their languages are better represented in the digital space. This approach not only supports language preservation but also strives to make advanced AI technologies more accessible and relevant to users in Africa. 

“Our language model is not just a technological achievement; it is a step towards greater linguistic equality and cultural preservation,” said Atnafu Tonja, fundamental research lead at Lelapa AI.  

* https://arxiv.org/html/2408.17024v1

* https://medium.com/@lelapa_ai/inkubalm-a-small-language-model-for-low-resource-african-languages-dc9793842dec

* https://huggingface.co/lelapa/InkubaLM-0.4B




# Text - RAG Tools and Platforms

## Why might this be useful for community networking and/or capacity building?



## Sort

https://medium.com/@nydas/building-an-offline-rag-chatbot-with-custom-frontend-and-websockets-3a739878adf1

https://github.com/jonfairbanks/local-rag




## Datasets

* https://www.reddit.com/r/LocalLLaMA/comments/1bjlzna/whats_the_fastest_route_to_success_to_performing/

* https://huggingface.co/datasets/legacy-datasets/wikipedia


## WikiChat

Large language model (LLM) chatbots like ChatGPT and GPT-4 get things wrong a lot, especially if the information you are looking for is recent ("Tell me about the 2024 Super Bowl.") or about less popular topics ("What are some good movies to watch from [insert your favorite foreign director]?"). WikiChat uses Wikipedia and the following 7-stage pipeline to makes sure its responses are factual. Each numbered stage involves one or more LLM calls.

- https://github.com/stanford-oval/WikiChat



##  Android-Document-QA

A simple Android app that allows the user to add a PDF/DOCX document and ask natural-language questions whose answers are generated by the means of an LLM


- - https://github.com/shubham0204/Android-Document-QA




# Coding Tools and Platforms

## Why might this be useful for community networking and/or capacity building?

- In underconnected communities without access to resources such as stackoverflow, offline ML coding tools could be use for software development and education. 

- AI-assisted coding tools can empower local citizen scientists to analyze data collected offline (e.g., weather patterns, agricultural yields, health trends) and develop solutions using generated code.



What's the current best way to use AI coding assistants in VSCode?:
- https://www.redditmedia.com/r/vscode/comments/1fpqzh9/whats_the_current_best_way_to_use_ai_coding/


## Continue

https://github.com/continuedev/continue

"Continue is the leading open-source AI code assistant. You can connect any models and any context to build custom autocomplete and chat experiences inside VS Code and JetBrains".


## Qwen2.5

Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. 

* https://huggingface.co/Qwen/Qwen2.5-72B-Instruct

* https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct

* https://github.com/QwenLM/Qwen2.5

* https://arxiv.org/pdf/2409.12186

* https://ollama.com/library/qwen2.5-coder

* https://venturebeat.com/ai/alibaba-new-ai-can-code-in-92-languages-and-its-completely-free/

* https://www.reddit.com/r/LocalLLaMA/comments/1h7nsg2/are_you_still_happy_with_qwen25_coder_32b/



## Claude 3.5 Sonnet

- https://www.anthropic.com/news/3-5-models-and-computer-use


# Image Generation Tools and Platforms

## Why might this be useful for community networking and/or capacity building?

- Imagine a community wanting to build its own Wi-Fi network using readily available materials. AI image generation could translate technical instructions into clear visual guides. Complex steps like wiring hardware, configuring antennas, or troubleshooting connections could be illustrated through easily understandable diagrams and animations.
- Communities can create localized learning materials by generating visuals for topics ranging from basic computer literacy to advanced networking concepts. This empowers individuals to acquire technical skills independently, fostering self-reliance and innovation.
- Communities can use AI to generate visualizations of potential connectivity infrastructure based on their specific geography and needs. This helps them plan strategically, identify optimal locations for access points
  


## Fooocus

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images. Fooocus has also simplified the installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

- https://github.com/lllyasviel/Fooocus


## Generative AI for Krita

- https://github.com/Acly/krita-ai-diffusion

This is a plugin to use generative AI in image painting and editing workflows from within Krita. For a more visual introduction, see www.interstice.cloud

The main goals of this project are:

    Precision and Control. Creating entire images from text can be unpredictable. To get the result you envision, you can restrict generation to selections, refine existing content with a variable degree of strength, focus text on image regions, and guide generation with reference images, sketches, line art, depth maps, and more.
    Workflow Integration. Most image generation tools focus heavily on AI parameters. This project aims to be an unobtrusive tool that integrates and synergizes with image editing workflows in Krita. Draw, paint, edit and generate seamlessly without worrying about resolution and technical details.
    Local, Open, Free. We are committed to open source models. Customize presets, bring your own models, and run everything local on your hardware. Cloud generation is also available to get started quickly without heavy investment.



## SD.Next

- https://github.com/vladmandic/automatic

SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models 


## Stable Diffusion web UI


A web interface for Stable Diffusion, implemented using Gradio library.

- https://github.com/AUTOMATIC1111/stable-diffusion-webui


## ComfyUI

The most powerful and modular diffusion model GUI and backend.

- https://github.com/comfyanonymous/ComfyUI




# Audio Tools and Platforms

## Why might this be useful for community networking and/or capacity building?




## Whisper

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

* https://github.com/openai/whisper

* https://arxiv.org/pdf/2212.04356


#### How to install and use Whisper offline (no internet required)

Purpose: These instructions cover the steps not explicitly set out on the main Whisper page, e.g. for those who have never used python code/apps before and do not have the prerequisite software already installed.

* https://github.com/openai/whisper/discussions/1463

* https://github.com/Purfview/whisper-standalone-win



## local-talking-llm

After my latest post about how to build your own RAG and run it locally. Today, we're taking it a step further by not only implementing the conversational abilities of large language models but also adding listening and speaking capabilities. The idea is straightforward: we are going to create a voice assistant reminiscent of Jarvis or Friday from the iconic Iron Man movies, which can operate offline on your computer. Since this is an introductory tutorial, I will implement it in Python and keep it simple enough for beginners. Lastly, I will provide some guidance on how to scale the application. 

- https://github.com/vndee/local-talking-llm



## Music

### FluxMusic

- https://github.com/feizc/FluxMusic

Text-to-Music Generation with Rectified Flow Transformers 


### OpenMusic

OpenMusic: SOTA Text-to-music (TTM) Generation 

- https://github.com/ivcylc/qa-mdt



# Video Generation Tools and Platforms

## Why might this be useful for community networking and/or capacity building?



## Awesome Video Diffusion

A curated list of recent diffusion models for video generation, editing, restoration, understanding, nerf, etc.

- https://github.com/showlab/Awesome-Video-Diffusion


## CogVideo
text and image to video generation

* https://github.com/THUDM/CogVideo

* Demo: https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space


## LTX-Video

LTX-Video is the first DiT-based video generation model that can generate high-quality videos in real-time. It can generate 24 FPS videos at 768x512 resolution, faster than it takes to watch them. The model is trained on a large-scale dataset of diverse videos and can generate high-resolution videos with realistic and diverse content.

- https://github.com/Lightricks/LTX-Video



# CAD Generation Tools and Platforms

## Why might this be useful for community networking and/or capacity building?

- Users without technical expertise in 3D printing/additive manufacturing would be able to type in a prompt or use a reference image to generate a 3D model which could be printed/built locally. For example one could type the dimensions of a mount or bracket, or hardware case, to produce a CAD model for local production.

- Individuals without specialized training can leverage these tools to create customized designs for 3D printing, potentially leading to local production of connectivity-related tools, educational models, and more. This fosters self-reliance and innovation at a grassroots level.

- On-demand repairs facilitated by AI-generated instructions.


## Trellis

* https://trellis3d.github.io/

* https://github.com/Microsoft/TRELLIS

* https://huggingface.co/spaces/JeffreyXiang/TRELLIS


TRELLIS is a large 3D asset generation model. It takes in text or image prompts and generates high-quality 3D assets in various formats, such as Radiance Fields, 3D Gaussians, and meshes. The cornerstone of TRELLIS is a unified Structured LATent (SLAT) representation that allows decoding to different output formats and Rectified Flow Transformers tailored for SLAT as the powerful backbones. We provide large-scale pre-trained models with up to 2 billion parameters on a large 3D asset dataset of 500K diverse objects. TRELLIS significantly surpasses existing methods, including recent ones at similar scales, and showcases flexible output format selection and local 3D editing capabilities which were not offered by previous models.


## Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts
- https://sadilkhan.github.io/text2cad-project/


