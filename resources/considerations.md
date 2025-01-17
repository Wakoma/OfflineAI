


# Benefits of/Reasons for Doing Local/Offline ML

- Resilience: Offline systems are less vulnerable to internet outages and disruptions, ensuring continued functionality in unpredictable environments.
- Data Privacy:  Processing data locally reduces reliance on cloud services, enhancing privacy and data security
- Reduced costs: Eliminating the need for constant internet access and potential cloud computing fees can significantly reduce costs


# Challenges of Local/Offline ML

- Computational Resources:  Running complex machine learning models requires significant computational power. Devices in resource-limited settings may lack the necessary processing power and memory.
- Model Size: Large pre-trained models can be difficult to deploy on devices with limited storage capacity. Techniques like model compression and quantization are crucial but come with trade-offs in accuracy
- Data Acquisition and Preprocessing: Obtaining and preparing relevant data for training and deploying models offline can be challenging without reliable internet access for data collection and cleaning.
- Model Maintenance and Updates: Keeping models up-to-date with new data and improvements can be difficult without regular internet connectivity.


# Resource requirements

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




# Running LLMs Low End PCs

## tl:dr

Low-end Machine Tips:
- TPS: or t/s, means Tokens per second. The higher the better. 1t/s is slow, and you'll be waiting for the output. 
  - Aim for more than 10 t/s if possible.
  - LLM tokens per second (TPS) measures how quickly a Large Language Model can process text, with each token representing a word or subword
  - You can use a larger model on low-end machine, but the speed at which the LLM spits out a response will be much slower.  This is still offline though. 
- Software: If you don't know where to start, install GPT4ALL (https://gpt4all.io/index.html) or kobold.
- Models: try a 7B or 8B model. If it's too slow for you, try a 3B model. 
  - Look for models that offer quantized versions. Quantization reduces the precision of the model weights, making them smaller and faster to process.
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

## RAG





## Clippings

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

"If your talking absolute BARE minimum, I can give you a few tiers of minimums starting at lowest of low system requirements."

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




----



# Image Generation on Low-End PCs



## tl:dr

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



## Clippings

You can run A1111 all on CPU if you add all these command line arguments to the webuser-ui.bat : --use-cpu all --precision full --no-half --skip-torch-cuda-test

You can also run ComfyUI purely on CPU, just start it using the run_cpu.bat no extra steps needed.

They will be very very slow but still work. Only way to know if they will work for your system is to try them.

https://www.reddit.com/r/StableDiffusion/comments/17bzn30/running_sd_locally_without_a_gpu/

"Sure, it'll just run on the CPU and be considerably slower. Might need at least 16GB of RAM."

"I use pc with no dedicated gpu, 16 gb of RAM. It takes around 4 minutes to render 512x512 picture, 25 steps."

https://www.reddit.com/r/StableDiffusion/comments/108hsd3/can_i_run_sd_without_a_dedicated_gpu/


I wanted to try running it on my CPU only setup without GPU to see if there's anyway that I can run without GPU but still made it so I would love to learn from your tips

My setups includes:
▸ Quad-core CPU (Intel Core i7-1165G7)
▸ 64 GB RAM
They took me ~10-15 minutes per image (512x512 resolution) and ~8 GB RAM memory usage.
https://www.redditmedia.com/r/StableDiffusion/comments/1hiee9l/anyone_running_stable_diffusion_without_gpu/


----

# Audio on Low-End PCs

https://www.redditmedia.com/r/LocalLLaMA/comments/1fx9lo9/speech_to_text_on_laptop_without_api_calls/


