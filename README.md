# LocalML
Local/Offline Machine Learning Resources

This is intended to be catalog of open source ML tools and approaches, along with their potential applications for community networking and/or capacity building. We will attempt to label any closed source tools as such.   

The Main list of tools can be found here: https://github.com/Wakoma/LocalML/blob/main/resources/tools.md.

The shared Zotero Library for this project can be found here: https://www.zotero.org/groups/5718368/localml/library

This research project is generously funded by [APC](https://www.apc.org/). 


## Benefits of Local/Offline ML

- Resilience: Offline systems are less vulnerable to internet outages and disruptions, ensuring continued functionality in unpredictable environments.
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








