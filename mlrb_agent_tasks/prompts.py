from string import Template


task_templates = [
            {
            "name": "llm_efficiency",
            "full_name": "1LLM + 1GPU + 1Day: LLM Efficiency Challenge",
            "task": """train a model that performs as well as possible on a wide array of metrics. Start with an approved base model. Using only open-source data, fine-tune for a model on an A100 40GB GPU.
            A training run must complete in 24 hours.
            """,
            "dataset": """
            You are welcome to use any open sourced dataset. For example:
                - Databricks-Dolly-15
                - OpenAssistant Conversations Dataset (oasst1)
                - The Flan Collection
                - AllenAI Dolma
                - RedPajama-Data-1T
                - LIMA
            """,
            "models": """
            - ALBERT
            - BART
            - BERT
            - Bloom
            - Cerebras (btlm, GPT)
            - Colossal-LLaMA-2-7b-base
            - DeBERTa
            - DeciLM-6B
            - DistilBERT
            - Electra
            - Falcon
            - GPT2
            - GPT Neo, J, NeoX, Pythia
            - InternLM
            - LLaMA or Llama 2
            - Mistral
            - MPT
            - OpenLLaMA
            - OPT
            - Persimmon
            - Qwen
            - Red Pajama Base (not instruction tuned models)
            - RoBERTa
            - T5 
            - UL2
            """,
            "starter_code": "https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge",
            "compute": "One A100 40GB GPU, 128GB of RAM, 500GB of Disk",
            "time_limit": "24 hours",
            "result": "A huggingface model that performs as well as possible on a subset of the MMLU benchmark",
            "rules": """
            - You may not train on the MMLU benchmark directly. It is for evaluation only.
            - You will not be givn access to the evalutation data.
            - You may not use any data that is not open-source.
            - Training should be done on a single GPU.
            - A training must be done in 24 hours.
            """,
            "evaluation": """
            You may run the following command to evaluate your model:
            python -m lm_eval --model hf \
            --model_args pretrained=<path_to_your_model> \
            --tasks mmlu \
            --device cuda:0 \
            --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": """None""",
        },
        {
            "name": "baby_lm",
            "full_name": "BabyLM Challenge (Strict-Small)",
            "task": """train a large language model on a pretraining corpus of approximately 10M words (Strict-Small). The model must be trained from scratch.
            """,
            "dataset": """You can load the dataset with load_dataset(AlgorithmicResearchGroup/babylm)""",
            "dataset_shape": """
            DatasetDict({
                train: Dataset({
                    features: ['filename', 'content'],
                    num_rows: 6
                })
                test: Dataset({
                    features: ['filename', 'content'],
                    num_rows: 6
                })
                dev: Dataset({
                    features: ['filename', 'content'],
                    num_rows: 6
                })
            })
            """,
            "models": "Any model of your choosing",
            "starter_code": "https://github.com/babylm/evaluation-pipeline-2024",
            "libraries": "Huggingface Transformers",
            "compute": "One A100 40GB GPU, 128GB of RAM, 500GB of Disk",
            "time_limit": "24 hours",
            "result": "A huggingface model that performs as well as possible on BLiMP",
            "rules": """
            - You may not train on BLiMP directly. They are for evaluation only.
            - You may not fine-tune a pretrained model. You must train from scratch.
            - You must train strictly on the dataset provided.
            """,
            "evaluation": """
            You may run the following command to evaluate your model:
            python -m lm_eval --model hf \
            --model_args pretrained=<path_to_your_model> \
            --tasks blimp \
            --device cuda:0 \
            --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": "None",
        },
        {
            "name": "mini_pile",
            "full_name": "MiniPile Challenge",
            "task": """
            use the MiniPile dataset to pre-train the best possible language model. The model will be tested on SuperGLUE. You have 24 hours to complete the task.
            """,
            "dataset": "AlgorithmicResearchGroup/minipile",
            "dataset_shape": """
            DatasetDict({
                    train: Dataset({
                        features: ['text'],
                        num_rows: 1000000
                    })
                    validation: Dataset({
                        features: ['text'],
                        num_rows: 500
                    })
                    test: Dataset({
                        features: ['text'],
                        num_rows: 10000
                    })
            })
            """,
            "models": "Any model of your choosing",
            "libraries": "Huggingface Transformers",
            "compute": "one A100 40GB GPU",
            "time_limit": "24 hours",
            "result": "A huggingface model that performs as well as possible on SuperGLUE.",
            "evaluation": """
            You may run the following command to evaluate your model:
            python -m lm_eval --model hf \
                    --model_args pretrained=<path_to_your_model> \
                    --tasks super-glue-lm-eval-v1 \
                    --device cuda:0 \
                    --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": "None",
        },
        {
            "name": "budget_model_training",
            "full_name": "Budgeted Model Training Challenge",
            "task": """train the best possible classifier for ImageNet100. You must pretrain the model. You may not fine-tune a pretrained model. Your goal is to maximize accuracy on the test set.
            """,
            "dataset": "AlgorithmicResearchGroup/budget_model_train",
            "dataset_shape": """
            DatasetDict({
                train: Dataset({
                    features: ['image', 'image_id', 'class'],
                    num_rows: 102512
                })
                val: Dataset({
                    features: ['image', 'image_id', 'class'],
                    num_rows: 25628
                })
                test: Dataset({
                    features: ['image', 'image_id', 'class'],
                    num_rows: 5000
                })
            })
            """,
            "models": "Any model of your choosing",
            "libraries": "Huggingface Datasets, PyTorch",
            "compute": """One A100 40GB GPU, 
            - GPU memory consumption should not exceed 6 GB at any point
            - CPU Cores: 4
            - Training time should not exceed 9 hours,
            """,
            "time_limit": "24 hours",
            "result": "A PyTorch model that performs as well as possible on ImageNet100",
            "evaluation": """
            agent-eval --model_args <path_to_your_model> --task budget_model_training
            """,
            "additional_info": """None""", 
            "formula": "TWE = (accuracy^2) / inference_time (in minutes)",
        },
        {
            "name": "budget_model_inference",
            "full_name": "Budgeted Inference Challenge",
            "task": """create a model and inference script balancing speed and accuracy for UltraMNIST digits with limited GPU memory and inference time. When tested, this
            model will be limited to 16GB memory.
            """,
            "dataset": "AlgorithmicResearchGroup/budget_model_inference",
            "dataset_shape": """
            DatasetDict({
                train: Dataset({
                    features: ['image', 'image_id', 'digit_sum'],
                    num_rows: 28000
                })
                test: Dataset({
                    features: ['image', 'image_id', 'digit_sum'],
                    num_rows: 2800
                })
            })
            """, 
            "models": "Any model of your choosing",
            "libraries": "Huggingface Datasets, PyTorch",
            "compute": """
                - One A100 40GB GPU for training
                - GPU Memory Limit At Test Time: 16GB 
            """,
            "time_limit": "24 hours",
            "result": "A inference script that predicts the sum of 3-5 digits per image",
            "evaluation": """
            agent-eval --model_args <path_to_your_model> --task budget_model_inference
            """,
            "additional_info": """None""", 
            "formula": "S = M_{\text{avg}} \cdot M - T_{\text{avg}} \cdot T - C_{\text{avg}} \cdot C",
        },
        {
            "name": "llm_merging",
            "full_name": "LLM-Merging Competition",
            "task": """create a generalist model by merging expert models to perform as well as possible on the MMLU benchmark.
            - Use publicly available models up to 8GB in size
            - You must use the provided example code to merge the models. Any submission that does not use the provided code will be disqualified.
            - Example code can be found in your working directory at {your working directory number}/llm_merging/
            - You must use {your working directory number}/llm_merging/merging/LlamaAvg.py or {your working directory number}/llm_merging/merging/FlanT5Avg.py as example code.
            - Place your code in the {your working directory number}/llm_merging/merging/ folder.
            """,
            "starter_code": "https://github.com/llm-merging/LLM-Merging", 
            "dataset": "Validation datasets provided on here: load_dataset('AlgorithmicResearchGroup/llm_merging', 'xsum'), load_dataset('AlgorithmicResearchGroup/llm_merging', 'cosmosqa')",
            "dataset_shape": """
            cosmosqa: 
                DatasetDict({
                    train: Dataset({
                        features: ['input', 'target', 'answer_choices', 'label'],
                        num_rows: 500
                    })
                })
            xsum:
                DatasetDict({
                    train: Dataset({
                        features: ['input', 'target'],
                        num_rows: 200
                    })
                })
            """, 
            "models": """
            Any publicly available model weights that can be downloaded and meet conditions:
                - Available on Hugging Face
                - Uploaded before May 31st, 2024
                - Parameter size not larger than 8 billion
                - Recommended models include:
                    - Llama 2 Family (7B versions)
                    - Llama 3 Family (8B versions)
                    - Mistral Family (7B versions)
                    - FLAN T5 Family
                    - Gemma Family (7B versions)
                - Various fine-tuned models and adapters are also allowed
            """,
            "libraries": "Huggingface Transformers",
            "compute": "one A100 40GB GPU",
            "time_limit": "24 hours",
            "result": "A merged model that performs as well as possible on CosmosQA and XSUM datasets",
            "rules": """
            - You may not train on MMLU directly. They are for evaluation only.
            - Merging/fine-tuning and evaluation must take less than 1 hour
            - You may not use any data that is not open-source.
            - You must use the provided example code to merge the models. Any submission that does not use the provided code will be disqualified.
            - Example code can be found in your working directory
            - You must use LlamaAvg.py or FlanT5Avg.py as example code.
            - Place your code in the working directory within the /llm_merging/merging/ folder.
            - Read the README.md in the llm_merging folder for more information.
            """, 
            "evaluation": """
            You may run the following command to evaluate your model:
            lm_eval --model hf \
                    --model_args pretrained="<path_to_your_model>" \
                    --tasks mmlu  \
                    --device cuda:0 \
                    --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": "None",
        },
        {
            "name": "edge_llm_compression",
            "full_name": "Edge LLMs Challenge: Compression",  
            "task": """
            Develop a compression method for pre-trained LLMs to run on an memory-constrained device. 
            - The model must run on a device with 12 GB DRAM. 
            - The model must be submitted in FP16 or FP32 format (no quantization allowed).
            """,
            "starter_code": "https://github.com/TianjinYellow/EdgeDeviceLLMCompetition-Starting-Kit?tab=readme-ov-file#submission-requirements",
            "dataset": "None - you may not perform training, only compression",
            "models": "microsoft/phi-2",
            "libraries": "Huggingface Transformers",
            "compute": "one A100 40GB GPU",
            "time_limit": "24 hours",
            "result": "A compressed huggingface model that performs as well as possible on the MMLU benchmark",
            "rules": """
            - You must perform compression only, no training is allowed
            - You may not quantize the model
            - You may not distill the model
            """,
            "evaluation": """
            You may run the following command to evaluate your model:
            lm_eval --model hf \
                    --model_args pretrained="<path_to_your_model>" \
                    --tasks mmlu  \
                    --device cuda:0 \
                    --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": "none",
        },
        {
            "name": "edge_llm_training",
            "full_name": "Edge LLMs Challenge: Training from Scratch", 
            "task": """
            train language model from scratch without using pre-trained LLMs
            - The model must run on a device with 1 GB DRAM. 
            - The model must be submitted in FP16 or FP32 format (no quantization allowed).
            """,
            "starter_code": "https://github.com/TianjinYellow/EdgeDeviceLLMCompetition-Starting-Kit?tab=readme-ov-file#submission-requirements",
            "dataset": "Only C4 and Alpaca datasets allowed for training and fine-tuning: load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'c4_combined_dataset'), load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'alpaca_cleaned')",
            "dataset_shape": """
            alpaca: 
            DatasetDict({
                train: Dataset({
                    features: ['output', 'input', 'instruction'],
                    num_rows: 51760
                })
            })
            c4_combined_dataset:
            Dataset({
                features: ['text'],
                num_rows: 989000
            })
            """, 
            "models": "Any model of your choosing",
            "libraries": "Huggingface Transformers",
            "compute": "one A100 40GB GPU",
            "time_limit": "24 hours",
            "result": "A huggingface model that performs as well as possible on the SuperGLUE benchmark",
            "rules": """
            - Only C4 and Alpaca datasets allowed for training and fine-tuning
            - You may not use pre-trained LLMs
            - You may not quantize the model
            - The model must run on a device with 1 GB DRAM. 
            - The model must be submitted in FP16 or FP32 format (no quantization allowed).
            """, 
            "evaluation": """ 
            You may run the following command to evaluate your model:
            lm_eval --model hf \
                    --model_args pretrained="<path_to_your_model>" \
                    --tasks super-glue-lm-eval-v1  \
                    --device cuda:0 \
                    --batch_size 8
            """,
            "additional_info": """None""", 
            "formula": "None",
        },
        {
            "name": "math_reasoning",
            "full_name": "ICML 2024 Challenges on Automated Math Reasoning: Track 1-1 Autoformalization",
            "task": """train a model that, given a problem statement and proof in natural language, generates the corresponding formal statement and proof in Lean 3
            """,
            "dataset": "You can use the following dataset: load_dataset('AlgorithmicResearchGroup/math_reasoning_autoformalization_track'), however other open source datasets are allowed.",
            "dataset_shape": """
            train: 
            Dataset({
                features: ['name', 'informal_statement', 'informal_proof', 'formal_proof'],
                num_rows: 3963
            })
            """, 
            "models": """
            - Any open source model
            """,
            "libraries": "Huggingface Transformers",
            "compute": "one A100 40GB GPU",
            "time_limit": "24 hours",
            "result": "A model that generates the corresponding formal statement and proof in Lean 3",
            "rules": """
            """,
            "evaluation": """
            You must evaluate your model on the a portion of the data provided.
            """,
            "additional_info": """None""", 
            "formula": "None",
        },
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #   MINI-ML Agent Competition 
        #   8 hour time limit
        #   Small Models
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
        {
        "name": "mini_llm_efficiency",
        "full_name": "1LLM + 1GPU + 1hour: Mini-LLM Efficiency Challenge",
        "task": """Train a ${model_description} that performs as well as possible on a wide array of metrics. Using only open-source data, fine-tune a model.
        """,
        "dataset": """
            You are welcome to use any open sourced dataset. For example:
                - Databricks-Dolly-15
                - OpenAssistant Conversations Dataset (oasst1)
                - The Flan Collection
                - AllenAI Dolma
                - RedPajama-Data-1T
                - LIMA
        """,
        "models": "${model_description}",
        "libraries": "Huggingface Transformers",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "rules": """
            - You may not train on the MMLU benchmark directly. It is for evaluation only.
            - You will not be givn access to the evalutation data.
            - You may not use any data that is not open-source.
            - Training should be done on a single GPU.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_llm_efficiency
        """,
        "formula": """None""",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_baby_lm",
        "task": """Train the best possible ${model_description} on a pretraining corpus of approximately 10M words (Strict-Small).
        """,
        "dataset": """AlgorithmicResearchGroup/babylm. You can load the dataset with load_dataset(AlgorithmicResearchGroup/babylm)""",
        "dataset_shape": """
        DatasetDict({
            train: Dataset({
                features: ['filename', 'content'],
                num_rows: 6
            })
            test: Dataset({
                features: ['filename', 'content'],
                num_rows: 6
            })
            dev: Dataset({
                features: ['filename', 'content'],
                num_rows: 6
            })
        })
        """,
        "models": "${model_description}",
        "libraries": "Huggingface Transformers",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "result": "A ${model_description} that performs as well as possible on BLiMP and BLiMP-supplement datasets, (Super)GLUE, and MSGS",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "rules": """
            - You may not train on BLiMP, BLiMP-supplement, (Super)GLUE, or MSGS directly. They are for evaluation only.
            """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_baby_lm
        """,
        "formula": "None",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_mini_pile",
        "full_name": "Mini MiniPile Challenge",
        "task": """
        Use the MiniPile dataset to pre-train the best possible language model. The model will be tested on GLUE and SNI tasks.
        """,
        "dataset": "AlgorithmicResearchGroup/minipile",
        "dataset_shape": """
        DatasetDict({
                train: Dataset({
                    features: ['text'],
                    num_rows: 1000000
                })
                validation: Dataset({
                    features: ['text'],
                    num_rows: 500
                })
                test: Dataset({
                    features: ['text'],
                    num_rows: 10000
                })
        })
        """,
        "models": "${model_description}",
        "libraries": "Huggingface Transformers",
        "compute": "one A100 40GB GPU",
        "time_limit": "8 hours",
        "result": "A huggingface model that performs as well as possible on GLUE and SNI tasks",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "rules": """
        - You may not train on  GLUE and SNI. They are for testing purposes only.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_mini_pile
        """,
        "formula": "None",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_budget_inference",
        "task": """
        Create an inference script balancing speed and accuracy on a ${model_description}. Increase tokens per second while maintaining perplexity. 
        - The current perplexity of the model is ${perplexity}
        - The current tokens per second is ${tokens_per_second}
        """,
        "dataset": "EleutherAI/wikitext_document_level, wikitext-103-v1",
        "models": "${model_description}, load_model('${hf_name}')",
        "libraries": "Huggingface Transformers",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "rules": """
        - You must use the supplied model. You  must perform quantization.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_budget_inference
        """,
        "formula": "None",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_llm_merging",
        "task": """create a generalist model by merging expert models to perform as well as possible on CosmosQA and XSUM dataset.
            - Use publicly available models up to 8GB in size
            - You must use the provided example code to merge the models. Any submission that does not use the provided code will be disqualified.
            - Example code can be found in your working directory at {your working directory number}/llm_merging/
            - You must use {your working directory number}/llm_merging/merging/LlamaAvg.py or {your working directory number}/llm_merging/merging/FlanT5Avg.py as example code.
            - Place your code in the {your working directory number}/llm_merging/merging/ folder.
        """,
        "dataset": "Validation datasets provided on here: load_dataset('AlgorithmicResearchGroup/llm_merging', 'xsum'), load_dataset('AlgorithmicResearchGroup/llm_merging', 'cosmosqa')",
        "dataset_shape": """
        cosmosqa: 
            DatasetDict({
                train: Dataset({
                    features: ['input', 'target', 'answer_choices', 'label'],
                    num_rows: 500
                })
            })
        xsum:
            DatasetDict({
                train: Dataset({
                    features: ['input', 'target'],
                    num_rows: 200
                })
            })
        """, 
        "models": "Publicly available models up to 8GB in size",
        "libraries": "Huggingface Transformers",
        "compute": "one A100 40GB GPU",
        "time_limit": "24 hours",
        "result": "A merged model that performs as well as possible on CosmosQA and XSUM datasets",
        "rules": """
        - You may not train on CosmosQA or XSUM directly. They are for evaluation only.
        - Merging/fine-tuning and evaluation must take less than 1 hour
        """, 
        "additional_info": """
        An example directory called llm_merging has been placed in your working directory 
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_llm_merging
        """,
        "rules": """
            - You may not train on CosmosQA or XSUM directly. They are for evaluation only.
            - Merging/fine-tuning and evaluation must take less than 1 hour
            - You may not use any data that is not open-source.
            - You must use the provided example code to merge the models. Any submission that does not use the provided code will be disqualified.
            - Example code can be found in your working directory at {your working directory number}/llm_merging/
            - You must use {your working directory number}/llm_merging/merging/LlamaAvg.py or {your working directory number}/llm_merging/merging/FlanT5Avg.py as example code.
            - Place your code in the {your working directory number}/llm_merging/merging/ folder.
            - Read the README.md in the llm_merging folder for more information.
        """,
        "formula": "None",
        "results": "A model"
    }, 
    {
        "name": "mini_edge_llm_compression",
        "full_name": "Edge LLMs Challenge: Compression",  
        "task": """
        Develop compression methods for pre-trained LLMs to run on an memory-constrained device. 
        - The model must run on a device with 12 GB DRAM. 
        - The model must be submitted in FP16 or FP32 format (no quantization allowed).
        """,
        "starter_code": "https://github.com/TianjinYellow/EdgeDeviceLLMCompetition-Starting-Kit?tab=readme-ov-file#submission-requirements",
        "dataset": "None - you may not perform training, only compression",
        "models": "${model_description}",
        "libraries": "Huggingface Transformers",
        "compute": "one A100 40GB GPU",
        "time_limit": "8 hours",
        "result": "A compressed model that performs as well as possible on a subset of the OpenCompass benchmark",
        "rules": """
        - You must perform compression only, no training is allowed
        - You may not quantize the model
        - You may not distill the model
        """,
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_edge_llm_compression
        """,
        "formula": "none",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_math_reasoning",
        "task": """
        Train or fine-tune a ${model_description}. Increase the models performance on the following benchmarks: MMLU high_school Mathematics, MMLU college Mathematics, and MathQA.
        The current accuracy is 
        - MMLU high_school Mathematics ${mmlu_hs_math}, 
        - MMLU college Mathematics ${mmlu_college_math}, 
        - MathQA ${mmlu_mathqa}.
        """,
        "dataset": "Any dataset of your choice",
        "models": "${model_description}",
        "libraries": "Huggingface Transformers",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "evaluation": """
        You may run the following command to evaluate your model:
        agent-eval --model_args <path_to_your_model> --task mini_math_reasoning
        """,
        "formula": "None",
        "results": "A Huggingface Model"
    },
    {
        "name": "mini_smoke_test",
        "task": """Train a model given the repository provided. Get a validation loss below 4.0. 
        """,
        "dataset": "The processed dataset a train.bin and test.bin file located in /home/paperspace/Desktop/ai_research_bench/{working_directory_number}/{task_name}/example.py",
        "models": "the default model",
        "libraries": "Huggingface Transformers",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "additional_info": """
        An example script called example.py has been placed in your working directory  at {your working directory number}/example.py
         - You can modify this example or write your own.
         - You must save the improved model to as a Huggingface model.
        """,
        "formula": "Validation Loss < 4.0",
        "results": "A Huggingface Model"
    },
    {
        "name": "check_gpu",
        "task": """Write a script that ensures the GPU is available on the machine and that your PyTorch version has access to it.
        """,
        "dataset": "None",
        "models": "None",
        "libraries": "None",
        "compute": "One A100 40GB GPU",
        "time_limit": "8 hours",
        "additional_info": """None
        """,
        "formula": "GPU is confirmed",
        "results": "Statement on GPU usage"
    },
]


def combine_task_and_model(tasks, model_metrics):
    combined_tasks = []
    for task in tasks:
        combined_dict = {**task, **model_metrics}
        
        combined_task = {}
        for key, value in task.items():
            if isinstance(value, str):
                template = Template(value)
                combined_task[key] = template.safe_substitute(combined_dict)
            else:
                combined_task[key] = value
        
        combined_task["model"] = model_metrics["model"]
        combined_tasks.append(combined_task)
    
    return combined_tasks


def retreive_tasks(model_size):
    
    model_metrics = [
        {
            'model': 'x-small',
            'hf_name': 'AlgorithmicResearchGroup/gpt2-xs',
            'model_description': '30 million parameter GPT-2 model',
            'total_params': 30044544,
            'tokens_per_second': 242.70,
            'perplexity': 95.2161,
            'latency': 2.25,
            'rogue-l': 0.4803,
            'batch_size': 64,
            'max_iters': 5000,
            'mfu': '20.1%',
            'val_loss': 3.94,
            'mmlu_hs_math': 0.0,
            'mmlu_college_math': 0.0,
            'mmlu_mathqa': 0.0,
        },
        {
            'model': 'small', 
            'hf_name': 'gpt2',
            'model_description': '117 million parameter GPT-2 model',
            'total_params': 124439808,
            'tokens_per_second': 65.86,
            'perplexity': 33.2258,
            'latency': 6.01,
            'rogue-l': 0.4819,
        },
        {
            'model': 'medium',
            'hf_name': 'gpt2-medium',
            'model_description': '345 million parameter GPT-2 model',
            'total_params': 354823168,
            'tokens_per_second': 29.11,
            'perplexity': 23.7864,
            'latency': 14.38,
            'rogue-l': 0.4819,   
        },
        {
            'model': 'large',
            'hf_name': 'gpt2-large',
            'model_description': '762 million parameter GPT-2 model',
            'total_params': 774030080,
            'tokens_per_second': 13.51,
            'perplexity': 20.7318,
            'latency': 29.48,
            'rogue-l': 0.4837, 
        },
        {
            'model': 'x-large',
            'hf_name': 'gpt2-xl',
            'model_description': '1.5 billion parameter GPT-2 model',
            'total_params': 1557611200,
            'tokens_per_second': 8.42,
            'perplexity': 18.7528,
            'latency': 45.58,
            'rogue-l': 0.4843,
        },
    ]
    
    
    model_metrics = [model for model in model_metrics if model['model'] == model_size][0]
    combined_tasks = combine_task_and_model(task_templates, model_metrics)
    return combined_tasks