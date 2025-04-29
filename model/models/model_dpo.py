import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
import math
import pandas as pd
from collections import Counter
import faiss
from datasets import load_from_disk
import datasets

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from models.model_base import PreTrainedModelWrapper

class AutoDPOModelForCausalLM(PreTrainedModelWrapper):
    """
    An autoregressive model with support for custom modules in addition to the language model.
    This class inherits from `PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the custom module class you designed. Currently, the supported args are: is_rag, documents_dir, encoder_model_path
    """

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ("is_rag","document_dir","encoder_model_path")
    is_rag = False
    document_dir = None
    encoder_model_path = None
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        super().__init__(pretrained_model, **kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        self.is_rag = custom_module_kwargs["is_rag"]
        if self.is_rag:
            self.document_dir = custom_module_kwargs["document_dir"]
            self.encoder_model_path = custom_module_kwargs["encoder_model_path"]
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            output_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        outputs = self.pretrained_model(input_ids = input_ids,
                                            attention_mask = attention_mask, 
                                            past_key_values = past_key_values,
                                            **kwargs)
        output_dict["logits"] = outputs.logits
        output_dict["past_key_values"] = outputs.past_key_values

        ###############################################################

        return output_dict

    def get_logprobs(self, batch, tokenizer):
      """
      Computes the log probabilities of a response using the model respectively.

      Note: this method was completely done by hand in a naive manner and then parallelized
      and put on gpu by ChatGPT4o.

      Args:
          batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
              The data format is as follows:
              {
                  "prompt": List[str],
                  "chosen": List[str],
                  "rejected": List[str],
                  "chosen_logps": Optional(torch.FloatTensor)
                  "rejected_logps": Optional(torch.FloatTensor)
              }
          tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
      Returns:
          A tuple of two tensors: (chosen_logps, rejected_logps)
          chosen_logps (`torch.FloatTensor`):
              Log probabilities of the chosen responses. Shape: (batch_size,)
          rejected_logps (`torch.FloatTensor`):
              Log probabilities of the rejected responses. Shape: (batch_size,)
      """
      # Ensure the model is on the correct device
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.pretrained_model.to(device)
      
      if not tokenizer.pad_token:
          tokenizer.pad_token = tokenizer.eos_token

      def truncate_and_concatenate(prompt, answer, max_length=1024):
          prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length // 2)
          answer_tokens = tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length // 2)

          combined_input_ids = torch.cat([prompt_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1)
          combined_attention_mask = torch.cat([prompt_tokens["attention_mask"], answer_tokens["attention_mask"]], dim=-1)

          return {
              "input_ids": combined_input_ids.to(device),
              "attention_mask": combined_attention_mask.to(device)
          }

      def pad_to_max_length(input_dict, max_length=1024):
          pad_length = max_length - input_dict["input_ids"].size(1)
          if pad_length > 0:
              pad_input_ids = torch.full((input_dict["input_ids"].size(0), pad_length), tokenizer.pad_token_id, device=device)
              pad_attention_mask = torch.zeros((input_dict["attention_mask"].size(0), pad_length), device=device)

              input_dict["input_ids"] = torch.cat([input_dict["input_ids"], pad_input_ids], dim=1)
              input_dict["attention_mask"] = torch.cat([input_dict["attention_mask"], pad_attention_mask], dim=1)

          return input_dict

      # Prepare inputs
      chosen_inputs = [truncate_and_concatenate(p, c) for p, c in zip(batch["prompt"], batch["chosen"])]
      rejected_inputs = [truncate_and_concatenate(p, r) for p, r in zip(batch["prompt"], batch["rejected"])]

      # Pad inputs to max length
      chosen_inputs = [pad_to_max_length(ci) for ci in chosen_inputs]
      rejected_inputs = [pad_to_max_length(ri) for ri in rejected_inputs]

      # Stack inputs
      chosen_input_ids = torch.stack([ci["input_ids"].squeeze(0) for ci in chosen_inputs])
      rejected_input_ids = torch.stack([ri["input_ids"].squeeze(0) for ri in rejected_inputs])
      chosen_attention_mask = torch.stack([ci["attention_mask"].squeeze(0) for ci in chosen_inputs])
      rejected_attention_mask = torch.stack([ri["attention_mask"].squeeze(0) for ri in rejected_inputs])

      # Create masks for log probability calculation
      chosen_log_prob_mask = chosen_attention_mask
      rejected_log_prob_mask = rejected_attention_mask

      with torch.no_grad():
          chosen_outputs = self.pretrained_model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
          rejected_outputs = self.pretrained_model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)

          chosen_logits = chosen_outputs.logits
          rejected_logits = rejected_outputs.logits

          chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
          rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

          chosen_token_logps = chosen_log_probs.gather(2, chosen_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
          rejected_token_logps = rejected_log_probs.gather(2, rejected_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

          chosen_logps = (chosen_token_logps * chosen_log_prob_mask[:, 1:]).mean(dim=1)
          rejected_logps = (rejected_token_logps * rejected_log_prob_mask[:, 1:]).mean(dim=1)
          
      return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the prediction step that computes the rewards
        # ======================================================================
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        
        # Same beta as during training
        beta = 0.35
        chosen_rewards = beta*(policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta*(policy_rejected_logps - reference_rejected_logps).detach()

        output_dict = {
            "chosen_rewards": chosen_rewards.cpu(),
            "rejected_rewards": rejected_rewards.cpu()
        }
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`list` of `dict`):
                A list of dictionaries containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": str,
                    "choices": List[str],
                    "answer": str,
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to the device
        self.pretrained_model.to(device)
        self.pretrained_model.eval()

        output_dict = {"preds": []}

        # Process questions and choices using the reverse_format_question
        questions_choices = list(map(self.reverse_format_question, batch['question']))

        # Separate questions and choices
        questions = [qc[0] for qc in questions_choices]
        choices = [qc[1] for qc in questions_choices]

        # Transpose choices to match the required format [(A, A, ...), (B, B, ...), ...]
        choices = list(map(list, zip(*choices)))

        batch["question"] = questions
        batch["choices"] = choices

        questions = batch['question']
        choices = list(zip(*batch['choices']))
        choices = [list(item) for item in choices]
        answers = batch['answer']
        contexts = ["" for _ in questions]

        if self.is_rag:

            # Load DPR question encoder and tokenizer
            question_encoder = DPRQuestionEncoder.from_pretrained(self.encoder_model_path)
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.encoder_model_path)

            for i in range(len(questions)):
                question = questions[i]
                inputs = question_tokenizer(question, return_tensors="pt",truncation=True, max_length=512)
                question_embedding = question_encoder(**inputs).pooler_output.detach().cpu().numpy()

                # Load the dataset and FAISS index
                dataset = load_from_disk(self.document_dir + "/encoded_dataset")
                dataset["train"].load_faiss_index("embeddings", self.document_dir + "/encoded_dataset_index.faiss")

                # Perform the retrieval
                scores, retrieved_examples = dataset["train"].get_nearest_examples("embeddings",
                                                                                   np.array(question_embedding), k=1)

                retrieved_contexts = retrieved_examples["text"]
                context = " ".join(retrieved_contexts)
                contexts[i] = context

        #Answer the following multiple-choice questions by selecting the correct letter (A, B, C, ...).\n\n
        task_prompt = f"""
Answer the following multiple-choice questions by selecting the correct letter (A, B, C, D).

Question: The solid-state structures of the principal allotropes of elemental boron are made up of which of the following structural units?
Options: 
A. B12 icosahedra
B. B8 cubes 
B6 octahedra
D. B4 tetrahedra
Answer: A

Question: Question: A machine learning problem involves four attributes plus a class. The attributes have 3, 2, 2, and 2 possible values each. The class has 3 possible values. How many maximum possible different examples are there?
Options: 
A. 12
B. 24
C. 48 
D. 72
Answer: D

Question: Of the following atoms, which has the lowest electron affinity?
Options: 
A. F 
B. Si
C. Ca
D. O
Answer: C

Question: The strongest base in liquid ammonia is?
Options:
A. NH3
B. NH2
C. NH4+
D. N2H4
Answer: B
    """

        if self.is_rag:
            for i in range(len(questions)):
                questions[i] = contexts[i] + " \n\n" + questions[i]

        task_prompt_ids = tokenizer(task_prompt, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
        num_options = max([len([choice for choice in choice_list if choice != '']) for choice_list in choices])

        options = [chr(65 + i) for i in range(num_options)]  # ['A', 'B', 'C', ...]
        options_ids = [tokenizer(letter, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device) for letter in options]

        for i in range(len(questions)):

            non_empty_choices = [choice for choice in choices[i] if choice != '']
            all_permutations = list(itertools.permutations(non_empty_choices))
            num_trials = len(all_permutations)
            
            predictions = []
            last_letter = ''
            patience = 0
            for perm in all_permutations:  # Explore all permutations
                shuffled_choices = [choice[3:] for choice in perm]  # Remove 'A) ', 'B) ', etc.
                question_part = f'\n\nQuestion: {questions[i]}\nOptions:\n'
                for j, choice in enumerate(shuffled_choices):
                    question_part += f'{options[j]}. {choice}\n'


                question_part += 'Answer: '
                question_part_ids = tokenizer(question_part, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1024-len(task_prompt_ids[0]))['input_ids'].to(device)
                input_ids = torch.cat((task_prompt_ids, question_part_ids), dim=1).to(device)
                #print(task_prompt + question_part)
                with torch.no_grad():
                    model_outputs = self.pretrained_model(input_ids)
                    logits = model_outputs.logits

                    vocab_probas = torch.softmax(logits[:, -1, :], dim=-1)
                    option_probs = vocab_probas.gather(1, torch.tensor([options_ids[j] for j in range(len(perm))], dtype=torch.int64).unsqueeze(0).to(device))
                    chosen_option_index = torch.argmax(option_probs).item()
                    #print(chosen_option_index)

                    chosen_option = perm[chosen_option_index]
                    #print(chosen_option)

                    if chosen_option == last_letter:
                        patience += 1
                    else:
                        patience = 1
                        last_letter = chosen_option

                    predictions.append(chosen_option)
                if patience == 3:
                    break

            # Majority vote based on original choices
            majority_vote = Counter(predictions).most_common(1)[0][0]

            # Find the corresponding letter of the majority vote in the original choices
            final_prediction = options[choices[i].index(majority_vote)]
            output_dict["preds"].append(final_prediction)

        return output_dict

    def reverse_format_question(self,formatted_question):
        parts = formatted_question.split('\n\nOptions:\n')
        question = parts[0]
        choices = [choice for choice in parts[1].split('\n')][:-2]
        return question, choices


class AutoDPOModelForSeq2SeqLM(PreTrainedModelWrapper):
    r"""
    A seq2seq model with support for custom modules in addition to the transformer model.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to any `CustomModule` classes.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            ouput_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        output_dict = self.pretrained_model(input_ids = input_ids, 
                                            attention_mask = attention_mask, 
                                            past_key_values = past_key_values,
                                            **kwargs)
        ###############################################################

        return output_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        chosens = []
        rejects = []

        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
            chosen_ids = tokenizer.encode(chosen, return_tensors="pt")
            rejected_ids = tokenizer.encode(rejected, return_tensors="pt")


            chosen_logp = 0
            rejected_logp = 0

            for i in range(len(chosen_ids)):
                prompt_logits = self.pretrained_model(prompt_ids).logits
                chosen_logp += torch.log_softmax(prompt_logits, dim=-1)[0, chosen_ids.squeeze()[i]]
                # Update the prompt_ids to include the true token from the previous step
                prompt_ids = torch.cat([prompt_ids, chosen_ids[:, i].unsqueeze(0)], dim=-1)

            prompt_ids = tokenizer.encode(prompt,
                                          return_tensors="pt")  # Reset the prompt_ids for the rejected responses

            for i in range(len(rejected_ids)):
                prompt_logits = self.pretrained_model(prompt_ids).logits
                rejected_logp += torch.log_softmax(prompt_logits, dim=-1)[0, rejected_ids.squeeze()[i]]
                # Update the prompt_ids to include the true token from the previous step
                prompt_ids = torch.cat([prompt_ids, rejected_ids[:, i].unsqueeze(0)], dim=-1)

            chosens.append(chosen_logp)
            rejects.append(rejected_logp)

        chosen_logps = torch.stack(chosens, dtype = torch.float32)
        rejected_logps = torch.stack(rejects, dtype = torch.float32)

        ###############################################################

        return (chosen_logps, rejected_logps)

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward scores of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the dpo loss function to compute the rewards
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================

        chosen_rewards = policy_chosen_logps - reference_chosen_logps - (policy_rejected_logps - reference_rejected_logps)
        rejected_rewards = policy_rejected_logps - reference_rejected_logps - (policy_chosen_logps - reference_chosen_logps)

        output_dict = {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards
        }
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`list` of `dict`):
                A list of dictionaries containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": str,
                    "choices": List[str],
                    "answer": str,
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict