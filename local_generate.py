    # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.data import Message, Role
from torchtune.training import FullModelTorchTuneCheckpointer
from torchtune.training.checkpointing._utils import safe_torch_load

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)

        if self._quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in self._quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )

        # if self._quantization_mode is None:
        #     ckpt_dict = checkpointer.load_checkpoint()
        # else:
        #     # weights_only needs to be False when loading a quantized model
        #     ckpt_dict = checkpointer.load_checkpoint(weights_only=False)
        # TODO: hack loading. 
        model_state_dict = safe_torch_load(checkpointer._checkpoint_path)
        
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=model_state_dict,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
            for k, v in model_state_dict.items():
                model_state_dict[k] = v.to(self._device)
            model.load_state_dict(model_state_dict, assign=True)
        else:
            model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Dict[Role, str],
    ) -> List[int]:
        """
        Convert the prompt string to a user message with optional system messages
        and tokenize using the prompt template defined on the tokenizer.
        """
        messages = []
        if "system" in prompt and prompt["system"] is not None:
            messages.append(Message(role="system", content=prompt["system"]))
        messages.extend(
            [
                Message(role="user", content=prompt["user"], eot=True),
                # Empty assistant message to kick-start generation
                Message(role="assistant", content=""),
            ]
        )
        return self._tokenizer({"messages": messages}, inference=True)["tokens"]

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(
            cfg.prompt,
        )
        #tokens = [1] + list("Hello ".encode('utf-8'))  # Starting prompt
        custom_generate_next_token = None

        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                )
        # Use patch based model decoding. 
        t0 = time.perf_counter()

        # tokenizer = ByteLatentModelTokenizer()  # Your tokenizer
        #tokens = self._tokenizer.encode("Tell me about cats\n", add_eos=False)
        #tokens = self._tokenizer.encode("Hello ", add_eos=False)
        # generated = model.generate_with_patches(prompt, max_new_tokens=50, top_k=50, temperature=0.7)
        # print("Generated bytes:", generated)
        # print("Decoded:", tokenizer.decode(generated))

        prompt = torch.tensor(tokens, dtype=torch.long, device=self._device) # Changed to long.
                
        # generated_tokens = self._model.generate(prompt, max_new_tokens=20, top_k=10, temperature=0.1)  
        # decoded = bytes(generated_tokens[0]).decode('utf-8', errors='ignore')  # Decode bytes to text
        # print("Generated text (direct):", decoded)

        generated_tokens = self._model.unified_generate(prompt, use_patches=False, frequency_penalty=0.0, repetition_penalty=1.0, max_new_tokens=20, top_k=50, temperature=0.3)  
        decoded = bytes(generated_tokens[0]).decode('utf-8', errors='ignore')  # Decode bytes to text
        print("Generated text (direct, t0.3, tk0, fp/rp):", decoded)

        generated_tokens = self._model.unified_generate(prompt, use_patches=True, max_patch_size=1, frequency_penalty=0.0, repetition_penalty=1.0, max_new_tokens=20, top_k=50, temperature=0.3)  
        decoded = bytes(generated_tokens[0]).decode('utf-8', errors='ignore')  # Decode bytes to text
        print("Generated text (patches, t1, tk0, fp/rp):", decoded)
        
        # generated_tokens = self._model.generate_with_patches(prompt, max_patch_size=1, max_new_tokens=20, top_k=25, temperature=0.1) 
        # decoded = bytes(generated_tokens[0]).decode('utf-8', errors='ignore')  # Decode bytes to text
        # print("Generated text (patches):", decoded)

        #model = ByteLatentQwen2p5Decoder(...)  # Your model instance
        #prompt = [2]  # Example: BOS token
        #generated_tokens = self._model.generate_with_patches(prompt, max_new_tokens=50)
        logits = self._model(generated_tokens[:, :-1])  # Input all but last token
        targets = generated_tokens[:, 1:]  # Predict next tokens
        loss = nn.CrossEntropyLoss()(logits.view(-1, 256), targets.view(-1))
        print("Loss on generated (patches):", loss.item())
        
        
        # generated_tokens = self._model.generate_with_patches(
        #     prompt, 
        #     max_new_tokens=cfg.max_new_tokens,
        #     temperature=cfg.temperature,
        #     top_k=cfg.top_k,
        # )
        # generated_tokens = generated_tokens.tolist()
        t = time.perf_counter() - t0

        #logger.info(self._tokenizer.decode(generated_tokens[0]))

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
