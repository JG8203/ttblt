# ttblt

A simplified implementation of Byte Latent Transformers as a TorchTune recipe.

https://github.com/facebookresearch/blt is the original (and much more comprehensive!) repo. This project focuses just on fine-tuning an existing pretrained model, and is presented as a TorchTune recipe. 

```
tune run full_finetune_single_device.py --config qwen2_5_3B_blt_full_single_device.yaml
```

The implementation is:
* a simple tokenizer that just takes UTF-8 bytes
* a small local encoder transformer
* simplified patching logic that uses a local entropy measure within sequences to determine patching, with some very basic thresholding
* adding cross-attention onto existing layers from the pretrained model to attend to the patches

The example uses Qwen 2.5 3B (as the original paper experimented with Llama, so I figured some variety would be interesting). It uses the Alpaca dataset, in pretty much the standard Torchtune single device fine tune recipe. Surprisingly, it gets to reasonable results even with this configuration. 

Note that the recipe is modified to:
* Allow non-strict loading of the Qwen checkpoint as we have the extra BLT params
* Specifically filter out the token embeddings, as we dump those in favor of learning a simple byte specific embedding (which hopefully doesn't do too much, but I haven't really ablated anything)

For memory purposes the cross-attention is limited to the last 6 layers of the Qwen model. 
