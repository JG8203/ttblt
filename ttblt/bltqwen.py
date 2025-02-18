import torch
from torch import nn
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from torchtune.modules.transformer import TransformerDecoder
from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    TransformerCrossAttentionLayer,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.model_fusion import FusionLayer
from torchtune.models.qwen2._component_builders import qwen2_mlp

BOE_ID = 2
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

################################################
# Local encoder/decoder (with cross-attn)
################################################

def build_local_encoder(
    vocab_size: int = 256,
    embed_dim: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 8,
    hidden_dim: int = 4096,
    norm_eps: float = 1e-5,
    attn_dropout: float = 0.0,
    max_seq_len: int = 2048,
    num_layers: int = 4,
    dtype=torch.bfloat16
):
    """
    Build a local "encoder" using torchtune's TransformerDecoder, but with:
      - no cross-attention
      - self-attention only
      - Qwen2 MLP style
      - final RMS norm and no output projection (Identity)
    """
    head_dim = embed_dim // num_heads
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    # Build self-attention layers with Qwen MLP
    layers = nn.ModuleList()
    for _ in range(num_layers):
        # TODO: KV cache?
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    local_encoder = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=nn.Identity(),  # no final projection
    )
    local_encoder = local_encoder.to(dtype=dtype)
    return local_encoder

################################################
# dynamic patching
################################################

def compute_local_entropy(bytes_tensor, window_size=8):
    """Return a per-token "entropy" measure to guide patching
    
    Arguments:
        bytes_tensor: Torch.tensor[batch_size, seq_len] byttes to calc entropy on
        window_size: int size to window across

        local_entropy: torch.Tensor[batch_size, seq_len]
    """
    # bytes_tensor: 
    device = bytes_tensor.device
    batch_size, seq_len = bytes_tensor.shape
    
    # We’ll keep a sliding frequency table. Initialize all zeros:
    freq = torch.zeros(batch_size, 256, device=device)
    local_entropy = torch.zeros(batch_size, seq_len, device=device)
    
    for pos in range(seq_len):
        # add current byte
        current_byte = bytes_tensor[:, pos]
        freq[torch.arange(batch_size), current_byte] += 1
        
        # compute distribution
        dist = freq / freq.sum(dim=1, keepdim=True).clamp_min(1e-8)
        # compute -p*log2(p)
        ent = -(dist * (dist + 1e-8).log2()).sum(dim=1)
        local_entropy[:, pos] = ent
        
        # remove oldest byte if we exceed window size
        if pos >= window_size:
            oldest_byte = bytes_tensor[:, pos - window_size]
            freq[torch.arange(batch_size), oldest_byte] -= 1
    return local_entropy

def dynamic_patch(
    bytes_tensor: torch.Tensor,
    threshold: float = 3.0,   # starting entropy threshold in bits
    min_threshold: float = 2.0,    # lower bound
    max_threshold: float = 5.0,    # upper bound
    threshold_step_down: float = 0.1,  # how much to decrease threshold if no patches triggered
    threshold_step_up: float = 0.1,    # how much to increase threshold if we trigger a patch
    patch_size: int = 4,
    window_size: int = 8
):
    """
    A dynamic patching approach that adjusts the entropy threshold
    upward/downward depending on whether patches are being triggered too often or not enough.

    Args:
        bytes_tensor: [batch_size, seq_len]
        threshold: initial bits threshold for local entropy
        min_threshold, max_threshold: clamp thresholds
        threshold_step_down, threshold_step_up: step sizes
        patch_size: max patch length if we haven't triggered a boundary earlier
        window_size: for computing local entropy

    Returns:
        patch_ids: [batch_size, seq_len] with patch ID for each token
        local_ent: [batch_size, seq_len] local entropy in bits
    """
    local_ent = compute_local_entropy(bytes_tensor, window_size=window_size)
    batch_size, seq_len = bytes_tensor.shape
    
    # Each row’s patch assignment
    patch_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=bytes_tensor.device)

    current_patch = torch.zeros(batch_size, dtype=torch.long, device=bytes_tensor.device)
    patch_lengths = torch.zeros(batch_size, dtype=torch.long, device=bytes_tensor.device)

    # Keep track of how many consecutive tokens we have processed w/o triggering any new patch
    # so we can adjust threshold if we go too long
    consecutive_no_trigger = 0
    
    for pos in range(seq_len):
        patch_lengths += 1
        
        # Which batch elements exceed threshold or hit patch size
        trigger_new_patch = (local_ent[:, pos] > threshold) | (patch_lengths >= patch_size)
        triggered_rows = trigger_new_patch.nonzero(as_tuple=False).flatten()

        if len(triggered_rows) > 0:
            # For each row that triggered, increment patch and reset patch_lengths
            current_patch[triggered_rows] += 1
            patch_lengths[triggered_rows] = 0

            # Because at least 1 row triggered a patch, we can optionally adjust threshold upward
            # or downward. For example, *raising* threshold if we keep splitting too often:
            threshold = min(threshold + threshold_step_up, max_threshold)
            
            consecutive_no_trigger = 0
        else:
            # No new patch was triggered
            consecutive_no_trigger += 1
            # If we haven't triggered for a while, lower threshold so that we become more likely
            # to split in the future.
            threshold = max(threshold - threshold_step_down, min_threshold)

        patch_ids[:, pos] = current_patch

    return patch_ids, local_ent

def patch_reduce(h, patch_ids, reduce_op="mean"):
    """ 
    Arguments:
        h: [batch_size, seq_len, emb_dim]
        patch_ids: [batch_size, seq_len]
        reduce_op: e.g. "mean", "amin", "amax"
    
    returns: [batch_size, num_patches, emb_dim]
    """
    batch_size, seq_len, emb_dim = h.shape
    num_patches = patch_ids.max().item() + 1  # patch IDs go from 0..max

    # expand dims so we can scatter:
    expanded_ids = patch_ids.unsqueeze(-1).expand(-1, -1, emb_dim)
    reduced = torch.zeros(batch_size, num_patches, emb_dim, device=h.device, dtype=h.dtype)
    reduced = reduced.scatter_reduce(
        dim=1,
        index=expanded_ids,
        src=h,
        reduce=reduce_op,
        include_self=False,
    )
    return reduced

def compute_patch_size(so_far: torch.Tensor, threshold=3.0, max_patch=8):
    """
    heuristic function for deciding the patch length
    based on dynamic_patch logic or local entropy.
    
    so_far: [seq_len] or [1, seq_len], the current context of tokens (including newly generated ones).
    threshold: approximate bits threshold for deciding to break the patch.
    max_patch: a max patch length to avoid overly large chunks.

    Returns:
        predicted_patch_size: int
            the number of bytes to decode in the *next* patch in a single forward pass
    """
    # re-run dynamic_patch() on the entire sequence and see how big the last patch is. 
    # this is probably pretty wasteful!s
    # Then we decide how big the *next* patch would be if we continued. 
    # This is a simple way to reuse your dynamic_patch code.
    if so_far.dim() == 1:
        so_far = so_far.unsqueeze(0)  # [batch=1, seq_len]
    
    patch_ids, _ = dynamic_patch(so_far, threshold=threshold, patch_size=max_patch)

    # The ID of the last patch in that sequence:
    last_patch_id = patch_ids[0, -1].item()  # e.g. 3 means patches 0..3
    # Count how many tokens so far belong to the last patch
    # (i.e. sum of patch_ids == last_patch_id)
    count_last_patch = (patch_ids[0] == last_patch_id).sum().item()

    # guess that the next patch might be similar in size:
    # If we already used up to 'count_last_patch' tokens for the last patch,
    # we can try the same or smaller for the next patch. A simple approach is:
    predicted_size = max(1, min(count_last_patch, max_patch))

    return predicted_size

################################################
# Projection layer
################################################
class PatchToGlobalProjector(nn.Module):
    def __init__(self, local_dim, global_dim):
        super().__init__()
        self.proj = nn.Linear(local_dim, global_dim)
    def forward(self, x):
        return self.proj(x)


################################################
# ByteLatentQwen2p5Decoder with cross-attn
################################################
class ByteLatentQwen2p5Decoder(TransformerDecoder):
    def __init__(
        self,
        qwen_cfg: Dict[str, Any],
        local_encoder_cfg: Dict[str, Any],
        patch_size: int = 4,
        patching_threshold: float = 3.0,
        freeze_global_for_n_steps: int = 0,
        local_to_global_dim_proj: bool = True,
        cross_attend_layers: Optional[List[int]] = None,
    ):
        layers = nn.ModuleList()
        head_dim = qwen_cfg['embed_dim'] // qwen_cfg['num_heads']

        for _ in range(qwen_cfg['num_layers']):
            self_attn = MultiHeadAttention(
                embed_dim=qwen_cfg['embed_dim'],
                num_heads=qwen_cfg['num_heads'],
                num_kv_heads=qwen_cfg['num_kv_heads'],
                head_dim=head_dim,
                q_proj=nn.Linear(qwen_cfg['embed_dim'], qwen_cfg['num_heads'] * head_dim, bias=True),
                k_proj=nn.Linear(qwen_cfg['embed_dim'], qwen_cfg['num_kv_heads'] * head_dim, bias=True),
                v_proj=nn.Linear(qwen_cfg['embed_dim'], qwen_cfg['num_kv_heads'] * head_dim, bias=True),
                output_proj=nn.Linear(qwen_cfg['embed_dim'], qwen_cfg['embed_dim'], bias=False),
                kv_cache=None,
                max_seq_len=qwen_cfg['max_seq_len'],
                attn_dropout=qwen_cfg['attn_dropout'],
            )
            mlp = qwen2_mlp(dim=qwen_cfg['embed_dim'], hidden_dim=qwen_cfg['intermediate_dim'])
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=qwen_cfg['embed_dim'], eps=qwen_cfg['norm_eps']),
                mlp_norm=RMSNorm(dim=qwen_cfg['embed_dim'], eps=qwen_cfg['norm_eps']),
            )
            layers.append(layer)

        output = nn.Linear(
            qwen_cfg['embed_dim'],
            256,  # bytes
            bias=False,
        )
        # Initialize output bytes. 
        init_std = qwen_cfg['embed_dim'] ** -0.5
        with torch.no_grad():
            nn.init.trunc_normal_(
                output.weight,
                mean=0.0,
                std=init_std,
                a=-3*init_std,
                b=3*init_std,
            )
        byte_embedding = nn.Embedding(256, qwen_cfg["embed_dim"])

        super().__init__(
            tok_embeddings=byte_embedding,
            layers=layers,
            max_seq_len=qwen_cfg['max_seq_len'],
            num_heads=qwen_cfg['num_heads'],
            head_dim=head_dim,
            norm=RMSNorm(qwen_cfg['embed_dim'], eps=qwen_cfg['norm_eps']),
            # We are overriding this to be bytes, ignoring the tokenizer. 
            output=output,
            # We want the last hidden state for cross attending. 
            # output_hidden_states=[qwen_cfg['num_layers']-1]
        )

        if cross_attend_layers is None:
            cross_attend_layers = [qwen_cfg["num_layers"] - 1]  # e.g. only final layer
        self._inject_cross_attn(self, qwen_cfg, cross_attend_layers)

        self.local_encoder = build_local_encoder(**local_encoder_cfg)

        self.patch_size = patch_size
        self.patching_threshold = patching_threshold
        self.freeze_global_for_n_steps = freeze_global_for_n_steps
        self.current_step = 0
        self.global_frozen = freeze_global_for_n_steps > 0

        if local_to_global_dim_proj:
            local_dim = local_encoder_cfg['embed_dim']
            global_dim = qwen_cfg['embed_dim']
            #self.layers[0].attn.q_proj.in_features  # typically same as embed_dim but could differ
            self.patch_projector = PatchToGlobalProjector(local_dim, global_dim)
        else:
            self.patch_projector = nn.Identity()

        # We'll store how many chunks the user wants for final output
        self.num_output_chunks = 0  # default

    def _inject_cross_attn(
        self,
        decoder: TransformerDecoder,
        qwen_cfg: Dict[str, Any],
        cross_attend_layers: List[int],
    ):
        """
        Wraps each designated Qwen layer in a FusionLayer that adds a cross-attn
        sub-layer after self-attn. The new cross-attn weights won't match any
        pretrained checkpoint keys, so they'll be randomly initialized.
        """
        embed_dim = qwen_cfg["embed_dim"]
        num_heads = qwen_cfg["num_heads"]
        num_kv_heads = qwen_cfg["num_kv_heads"]
        head_dim = embed_dim // num_heads

        for idx in cross_attend_layers:
            old_layer = decoder.layers[idx]

            # Build a cross-attn sub-layer
            cross_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                attn_dropout=qwen_cfg["attn_dropout"],
                is_causal=False,  # cross-attn is typically non-causal\
            )
            cross_mlp = qwen2_mlp(dim=embed_dim, hidden_dim=qwen_cfg["intermediate_dim"])
            cross_layer = TransformerCrossAttentionLayer(
                attn=cross_attn,
                mlp=cross_mlp,
                ca_norm=RMSNorm(embed_dim, eps=qwen_cfg["norm_eps"]),
                mlp_norm=RMSNorm(embed_dim, eps=qwen_cfg["norm_eps"]),
            )
            fused = FusionLayer(
                layer=old_layer,       # existing self-attn + MLP
                fusion_layer=cross_layer,
                fusion_first=False,    # do self-attn first, then cross-attn
            )
            decoder.layers[idx] = fused  # replace it in place

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        super().set_num_output_chunks(num_output_chunks)
        self.num_output_chunks = num_output_chunks

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[Union[torch.Tensor, float]] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Freeze/unfreeze global if needed
        if self.global_frozen:
            for param in self.norm.parameters():
                param.requires_grad = False
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = False
            for p in self.output.parameters():
                p.requires_grad = False
            # if there’s a tok_embeddings for global
            for p in self.tok_embeddings.parameters():
                p.requires_grad = False
        else:
            for param in self.norm.parameters():
                param.requires_grad = True
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.output.parameters():
                p.requires_grad = True
            for p in self.tok_embeddings.parameters():
                p.requires_grad = True

        if self.training and self.global_frozen:
            self.current_step += 1
            if self.current_step >= self.freeze_global_for_n_steps:
                self.global_frozen = False  # unfreeze
        
        local_enc_out = self.local_encoder(tokens)
        # TODO: remove this hack. If not using chunked output, TransformerDecoder called .float
        # on the output, annoyingly throwing us into float32. 
        local_enc_out = local_enc_out.to(torch.bfloat16)
        
        patch_ids, tok_scores = dynamic_patch(
            tokens, threshold=self.patching_threshold, patch_size=self.patch_size
        )
        patch_embs = patch_reduce(local_enc_out, patch_ids, reduce_op="mean")
        patch_embs = self.patch_projector(patch_embs)
        
        # this is a list of chunk x [b, num_patches, 256]
        # batch_size, dec_seq_len = tokens.shape
        # _, n_patches, _ = patch_embs.shape
        # # Create an all-True mask: [B, T, N]
        # encoder_mask = patch_embs.new_ones(
        #     (batch_size, dec_seq_len, n_patches), dtype=torch.bool
        # )
        local_logits = super().forward(
            tokens,        
            mask=mask,
            encoder_input=patch_embs,
            # encoder_mask=encoder_mask, # For generation.
            input_pos=input_pos,
        )

        # Super handles chunking
        return local_logits
    
    def generate_with_patches(
        self,
        prompt: Union[torch.LongTensor, List[int]],
        max_new_tokens: int = 128,
        eos_id: int = EOS_ID,    # your EOS_ID
        threshold: float = 3.0,
        max_patch_size: int = 8,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> List[int]:
        """
        Generate up to `max_new_tokens` bytes using patch-based decoding.
        We'll do a single forward pass for each 'patch' chunk, hopefully bigger than 1 byte
        for efficiency. We do a simple step-by-step sampling inside that patch.

        Args:
            prompt: initial list of bytes or a tensor shaped [seq_len]
            max_new_tokens: total new bytes to produce
            eos_id: ID for end-of-sequence
            threshold: local entropy threshold for dynamic patch size
            max_patch_size: maximum patch chunk we consider
            temperature, top_k: sampling hyperparams

        Returns:
            all_tokens: a list of all bytes (prompt + newly generated)
        """
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device=next(self.parameters()).device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)  # shape [1, seq_len]

        all_tokens = prompt.clone()

        # We'll keep track of how many new tokens we've added
        generated_count = 0

        while generated_count < max_new_tokens:
            patch_len = compute_patch_size(all_tokens, threshold=threshold, max_patch=max_patch_size)
            # Avoid going beyond max_new_tokens
            patch_len = min(patch_len, max_new_tokens - generated_count)
            if patch_len <= 0:
                break

            current_seq_len = all_tokens.size(1)
            # The model will produce logits for positions [current_seq_len .. current_seq_len + patch_len - 1]
            # We can do that in a single pass by passing the entire tokens so far, extended with 
            # some placeholder for the next 'patch_len' bytes. We'll fill them in step-by-step from the logits.

            fill_value = 0  # default
            if hasattr(self.tok_embeddings, "padding_idx") and (self.tok_embeddings.padding_idx is not None):
                fill_value = self.tok_embeddings.padding_idx

            # We'll create a new input, same as all_tokens, with 'patch_len' dummy tokens appended (e.g. pad_id).
            # shape => [1, current_seq_len + patch_len]
            padded_input = torch.cat([
                all_tokens, 
                torch.full(
                    (1, patch_len), 
                    fill_value=fill_value, 
                    device=all_tokens.device, 
                    dtype=all_tokens.dtype)
            ], dim=1)

            # Now do a forward pass
            with torch.no_grad():
                # The model returns logits for each position in [B, T, 256]
                logits = self.forward(padded_input)  # shape [1, T, 256]
                # Chunked.
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)  # [1, T, 256]
            
            new_positions_logits = logits[0, current_seq_len : current_seq_len + patch_len, :]

            # Inside a patch do step-by-step sampling from the stored logits
            new_tokens = []
            for i in range(patch_len):
                step_logits = new_positions_logits[i]  # shape [256]
                # Possibly adjust for temperature
                if temperature != 1.0:
                    step_logits = step_logits / temperature
                
                # Possibly do top_k
                if top_k > 0:
                    # topk filter
                    values, indices = torch.topk(step_logits, top_k)
                    topk_mask = torch.ones_like(step_logits, dtype=torch.bool)
                    topk_mask[indices] = False
                    step_logits[topk_mask] = float('-inf')

                probs = step_logits.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                new_tokens.append(next_token)

                # If EOS, we can end generation altogether
                if next_token == eos_id:
                    patch_len = i + 1  # we actually only used i+1 steps
                    break

            new_tokens_t = torch.tensor(new_tokens, dtype=all_tokens.dtype, device=all_tokens.device).unsqueeze(0)
            all_tokens = torch.cat([all_tokens, new_tokens_t], dim=1)
            generated_count += len(new_tokens)

            # If we hit eos in the middle of the patch, break
            if new_tokens and new_tokens[-1] == eos_id:
                break

        return all_tokens


############################
# Tokenizer
############################

class ByteLatentModelTokenizer(nn.Module):
    def __init__(
        self,
        *,
        bpe_delim: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[Any] = None,
    ):
        super().__init__()
        if special_tokens is None:
            special_tokens = {
                "<|bos|>": BOS_ID,
                "<|eos|>": EOS_ID,
                "<|pad|>": PAD_ID,
            }
        self.special_tokens = special_tokens
        self.bos_id = self.special_tokens.get("<|bos|>", BOS_ID)
        self.eos_id = self.special_tokens.get("<|eos|>", EOS_ID)
        self.pad_id = self.special_tokens.get("<|pad|>", PAD_ID)
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        # naive UTF-8 byte approach:
        byte_data = list(bytes(text, encoding="utf-8", errors="ignore"))
        tokens = byte_data
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        # naive decode
        return bytes([t for t in tokens if t < 256]).decode("utf-8", errors="ignore")


    def tokenize_messages(self, messages: List[Dict[str, Any]], add_eos: bool = True) -> Tuple[List[int], List[bool]]:
        tokenized_messages = []
        mask = []
        for message in messages:
            if message.role != "ipython":
                tokenized_messages.extend(self.encode("".join([message.role, "\n"]), add_bos=False, add_eos=False))
            for item in message.content:
                if item['type'] == "text":
                    tokenized_messages.extend(
                        self.encode(item['content'], add_bos=False, add_eos=False)
                    )
        if add_eos:
            tokenized_messages.append(self.eos_id)
        mask = [False] * len(tokenized_messages)
        if self.max_seq_len:
            tokenized_messages = tokenized_messages[: self.max_seq_len]
            mask = mask[: self.max_seq_len]
        return tokenized_messages, mask

    def forward(self, sample: Mapping[str, Any], inference: bool = False) -> Mapping[str, Any]:
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample

############################
# Helper constructor
############################

def blt_tokenizer(
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[Any] = None,
    **kwargs,
) -> ByteLatentModelTokenizer:
    return ByteLatentModelTokenizer(
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        **kwargs,
    )

# Define Qwen 2.5 base model with additional layers, we will expect this to be loaded 
# with a pretrained Qwen checkpoint which should match. 
def qwen2_5_blt(
    # Warm up decoder/encoder. Tried it with both 100 and 0 and it seemed to work better without
    # the warmup, but never tested a longer warmup. 
    freeze_global_for_n_steps=0, 
    local_to_global_dim_proj=True,
) -> ByteLatentQwen2p5Decoder:
    qwen_cfg = dict(
        vocab_size=151936, # Kinda irrelevant
        embed_dim=2048,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        max_seq_len=32768,
        intermediate_dim=11008,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )

    local_enc_cfg = dict(
        vocab_size=256, 
        embed_dim=2048, # Keeping same as Qwen
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=4096,
        hidden_dim=4096,
        norm_eps=1e-5,
    )

    return ByteLatentQwen2p5Decoder(
        qwen_cfg=qwen_cfg,
        local_encoder_cfg=local_enc_cfg,
        patch_size=4,
        patching_threshold=3.0,
        freeze_global_for_n_steps=freeze_global_for_n_steps,
        local_to_global_dim_proj=local_to_global_dim_proj,
        # Cross-attending every other layer to reduce memory usage. 
        # cross_attend_layers=list(range(0, qwen_cfg["num_layers"], 2))
        cross_attend_layers=list(range(0, qwen_cfg["num_layers"], 3))
        # Alt last 6 layers
        # cross_attend_layers=range(qwen_cfg["num_layers"] - 6, qwen_cfg["num_layers"])
    )
