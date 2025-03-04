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

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
NUM_SPECIAL_TOKENS = 3
VOCAB_SIZE = 256 + NUM_SPECIAL_TOKENS  

################################################
# Local encoder/decoder (with cross-attn)
################################################

class HashNGramEmbedder(nn.Module):
    """
    Wraps a main byte embedding plus additional hash-based
    n-gram embedding lookups. Two modes are supported:
    
    1. Separate tables mode (default): One embedding table per n in [3, max_n].
    2. Shared table mode (if shared_table=True): A single embedding table is used
       for all n-gram sizes.
       
    When using the shared table mode, a learned n-gram size embedding is added so that the model can
    distinguish among n-gram lengths.
    
    The final embedding at each position is:
       main_embed(byte) + sum_{n in 3..max_n} (ngram_embed [+ ngram_size_embed if shared])
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        max_n: int = 8,
        num_buckets: int = 500_000,
        vocab_size: int = VOCAB_SIZE,
        hash_base: int = 0,
        hash_mod: int = 2**23,
        shared_table: bool = True  # switchable mode
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_n = max_n
        self.num_buckets = num_buckets
        self.hash_base = hash_base
        if (self.hash_base == 0):
            self.hash_base = vocab_size + 1
        self.hash_mod = hash_mod
        self.shared_table = shared_table

        # Main byte embedding.
        self.main_embed = nn.Embedding(vocab_size, embed_dim)

        if shared_table:
            # One shared table for all n-gram sizes.
            self.shared_ngram_table = nn.Embedding(num_buckets, embed_dim)
            nn.init.normal_(self.shared_ngram_table.weight, mean=0.0, std=0.02)
            # n-gram size embedding: one learned vector per n in [3, max_n]
            # (max_n - 2) distinct n values.
            self.ngram_size_embed = nn.Embedding(max_n - 2, embed_dim)
            nn.init.normal_(self.ngram_size_embed.weight, mean=0.0, std=0.02)
        else:
            # Separate table per n-gram size.
            self.ngram_tables = nn.ModuleDict()
            for n in range(3, max_n + 1):
                table = nn.Embedding(num_buckets, embed_dim)
                nn.init.normal_(table.weight, mean=0.0, std=0.02)
                self.ngram_tables[str(n)] = table

        # Precompute powers for rolling hash for each n in [3, max_n] using modular exponentiation.
        for n in range(3, max_n + 1):
            powers_list = [pow(hash_base, n - 1 - k, hash_mod) for k in range(n)]
            powers = torch.tensor(powers_list, dtype=torch.int32)
            self.register_buffer(f'powers_{n}', powers)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: [batch_size, seq_len] of byte IDs in [0..255].
        Returns final embeddings of shape [batch_size, seq_len, embed_dim].
        All hash arithmetic is done in int32.
        """
        bsz, seq_len = tokens.shape
        if seq_len == 0:
            return torch.empty(bsz, 0, self.embed_dim, device=tokens.device)

        # Main byte embedding.
        out = self.main_embed(tokens)  # [bsz, seq_len, embed_dim]

        # For each n-gram size, compute and add n-gram embeddings.
        for n in range(3, self.max_n + 1):
            if seq_len < n:
                continue

            powers = getattr(self, f'powers_{n}')  # shape: [n]
            ngrams = tokens.unfold(1, n, 1).to(torch.int32)  # [bsz, seq_len - n + 1, n]
            if ngrams.numel() == 0:
                continue

            temp = (ngrams * powers.unsqueeze(0).unsqueeze(0)) % self.hash_mod
            hashed_vals = temp.sum(dim=2) % self.hash_mod  # [bsz, seq_len - n + 1]

            hashed_idxs = torch.zeros((bsz, seq_len), dtype=torch.int32, device=tokens.device)
            hashed_idxs[:, n - 1:] = hashed_vals
            hashed_idxs = hashed_idxs % self.num_buckets

            if self.shared_table:
                ngram_embed = self.shared_ngram_table(hashed_idxs.long())  # [bsz, seq_len, embed_dim]
                # Add n-gram size embedding only in shared mode.
                size_embed = self.ngram_size_embed(torch.tensor(n - 3, device=tokens.device))
                size_embed = size_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
                ngram_embed = ngram_embed + size_embed
            else:
                table = self.ngram_tables[str(n)]
                ngram_embed = table(hashed_idxs.long())

            out += ngram_embed

        num_contrib = 1 + (self.max_n - 2)  # main embedding + contributions for each n from 3 to max_n
        out = out / num_contrib
        return out

class LocalDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        global_dim: int,
        vocab_size: int,
        num_layers: int = 8,
        num_cross_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        hidden_dim: int = 4096,
        norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        max_seq_len: int = 4096,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        
        # Self-attention layers
        layers = nn.ModuleList()
        for _ in range(num_layers):
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
                is_causal=True,
            )
            mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(embed_dim, eps=norm_eps),
                mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
            )
            layers.append(layer)
        
        # Decoder with no token embeddings
        self.decoder = TransformerDecoder(
            tok_embeddings=nn.Identity(),  # No embedding layer needed
            layers=layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(embed_dim, eps=norm_eps),
            output=nn.Identity(),
        )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            cross_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
                k_proj=nn.Linear(global_dim, num_kv_heads * head_dim, bias=True),
                v_proj=nn.Linear(global_dim, num_kv_heads * head_dim, bias=True),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
                is_causal=False,
            )
            mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            cross_layer = TransformerCrossAttentionLayer(
                attn=cross_attn,
                mlp=mlp,
                ca_norm=RMSNorm(embed_dim, eps=norm_eps),
                mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
            )
            self.cross_attn_layers.append(cross_layer)
        
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.to(dtype=dtype)
    
    def forward(self, byte_embeds, patch_embs, patch_ids):
        # Pass byte embeddings directly to the decoder
        x = self.decoder(byte_embeds, mask=None)  # Shape: [batch_size, seq_len, embed_dim]
        x = x.to(torch.bfloat16) # TODO: Fix cast because no chunking.

        # Apply cross-attention with patch embeddings
        if self.cross_attn_layers:
            num_patches = patch_embs.size(1)
            mask = (
                patch_ids.unsqueeze(2) == torch.arange(num_patches, device=byte_embeds.device).unsqueeze(0).unsqueeze(0)
            )  # Shape: [batch_size, num_patches, seq_len]
            for cross_layer in self.cross_attn_layers:
                x = cross_layer(x, encoder_input=patch_embs, encoder_mask=mask)
        
        # Compute logits
        logits = self.output(x)  # Shape: [batch_size, seq_len, vocab_size]
        return logits

class LocalEncoderWithPooling(nn.Module):
    def __init__(self, base_encoder, cross_attn_layers, embed_dim, global_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.cross_attn_layers = cross_attn_layers
        self.patch_projector = PatchToGlobalProjector(embed_dim, global_dim) 

    def forward(self, bytes, patch_ids):
        # Get byte-level embeddings from the base encoder
        byte_embeds = self.base_encoder(bytes, mask=None)
        byte_embeds = byte_embeds.to(torch.bfloat16)

        # Apply cross-attention if layers exist
        if self.cross_attn_layers:
            # num_patches = patch_ids.max().item() + 1
            local_encoder_mask = (patch_ids.unsqueeze(1) == patch_ids.unsqueeze(2))  # Shape: [batch_size, seq_len, seq_len]
            patch_embs = byte_embeds  # Start with byte embeddings
            for cross_layer in self.cross_attn_layers:
                patch_embs = cross_layer(patch_embs, encoder_input=byte_embeds, encoder_mask=local_encoder_mask)
        else:
            patch_embs = byte_embeds

        # Reduce byte-level embeddings to patch-level embeddings
        patch_embs = patch_reduce(patch_embs, patch_ids, reduce_op="mean")  # Shape: [batch_size, num_patches, embed_dim]

        # Project to global dimension
        patch_embs = self.patch_projector(patch_embs)  # Shape: [batch_size, num_patches, global_dim]

        return byte_embeds, patch_embs  # Return both byte embeddings and reduced patch embeddings

def build_local_encoder(
    global_dim: int,
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 8,
    hidden_dim: int = 4096,
    norm_eps: float = 1e-5,
    attn_dropout: float = 0.0,
    max_seq_len: int = 2048,
    num_layers: int = 4,
    num_cross_layers = 4,
    dtype=torch.bfloat16,
    use_hash_ngrams=True,
    max_ngram: int = 8, 
    num_ngram_buckets: int = 500000, 
):
    head_dim = embed_dim // num_heads

    if use_hash_ngrams:
        tok_embeddings = HashNGramEmbedder(
            embed_dim=embed_dim,
            max_n=max_ngram,
            num_buckets=num_ngram_buckets,
            vocab_size=vocab_size
        )
    else:
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
            is_causal=True,
        )
        mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    base_encoder = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=nn.Identity(),  # no final projection
    )

    # Cross-attention layers (optional)
    cross_attn_layers = nn.ModuleList()
    for _ in range(num_cross_layers):
        cross_attn = MultiHeadAttention(
            embed_dim=global_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(global_dim, num_heads * head_dim, bias=True),
            k_proj=nn.Linear(global_dim, num_kv_heads * head_dim, bias=True),
            v_proj=nn.Linear(global_dim, num_kv_heads * head_dim, bias=True),
            output_proj=nn.Linear(global_dim, global_dim, bias=False),
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            is_causal=False,
        )
        mlp = qwen2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        cross_layer = TransformerCrossAttentionLayer(
            attn=cross_attn,
            mlp=mlp,
            ca_norm=RMSNorm(global_dim, eps=norm_eps),
            mlp_norm=RMSNorm(global_dim, eps=norm_eps),
        )
        cross_attn_layers.append(cross_layer)
    
    local_encoder = LocalEncoderWithPooling(base_encoder, cross_attn_layers, embed_dim, global_dim)
    return local_encoder.to(dtype=dtype)

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
    freq = torch.zeros(batch_size, VOCAB_SIZE, device=device)
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
            VOCAB_SIZE  ,  # bytes
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

        output = nn.Identity()  # Global transformer doesn’t output logits
        emb = nn.Identity()
        super().__init__(
            tok_embeddings=emb,
            layers=layers,
            max_seq_len=qwen_cfg['max_seq_len'],
            num_heads=qwen_cfg['num_heads'],
            head_dim=head_dim,
            norm=RMSNorm(qwen_cfg['embed_dim'], eps=qwen_cfg['norm_eps']),
            output=output,
        )
        self.local_encoder = build_local_encoder(**local_encoder_cfg, global_dim=qwen_cfg['embed_dim'])
        self.local_decoder = LocalDecoder(
            embed_dim=qwen_cfg['embed_dim'],
            global_dim=qwen_cfg['embed_dim'],
            vocab_size=VOCAB_SIZE,
            max_seq_len=local_encoder_cfg['max_seq_len'],
        )

         # Collect parameters for different learning rates
        self.qwen_params = []
        self.new_params = []

        # Qwen parts
        self.qwen_params.extend(self.norm.parameters())
        for layer in self.layers:
            if isinstance(layer, TransformerSelfAttentionLayer):
                self.qwen_params.extend(layer.parameters())

        # New parts
        self.new_params.extend(self.tok_embeddings.parameters())
        self.new_params.extend(self.output.parameters())
        self.new_params.extend(self.local_encoder.parameters())
        self.new_params.extend(self.local_decoder.parameters())

        self.patch_size = patch_size
        self.patching_threshold = patching_threshold
        self.freeze_global_for_n_steps = freeze_global_for_n_steps
        self.current_step = 0
        self.global_frozen = freeze_global_for_n_steps > 0
        if self.global_frozen:
            self._update_freezing()

        # We'll store how many chunks the user wants for final output
        self.num_output_chunks = 0  # default

    def _update_freezing(self):
        if self.global_frozen:
            for param in self.qwen_params:
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

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
        patch_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Update freezing state if in training
        if self.training and self.global_frozen:
            self.current_step += 1
            if self.current_step >= self.freeze_global_for_n_steps:
                self.global_frozen = False
                self._update_freezing()

        if patch_ids is None:
            patch_ids, _ = dynamic_patch(tokens, threshold=self.patching_threshold, patch_size=self.patch_size)
    
        byte_embeds, patch_embs = self.local_encoder(tokens, patch_ids=patch_ids)
        global_out = super().forward(patch_embs, mask=mask, input_pos=input_pos)
        
        # Assuming the outs are chunked, take entry[0] of outputs.
        if self.num_output_chunks == 0:
            global_out = global_out.to(torch.bfloat16) # TODO: Another fix for the 0 chunk float. Need to move this dtype definiton.  
            logits = self.local_decoder(byte_embeds, global_out, patch_ids)
        else: 
            global_out_combined = torch.cat(global_out, dim=1) # Unchunking - it would be better not to pass this through I think. 
            clogits = self.local_decoder(byte_embeds, global_out_combined, patch_ids)
            logits = [chunk for chunk in clogits.chunk(self.num_output_chunks, dim=1)]
        return logits

    def unified_generate(
        self,
        prompt: Union[torch.LongTensor, List[int]],
        max_new_tokens: int = 128,
        eos_id: int = EOS_ID,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0,
        frequency_penalty: float = 0.1,
        repetition_penalty: float = 1.5,
        greedy: bool = False,
    ) -> torch.Tensor:
        # 1) Convert prompt to a tensor [1, seq_len]
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device=next(self.parameters()).device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0) # Add a batch dimension.

        device = prompt.device
        all_tokens = prompt.clone()

        # Track usage for freq / repetition penalty
        token_counts = torch.bincount(all_tokens[0], minlength=VOCAB_SIZE).to(device)
        recent_tokens = []

        # Start generating
        for step_i in range(max_new_tokens):

            # No patching => standard forward
            with torch.no_grad():
                logits = self.forward(all_tokens)

            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
            step_logits = logits[0, -1, :]

            # Apply temperature
            if temperature != 1.0:
                step_logits = step_logits / temperature

            # Repetition penalty
            for tk in recent_tokens:
                step_logits[tk] /= repetition_penalty

            # Frequency penalty
            step_logits -= token_counts * frequency_penalty

            # top-k
            if top_k > 0:
                vals, idxs = torch.topk(step_logits, top_k)
                mask = torch.ones_like(step_logits, dtype=torch.bool)
                mask[idxs] = False
                step_logits[mask] = float('-inf')

            # top-p
            probs = torch.softmax(step_logits, dim=-1)
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                step_logits = step_logits.scatter(-1, indices_to_remove, -1e9)

            # Re‐normalize            
            total_p = probs.sum()
            if total_p > 0:
                probs /= total_p
            else:
                probs = torch.ones_like(probs) / probs.size(-1)

            # Next token
            if greedy:
                next_token = torch.argmax(probs).item()
            else:
                next_token = torch.multinomial(probs, 1).item()

            # Add new token
            all_tokens = torch.cat(
                [all_tokens, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

            # Update counters
            token_counts[next_token] += 1
            recent_tokens.append(next_token)
            if len(recent_tokens) > 5:
                recent_tokens.pop(0)

            # EOS break
            if next_token == eos_id:
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
        # naive decode - skip special tokens
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
    freeze_global_for_n_steps=500,
    use_hash_ngrams=True,
    patch_size=4,
    max_seq_len=4096
) -> ByteLatentQwen2p5Decoder:
    qwen_cfg = dict(
        vocab_size=151936, # Kinda irrelevant
        embed_dim=2048,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
        intermediate_dim=11008,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )

    local_enc_cfg = dict(
        vocab_size=VOCAB_SIZE, 
        embed_dim=2048, # Keeping same as Qwen
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=max_seq_len,
        hidden_dim=4096,
        norm_eps=1e-5,
        num_cross_layers = 4,
        use_hash_ngrams=use_hash_ngrams,
        max_ngram= 8, 
        num_ngram_buckets=500000, 
    )

    return ByteLatentQwen2p5Decoder(
        qwen_cfg=qwen_cfg,
        local_encoder_cfg=local_enc_cfg,
        patch_size=patch_size,
        patching_threshold=3.0,
        freeze_global_for_n_steps=freeze_global_for_n_steps
    )
