from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from src.utils import *
import open_clip
import os, json
import gc 

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class WolfConfig(LlamaConfig):
    model_type = "wolf"
    multi_head = False
    use_mm_proj: bool = True
    mm_vision_tower: str = "openai/clip-vit-large-patch14"
    mm_hidden_size: int = 1024

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import logger, _CONFIG_FOR_DOC, LLAMA_INPUTS_DOCSTRING

class Llamatwin(LlamaModel):

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(Llamatwin, self).__init__(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attention_mask_point_list = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """From transformers.models.llama.modeling_llama"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask_point_list is not None:
            attention_mask = self.lego_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, attention_mask_point_list=attention_mask_point_list
            )
        
        elif self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        
        
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
            
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def lego_4d_causal_attention_mask_for_sdpa(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Union[torch.Size, Tuple, List],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        attention_mask_point_list=None
    ):
        """Overriding
        .../site-packages/transformers/modeling_attn_mask_utils.py
        def _prepare_4d_causal_attention_mask_for_sdpa
        """
        attn_mask_converter = WolfAttentionMaskConverter(is_causal=True, 
                                                         sliding_window=sliding_window, 
                                                         attention_mask_point_list=attention_mask_point_list)
        key_value_length = input_shape[-1] + past_key_values_length
        batch_size, query_length = input_shape
        is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy)

        if attention_mask_point_list is not None:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype=inputs_embeds.dtype,
                key_value_length=key_value_length,
            )
            if query_length > 1 and not is_tracing:
                expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                    expanded_4d_mask, attention_mask, unmasked_value=0.0
                )
            return expanded_4d_mask

        if attention_mask is not None:
            # 4d mask is passed through
            if len(attention_mask.shape) == 4:
                expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
                if tuple(attention_mask.shape) != expected_shape:
                    raise ValueError(
                        f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
                else:
                    # if the 4D mask has correct shape - invert it and fill with negative infinity
                    inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                    attention_mask = inverted_mask.masked_fill(
                        inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                    )
                    return attention_mask

            elif not is_tracing and torch.all(attention_mask == 1):
                if query_length == 1:
                    # For query_length == 1, causal attention and bi-directional attention are the same.
                    attention_mask = None
                elif key_value_length == query_length:
                    attention_mask = None
                else:
                    # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                    # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                    # Reference: https://github.com/pytorch/pytorch/issues/108108
                    pass
        elif query_length > 1 and key_value_length != query_length:
            # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
            # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
            attention_mask = True
        elif is_tracing:
            raise ValueError(
                'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
            )

        if attention_mask is None:
            expanded_4d_mask = None
        elif attention_mask is True:
            expanded_4d_mask = attn_mask_converter.to_causal_4d(
                input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
        else:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype=inputs_embeds.dtype,
                key_value_length=key_value_length,
            )

            # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
            # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
            #
            # This fix is not applied in case we are tracing with torch.jit.trace or symbolic_trace, as _unmask_unattended has a data-dependent
            # controlflow that can not be captured properly.
            # TODO: _unmask_unattended does not work either with torch.compile when using fullgraph=True. We should find a way to detect this case.
            if query_length > 1 and not is_tracing:
                expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                    expanded_4d_mask, attention_mask, unmasked_value=0.0
                )

        return expanded_4d_mask
    
class WolfAttentionMaskConverter(AttentionMaskConverter):

    def __init__(self, is_causal: bool, sliding_window: int | None = None, attention_mask_point_list=None):
        super().__init__(is_causal, sliding_window)
        if attention_mask_point_list is not None:
            self.attention_mask_point_list = attention_mask_point_list
        else:
            self.attention_mask_point_list = None

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Overridding
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        if self.attention_mask_point_list is None:
            return super().to_4d(attention_mask_2d, query_length, dtype, key_value_length)
        
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError("This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask.")

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(attention_mask_2d.device)
        expanded_attn_mask = self._expand_mask_with_intervals(attention_mask_2d, dtype, tgt_len=input_shape[-1], intervals=self.attention_mask_point_list).to(attention_mask_2d.device)
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)  # torch.finfo(dtype).min == almost -inf

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    def _expand_mask_with_intervals(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: int, intervals: torch.Tensor) -> torch.Tensor:
        """
        Expands 2D attention mask to 4D attention mask and applies masking based on provided intervals.
        
        Parameters:
        - mask: The base 2D attention mask tensor.
        - dtype: Data type of the output mask.
        - tgt_len: The target sequence length.
        - intervals: A tensor of shape [batch, N, 3] containing interval information.
        
        Returns:
        - A 4D attention mask tensor.
        """
        bsz, src_len = mask.size()
        device = mask.device
        # Initialize the expanded mask with zeros
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        # expanded_mask = torch.zeros((bsz, 1, tgt_len, src_len), dtype=dtype, device=device)

        # Apply intervals to mask
        for batch_idx in range(bsz):
            for interval in intervals[batch_idx]: # 7 intervals 
                # when first interval a == b
                a, b, c = interval
                # Masking tokens from a to b-1 for target tokens from b to c
                if b <= c:
                    expanded_mask[batch_idx, 0, b:c, a:b] = torch.finfo(dtype).min # masking

        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
        """Overriding
        Expands 2D attention mask to 4D attention mask by adding a large negative bias to not-attended positions.
        """
        mask_point_list = self.attention_mask_point_list
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class WolfLlamaModel(Llamatwin):
    config_class = WolfConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(WolfLlamaModel, self).__init__(config)

        self.vision_tower_name = (
            "openai/clip-vit-large-patch14"  # microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 # openai/clip-vit-large-patch14
        )
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP

            if "BiomedCLIP" in config.mm_vision_tower or "biomed_clip" in config.mm_vision_tower:
                model, _, _ = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
                self.vision_tower = [
                    model.visual.trunk
                ]  # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18
                self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            else:
                self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        attention_mask_point_list=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for Wolf pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)  # initial embeddings for text token
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # initial embeddings for text token

        vision_tower = getattr(self, "vision_tower", None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_feature, dummy_image_features = self.extract_visual_features(vision_tower, image.unsqueeze(0))
                        image_features.append(image_feature)
                else:
                    image_features, dummy_image_features = self.extract_visual_features(vision_tower, images)

            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)

            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0.0 * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]  # cur_image_idx == batch_idx
                    num_patches = cur_image_features.shape[0]  # 256
                    # find image token has or not
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]  # where img start

                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]

                        # import pdb; pdb.set_trace()
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            # concatenate image features with text embeddings
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:image_start_token_pos].detach(),
                                    cur_input_embeds[image_start_token_pos : image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[image_start_token_pos + num_patches + 1 : image_start_token_pos + num_patches + 2],
                                    cur_input_embeds[image_start_token_pos + num_patches + 2 :].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[image_start_token_pos + num_patches + 1 :],
                                ),
                                dim=0,
                            )
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(mask_index_start, mask_index_start + num_patches, device=masked_indices.device, dtype=masked_indices.dtype)
                    ).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_image_features,
                                cur_input_embeds[mask_index_start + num_patches :].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start + num_patches :]), dim=0
                        )
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(WolfLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask_point_list=attention_mask_point_list,
        )
        
    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        if "BiomedCLIP" in vision_tower:
            self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            return self.initialize_vision_modules_from_biomed_clip(
                vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False
            )
        else:
            return self.initialize_vision_modules_from_openai_clip(
                vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False
            )

    def initialize_vision_modules_from_openai_clip(
        self, vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, "vision_tower"):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            self.mm_projector.load_state_dict({k.split(".")[-1]: v for k, v in mm_projector_weights.items()})

        return dict(image_processor=image_processor, image_token_len=num_patches, vision_config=vision_config)

    def initialize_vision_modules_from_biomed_clip(
        self, vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        vision_config = openai_vision_tower.config
        del openai_vision_tower

        if not hasattr(self, "vision_tower"):
            model, _, _ = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            vision_tower = (
                model.visual.trunk
            )  # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18

            # from huggingface_hub import snapshot_download
            # BiomedCLIP_file_path = "biomed-clip-share"
            # # snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir=BiomedCLIP_file_path)
            # with open(os.path.join(BiomedCLIP_file_path, "open_clip_config.json"), 'r') as file:
            #     config = json.load(file)

        else:
            vision_tower = self.vision_tower[0]

        setattr(vision_tower, "config", vision_config)
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            self.mm_projector.load_state_dict({k.split(".")[-1]: v for k, v in mm_projector_weights.items()})

        return dict(image_processor=image_processor, image_token_len=num_patches, vision_config=vision_config)

    def extract_visual_features(self, vision_tower, images):
        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)

        if "BiomedCLIP" in self.vision_tower_name or "biomed_clip" in self.vision_tower_name:
            image_forward_outs = vision_tower.get_intermediate_layers(
                images, n=3
            )  # take last n blocks if n is an int, if in is a sequence, select by matching indices
            image_features = image_forward_outs[select_hidden_state_layer]
            image_features = image_features
            dummy_image_features = torch.zeros(196, 768, device=image_features.device, dtype=image_features.dtype)
        else:
            image_forward_outs = vision_tower(images, output_hidden_states=True)
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:]
            dummy_image_features = torch.zeros(256, 1024, device=image_features.device, dtype=image_features.dtype)

        return image_features, dummy_image_features


    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class WolfLlamaForCausalLM(LlamaForCausalLM):
    config_class = WolfConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = WolfLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        attention_mask_point_list = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            attention_mask_point_list=attention_mask_point_list,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # compute loss
        loss = None
        if labels is not None:
            # Auto-regressive model
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        torch.cuda.empty_cache()
        gc.collect()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device, tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.model.vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # for p in self.get_output_embeddings().parameters():
                # p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def initialize_vision_modules(
            self,
            vision_tower,
            mm_vision_select_layer,
            pretrain_mm_mlp_adapter,
        ):
        return self.model.initialize_vision_modules(vision_tower, mm_vision_select_layer, pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter)

    def get_vision_tower(self):
        return self.model.vision_tower[0]

    def get_model(self):
        return self.model

AutoConfig.register("wolf", WolfConfig)
AutoModelForCausalLM.register(WolfConfig, WolfLlamaForCausalLM)
