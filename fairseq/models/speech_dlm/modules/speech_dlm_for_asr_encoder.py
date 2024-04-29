import math
import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.speech_dlm.modules.speech_dlm_decoder_layer import (
    CrossChannelTransformerDecoderLayer,
    StandardTransformerDecoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.data.dictionary import Dictionary
from fairseq import utils

from typing import Dict, List, Optional, Tuple



class CrossChannelTransformerEncoderForASR(FairseqEncoder):
    """
    Cross-channel Transformer Decoder Block for parallel spoken dialogue units
    as described in the paper: https://arxiv.org/pdf/2203.16502.pdf, but for ASR;
    consisting of *args.decoder_layers* layers. Each layer is a
    :class:`StandardTransformerDecoderLayer` or
    :class:`CrossChannelTransformerDecoderLayer`.

    The "ForASR"-version is almost identical as the original, except that it
    inherits from FairseqEncoder instead of FairseqIncrementalDecoder.
    While the original implementation was designed for autoregressive generation,
    the ForASR version uses only a single forward pass thus it implements the
    FairseqEncoder-class.

    Also, the "ForASR"-version uses different input and output dictionaries as
    well as different embeddings for input and output.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        input_dictionary (~fairseq.data.Dictionary): input dictionary
        output_dictionary (~fairseq.data.Dictionary): output dictionary
        embedding_tokens (torch.nn.Embedding): input embedding
        channels (list): list of channel names (string)
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self,
            args,
            input_dictionary: Dictionary,
            output_dictionary: Dictionary,
            embed_tokens: nn.Embedding,
            channels: List[str],
            no_encoder_attn=False
        ):
        self.args = args
        super().__init__(input_dictionary)
        self.embed_tokens = embed_tokens
        self.channels = channels
        self.no_encoder_attn = no_encoder_attn
        self.apply_casual_mask = args.apply_causal_mask
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        # No output embeddings for ASR
        self.channels = channels

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        assert 0 <= args.decoder_cross_layers <= args.decoder_layers, (
            "The number of cross-channel attention decoder layers must be non-negative"
            f"and not exceeds the number of decoder layers (found {args.decoder_cross_layers})"
        )

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                if i < args.decoder_layers - args.decoder_cross_layers
                else self.build_cross_decoder_layer(args, no_encoder_attn)
                for i in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)
        self.non_cross_layers = args.decoder_layers - args.decoder_cross_layers

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            nn.Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim
            else None
        )

        self.output_projection = nn.Linear(self.output_embed_dim, len(output_dictionary), bias=False)
        nn.init.normal_(
            self.output_projection.weight,
            mean=0,
            std=self.output_embed_dim**-0.5,
        )


    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = StandardTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def build_cross_decoder_layer(self, args, no_encoder_attn=False):
        layer = CrossChannelTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer
    
    def forward(
            self,
            src_tokens: Dict[str, torch.LongTensor],
            src_lengths: Optional[Dict[str, List[torch.LongTensor]]] = None,
            features_only: bool = False,
            alignment_layer: Optional[int] = None, # remove???
            alignment_heads: Optional[int] = None, # remove???
            **kwargs
        ):
        """
        Args:
            src_tokens (Dict[str, torch.LongTensor]): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (Optional[Dict[str, List[torch.LongTensor]]]): lengths of each source sentence of shape
                `(batch)`
            alignment_layer (int, optional): return mean alignment over heads at this layer (default: last layer).
            alignment_heads (int, optional): return mean alignment over this many heads (default: all heads).

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: the output of shape
                `(src_len, batch, tgt_dict_len)` and the extra dictionary with
                the following optional items:
                - "attn" (List[Dict[str, Optional[torch.Tensor]]]): a list of
                    attention weights (one per layer) of shape `(batch, src_len, src_len)`
                - "inner_states" (List[Dict[str, Optional[torch.Tensor]]]): a list of
                    the intermediate decoder states (one per layer)
        """
        x, extra = self.extract_features(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

            
    
    def extract_features(
            self,
            src_tokens: Dict[str, torch.LongTensor],
            src_lengths: Optional[Dict[str, List[torch.LongTensor]]] = None,
            alignment_layer: Optional[int] = None, # remove???
            alignment_heads: Optional[int] = None, # remove???
            **kwargs
        ):
        """
        Args:
            src_tokens (Dict[str, torch.LongTensor]): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (Optional[Dict[str, List[torch.LongTensor]]]): lengths of each source
                sentence of shape `(batch)`
            #TODO full_context_alignment (bool, optional): don't apply auto-regressive mask
                to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over heads at this layer
                (default: last layer).
            alignment_heads (int, optional): return mean alignment over this many heads
                (default: all heads).

        Returns:
            Tuple(torch.Tensor, Dict[str, Optional[torch.Tensor]]): the encoder features of shape
                `(src_len, batch, embed_dim)` and the extra dictionary with
                the following optional items:
                - "attn" (List[Dict[str, torch.Tensor]]): a list of
                    attention weights
                - "inner_states" (List[Dict[str, torch.Tensor]]): a list of
                    the intermediate decoder states
                - "self_attn_mask" (Dict[str, torch.Tensor]): a dict of
                    self attention mask
                - "padding_mask" (Dict[str, torch.Tensor]): a dict of
                    padding mask
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        x_list = []
        for i, channel in enumerate(self.channels):
            x = self.embed_tokens(src_tokens[channel])

            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
            
            x = self.embed_scale * x
            
            if self.quant_noise is not None:
                x = self.quant_noise(x)

            if self.embed_positions is not None:
                x += self.embed_positions(src_tokens[channel])

            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            
            x = self.dropout_module(x)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            x_list.append(x)
        
        self_attn_padding_mask = None
        if src_tokens[channel[0]].eq(self.padding_idx).any():
            self_attn_padding_mask = src_tokens[channel[0]].eq(self.padding_idx)

        # decoder layers
        attn: Optional[Dict[torch.Tensor]] = None
        inner_states: List[Optional[Dict[str, torch.Tensor]]] = [
            {channel: x_list[i] for i, channel in enumerate(self.channels)}
        ]

        for idx, layer in enumerate(self.layers):
            if self.apply_casual_mask:
                self_attn_mask = self.buffered_future_mask(x_list[0])
            else:
                self_attn_mask = None

            # need to change to tensor for the checkpoint activation to work
            if isinstance(x_list, list):
                x_list = torch.stack(x_list) # C x T x B
                
            x_list, layer_attn_list, _ = layer(
                x_list,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_mask=self_attn_mask
            )

            inner_states.append(
                {channel: x_list[i] for i, channel in enumerate(self.channels)}
            )
            if idx == alignment_layer and all(
                layer_attn is not None for layer_attn in layer_attn_list
            ):
                attn = {
                    channel: layer_attn_list[i].float().to(x_list[0])
                    for i, channel in enumerate(self.channels)
                }
        
        # change back from tensor to list
        if not isinstance(x_list, list):
            x_list = list(torch.unbind(x_list))
        
        if attn is not None:
            for channel in attn:
                if alignment_heads is not None:
                    attn[channel] = attn[channel][:alignment_heads]

                # average probabilities over heads
                attn[channel] = attn[channel].mean(dim=0)
        
        for i, x in enumerate(x_list):
            if self.layer_norm is not None:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

            x_list[i] = x


        x = {channel: x_list[i] for i, channel in enumerate(self.channels)}

        return x, {"attn": [attn], "inner_states": inner_states, "self_attn_mask": self_attn_mask, "padding_mask": self_attn_padding_mask}
    
    def output_layer(self, features):
        """Project features to the vocabulary size.
        
        Args:
            features (Dict[str, torch.Tensor]): the decoder features of shape
        
        Returns:
            Dict[str, torch.Tensor]: the output class of shape
        """

        return {
            channel: self.output_projection(channel_features) for channel, channel_features in features.items()
        }

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


if __name__ == "__main__":
    from fairseq.models.speech_dlm.speech_dlm_for_asr import SpeechDLMForASRConfig, SpeechDLMForASR
    from fairseq.data import Dictionary
    from fairseq.tasks.speech_dlm_for_asr_task import SpeechDLMForASRTaskConfig
    from fairseq.dataclass.configs import FairseqConfig, CommonConfig
    from fairseq.criterions.speech_dlm_criterion import SpeechDLMCriterionConfig
    from omegaconf import OmegaConf
    from hydra.experimental import compose, initialize
    import os


    """
    config_path = os.path.join("..", "..", "..", "..", "fairseq", "config")
    #os.path.join("home", "studenter", "simendym", "Desktop", "multi-channel-asr", "fairseq", "config")
    with initialize(config_path=config_path):
        composed_cfg = compose(config_name="config")


    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )"""

    """with omegaconf_no_object_check():
        if cfg.task is None and getattr(args, "task", None):
            cfg.task = Namespace(**vars(args))
            from fairseq.tasks import TASK_REGISTRY

            _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
            cfg.task._name = args.task"""
    


    cfg = OmegaConf.structured(
        FairseqConfig()
    )

    common_cfg = OmegaConf.create(
        CommonConfig()
    )

    criterion_cfg = OmegaConf.create(
        SpeechDLMCriterionConfig()
    )

    model_cfg = OmegaConf.create(
        SpeechDLMForASRConfig()
    )
    model_cfg.decoder_layers = 6
    model_cfg.decoder_cross_layers = 4
    model_cfg._name = "model" #"speech_dlm_for_asr"

    task_cfg = OmegaConf.create(
        SpeechDLMForASRTaskConfig()
    )
    task_cfg.max_target_positions = 1024
    task_cfg._name = "task" #"speech_dlm_for_asr_task"

    cfg.common = common_cfg
    cfg.criterion = criterion_cfg
    cfg.model = model_cfg
    cfg.task = task_cfg

    print(cfg)

    print(cfg.model.max_target_positions)

    input_dictionary = Dictionary.load("model_files/dict.unitA.txt")
    output_dictionary = Dictionary.load("model_files/dict.ltr.txt")
    embed_tokens = SpeechDLMForASR.build_embedding(cfg.model, input_dictionary, embed_dim=cfg.model.decoder_embed_dim)
    channels = ["0", "1"]
    model = CrossChannelTransformerEncoderForASR(
        cfg.model,
        input_dictionary,
        output_dictionary,
        embed_tokens,
        channels
    )

    batch_size = 5
    src_length = 10

    src_tokens = {
        "0": torch.randint(0, 10, (batch_size, src_length)).long(),
        "1": torch.randint(0, 10, (batch_size, src_length)).long(),
    }

    x, _ = model(src_tokens)

    print("x", x.keys())
    print("input", src_tokens["0"].shape)
    print("output", x["0"].size())
    print("output", x)
    print("other stuff keys", _.keys())
    print("vocab size", len(output_dictionary))