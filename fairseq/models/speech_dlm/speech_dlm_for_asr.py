# This source code is based on fairseq/models/speech_dlm/speech_dlm.py
#
# Changes done by Simen Dymbe

import logging
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq import hub_utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    FairseqEncoderModel,
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import Embedding
from fairseq.models.speech_dlm.modules.speech_dlm_for_asr_encoder import CrossChannelTransformerEncoderForASR
from fairseq.tasks.speech_dlm_for_asr_task import SpeechDLMForASRTask
from omegaconf import II

from typing import Dict


DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@dataclass
class SpeechDLMForASRConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_cross_layers: int = field(
        default=4, metadata={"help": "num self cross attention decoder layers"}
    )
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    #
    # No shared embeddings, since ASR model outputs characters
    #
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    freeze_transformer_layers: str = field(
        default="",
        metadata={
            "help": "a comma-separated string of layer numbers to freeze (0-indexed)"
        },
    )
    freeze_input_embedding: bool = field(
        default=False, metadata={"help": "freeze input embeddings"}
    )
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions") # Is this needed?
    tpu: bool = II("common.tpu")
    # No duration_prediction
    # No delayed_duration_target
    #main_and_cross_weights: str = II("criterion.main_and_cross_weights") # Probably not needed
    build_from_dgslm: bool = field(
        default=False, metadata={"help": "build the model from a DGSLM model"}
    )
    dgslm_path: Optional[str] = field(
        default=None, metadata={"help": "path to the DGSLM model"}
    )
    dgslm_checkpoint_file: Optional[str] = field(
        default=None, metadata={"help": "checkpoint file for the DGSLM model"}
    )
    blank_weight: float = field(
        default=0.0, metadata={"help": "weight to add to the blank token"}
    )
    blank_mode: str = field(
        default="add", metadata={"help": "how to apply the blank weight"}
    )
    apply_causal_mask: bool = field(
        default=False, metadata={"help": "apply a casual mask to the attention scores"}
    )


@register_model("speech_dlm_for_asr", dataclass=SpeechDLMForASRConfig)
class SpeechDLMForASR(FairseqEncoderModel):
    """ASR model based on the Dialogue Language Model model (SpeechDLM)
    from: https://arxiv.org/pdf/2203.16502.pdf
    """
        
    def __init__(self, encoder: CrossChannelTransformerEncoderForASR, args: SpeechDLMForASRConfig):
        super().__init__(encoder)

        # CTC parameters
        self.blank_weight = args.blank_weight
        self.blank_mode = args.blank_mode

    def forward(self, src_tokens: Dict[str, torch.LongTensor], src_lengths: Dict[str, torch.Tensor], **kwargs):
        """
        Args:
            src_tokens (Dict[str, torch.LongTensor]): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (Dict[str, torch.Tensor]): source sentence lengths of shape `(batch)`

        Returns:
            encoder output of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)
    
    def prepare_logits(self, logits: Dict[str, torch.Tensor], padding_mask: Optional[torch.Tensor] = None):
        """
        Adds a weight to the blank token in the logits, and applies a padding mask if provided.
        """
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if padding_mask is not None and padding_mask.any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0

            if logits.size(0) > padding_mask.size(1):
                padding_mask = F.pad(
                    padding_mask, (1, 0), value=False
                )

            logits[padding_mask.T] = masking_tensor.type_as(logits)

        return logits
    
    def get_normalized_probs(self, output_dict: Dict[str, torch.Tensor], padding_mask: torch.Tensor, log_probs: bool) -> Dict[str, torch.Tensor]:
        """Get normalized probabilities (or log probs) from a net's output.
        
        Args:
            output_dict (Dict[str, torch.Tensor]): the output from the model
            padding_mask (Optional[torch.Tensor]): the padding mask to apply to the logits. It is the same
                for all channels, so it is passed as a single tensor
            log_probs (bool): whether to return log probabilities or not
        
        Returns:
            Dict[str, torch.Tensor]: the normalized probabilities (or log probs) for each channel
        """
        logits_dict = {}
        for channel, channel_output in output_dict.items():
            logits = self.prepare_logits(channel_output, padding_mask)

            if log_probs:
                logits_dict[channel] = utils.log_softmax(logits.float(), dim=-1)
            else:
                logits_dict[channel] = utils.softmax(logits.float(), dim=-1)
        
        return logits_dict

    @classmethod
    def build_model(cls, args: SpeechDLMForASRConfig, task: SpeechDLMForASRTask):
        """Build a new model instance."""

        if args.build_from_dgslm:
            # Load the weights of a generative SpeechDLM model, but replace the output layer
            assert args.dgslm_path is not None, "Must specify --dgslm-path if --build-from-dgslm is set"
            assert args.dgslm_checkpoint_file is not None, "Must specify --dgslm-checkpoint-file if --build-from-dgslm is set"

            encoder=cls.from_pretrained_dgslm(
                task=task,
                args=args,
                model_name_or_path=args.dgslm_path,
                checkpoint_file=args.dgslm_checkpoint_file,
                data_name_or_path=args.dgslm_path,
            )
        else:
            encoder=cls.build_encoder(args, task)

        if args.freeze_input_embedding:
            encoder.embed_tokens.weight.requires_grad = False
        
        if args.freeze_transformer_layers:
            layers_to_freeze = set(map(int, args.freeze_transformer_layers.split(",")))
            for layer in layers_to_freeze:
                for param in encoder.layers[layer].parameters():
                    param.requires_grad = False

        if args.freeze_input_embedding or args.freeze_transformer_layers:
            logger.info("Layers frozen: {}".format([n for n, p in encoder.named_parameters() if not p.requires_grad]))
            logger.info("Layers requiring grad: {}".format([n for n, p in encoder.named_parameters() if p.requires_grad]))

        return cls(encoder, args)
    
    @classmethod
    def build_encoder(cls, args, task):
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
            
        if args.decoder_cross_layers < 0:
            args.decoder_cross_layers = args.decoder_layers

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        # Build the unit embeddings
        input_embedding = cls.build_embedding(
            args, task.input_dict, args.decoder_input_dim
        )

        return CrossChannelTransformerEncoderForASR(
            args,
            task.input_dict,
            task.output_dict,
            input_embedding,
            channels=task.channels,
            no_encoder_attn=True # what is this?
        )


    @classmethod
    def build_embedding(cls, args, src_dict, embed_dim, path=None):
        # Input unit embeddings
        input_embedding = Embedding(len(src_dict), embed_dim, src_dict.pad())
        return input_embedding

    @classmethod
    def args_from_dgslm(cls, args, dgslm_args, copy_args=False):
        """Load the model arguments from a generative SpeechDLM model."""
        from dataclasses import asdict
        from omegaconf import OmegaConf
        from fairseq.dataclass.configs import FairseqConfig, CommonConfig
        from fairseq.criterions.speech_dlm_for_asr_criterion import SpeechDLMForASRCriterionConfig
        from fairseq.tasks.speech_dlm_for_asr_task import SpeechDLMForASRTaskConfig
        import copy

        dgslm_model_cfg = copy.deepcopy(dgslm_args)

        # Model args:
        model_cfg = OmegaConf.structured(SpeechDLMForASRConfig())
        dgslm_args.model = model_cfg

        for field in model_cfg.keys():
            if field in dgslm_model_cfg:
                setattr(dgslm_args.model, field, dgslm_model_cfg[field])
        
        # Task args:
        # Task is setup prior to building the, model, so they do not need to be copied
        
        # Criterion args:
        criterion_cfg =  OmegaConf.structured(SpeechDLMForASRCriterionConfig()) # TODO switch to SpeechDLMForASRCriterionConfig
        for field in criterion_cfg.keys():
            if field in dgslm_args.criterion:
                setattr(criterion_cfg, field, dgslm_args.criterion[field])
        dgslm_args.criterion = criterion_cfg

        return dgslm_args
    
    @classmethod
    def from_pretrained_dgslm(
        cls,
        task,
        args,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        """
        For loading the weights of the generative SpeechDLM decoder into an encoder for ASR, but replacing
        the output embeddings and linear output layer for the ASR task.
        """
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            load_checkpoint_heads=True,
            **kwargs,
        )
        dgslm_decoder = x["models"][0].decoder
        asr_config = cls.args_from_dgslm(args, x["args"])
        
        print(args.keys())
        # Build the ASR model, with the same architecture as the DGSLM model
        encoder = SpeechDLMForASR.build_encoder(asr_config.model, task)
        
        # Load the common weights from the DGSLM model
        missing_keys, unexpected_keys = encoder.load_state_dict(dgslm_decoder.state_dict(), strict=False) # Using strict=False to allow for missing parameters

        logger.info("Loaded ASR model from DGSLM model")
        logger.info("DGSLM weights not loaded to ASR model: {}".format(unexpected_keys))
        logger.info("ASR model weights initialized from scratch: {}".format(missing_keys))        

        return encoder
    

@register_model_architecture("speech_dlm_for_asr", "speech_dlm_for_asr_base")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_cross_layers = getattr(args, "decoder_cross_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    #args.share_decoder_input_output_embed = getattr(
    #    args, "share_decoder_input_output_embed", False
    #)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
