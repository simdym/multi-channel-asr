import math
import editdistance
from dataclasses import dataclass, field
from typing import Optional

import torch.nn.functional as F
import torch
from fairseq import metrics, utils, meters
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from omegaconf import II

from fairseq.tasks.speech_dlm_for_asr_task import SpeechDLMForASRTask



@dataclass
class SpeechDLMForASRCriterionConfig(FairseqDataclass):
    zero_infinity: Optional[bool] = field(
        default=False, metadata={"help": "zero infinities in loss"}
    )


@register_criterion("speech_dlm_for_asr_criterion", dataclass=SpeechDLMForASRCriterionConfig)
class SpeechDLMForASRCriterion(FairseqCriterion):
    """Criteron for usig the SpeechDLM model as described in the paper:
    https://arxiv.org/pdf/2203.16502.pdf for automatic speech recognition.
    
    CTC loss is used for the ASR task.
    """

    def __init__(
            self,
            cfg: SpeechDLMForASRCriterionConfig,
            task: SpeechDLMForASRTask
        ):
        super().__init__(task)
        self.blank_idx = self.task.output_dict.index("_")

        self.channels = task.channels
        self.post_process = "letter"

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample["net_input"]
        net_target = sample["net_target"]

        """print("ntokens", sample["ntokens"])
        print("nsentences", sample["nsentences"])
        print("src_len", net_input["src_lengths"].size())
        print("src_tokens", net_input["src_tokens"]["0"].size())
        print("tgt_len", net_target["tgt_lengths"]["0"].size())
        print("tgt_tokens", net_target["tgt_tokens"]["0"].size())
        print("src_len_nans", torch.isnan(net_input["src_lengths"]).sum())
        print("src_nans", torch.isnan(net_input["src_tokens"]["0"]).sum())
        print("tgt_len_nans", torch.isnan(net_target["tgt_lengths"]["0"]).sum())
        print("tgt_nans", torch.isnan(net_target["tgt_tokens"]["0"]).sum())"""

        #print("src_len / tgt_len", net_input["src_tokens"]["0"].size(-1) / net_target["tgt_tokens"]["0"].size(-1))

        if torch.isnan(net_input["src_lengths"]).sum():
            print("src_len_nans", torch.isnan(net_input["src_lengths"]).sum())

        if torch.isnan(net_input["src_tokens"]["0"]).sum():
            print("src_nans", torch.isnan(net_input["src_tokens"]["0"]).sum())
        
        if torch.isnan(net_target["tgt_lengths"]["0"]).sum():
            print("tgt_len_nans", torch.isnan(net_target["tgt_lengths"]["0"]).sum())
        
        if torch.isnan(net_target["tgt_tokens"]["0"]).sum():
            print("tgt_nans", torch.isnan(net_target["tgt_tokens"]["0"]).sum())

        # Forward pass
        output, _ = model(**net_input)

        sample_size = sample["ntokens"]

        for key, value in output.items():
            output[key] = value.transpose(0, 1)

        # Calculate log probabilities
        lprobs = model.get_normalized_probs(
            output, log_probs=True
        )

        with torch.backends.cudnn.flags(enabled=False):
            loss = self.compute_loss(
                lprobs, net_target["tgt_tokens"], net_input["src_lengths"], net_target["tgt_lengths"]
            )# / sample_size

        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample_size,
            "nsentences": sample["nsentences"]
        }

        if not model.training:
            c_err_tot = 0
            c_len_tot = 0
            w_errs_tot = 0
            w_len_tot = 0

            for channel in self.channels:
                c_err, c_len, w_errs, w_len = self.get_wer(
                    lprobs[channel], net_target["tgt_tokens"][channel], net_input["src_lengths"]
                )
                c_err_tot += c_err
                c_len_tot += c_len
                w_errs_tot += w_errs
                w_len_tot += w_len            
            
            logging_output["c_errors"] = c_err_tot
            logging_output["c_len"] = c_len_tot
            logging_output["w_errors"] = w_errs_tot
            logging_output["w_len"] = w_len_tot

        return (loss, sample_size, logging_output) #TODO: Add sample size and logging outputs
    
    def collapse_ctc_sequence(self, sequence):
        """Collapse the CTC sequences by removing consecutive duplicates and the blank token.
        
        Args:
            lprobs (torch.Tensor): Log-probabilities from the model of shape (TxC)."""
        with torch.no_grad():
            non_consec_sequence = sequence.unique_consecutive()
            return non_consec_sequence[non_consec_sequence != self.blank_idx]
    
    def get_most_probable_sequence(self, lprobs, beam=1):
        """Get the most probable sequence from the log-probabilities.
        
        Args:
            lprobs (torch.Tensor): Log-probabilities from the model of shape (TxC).
            beam (int): Beam size for beam search. Default is 1."""
        with torch.no_grad():
            if beam == 1:
                max_prob_tokens = lprobs.argmax(dim=-1)
                return self.collapse_ctc_sequence(max_prob_tokens)
            else:
                raise NotImplementedError("Beam search not implemented yet.")
    
    def get_wer(self, lprobs, target, input_lengths):
        """Compute the Word Error Rate (WER) and Character Error Rate (CER) for given log-probabilities and targets.
        
        Args:
            lprobs (torch.Tensor): Dict with log-probabilities from the model of shape (TxBxC).
            target (torch.LongTensor): Dict with target tokens of shape (TxB).
            input_lengths (torch.LongTensor): Length of each input of shape (B)
        
        Returns:
            Tuple[int, int, int, int]: Number of character errors, number of characters, number of word errors, number of words."""
        with torch.no_grad():
            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0

            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
            for lprob, tgt, inp_len in zip(lprobs_t, target, input_lengths):
                lprob = lprob[:inp_len] # Remove padding
                pred = self.get_most_probable_sequence(lprob)
                pred_chars = self.task.output_dict.string(pred, separator="")

                indicies = (
                    (tgt != self.task.output_dict.eos())
                    & (tgt != self.task.output_dict.pad())
                )
                target = tgt[indicies]

                target_chars = self.task.output_dict.string(target, separator="")

                c_err += editdistance.eval(target_chars, pred_chars)
                c_len += len(target_chars)

                target_words = post_process(target_chars, self.post_process).split()
                pred_words = post_process(pred_chars, self.post_process).split()

                w_errs += editdistance.eval(target_words, pred_words)
                w_len += len(target_words)
        
        return c_err, c_len, w_errs, w_len


    def compute_loss(self, lprobs, tgt_tokens, src_lenghts, tgt_lengths):
        """
        Compute the loss for the given sample.

        Args:
            lprobs (Dict[str, Tensor]): Dict with log-probabilities from the model.
            tgt_tokens (Dict[str, Tensor]): Dict with target tokens for computing the loss.
            src_lenghts (Tensor): Length of each input. Length is the same for both channels.
            tgt_lengths (Dict[str, Tensor]): Length of each target. Length can be different for each channel."""
        
        ctc_loss_sum = 0.0
        for channel in self.channels:
            ctc_loss_sum += F.ctc_loss(
                log_probs=lprobs[channel],
                targets=tgt_tokens[channel],
                input_lengths=src_lenghts, # Source lengths are the same for both channels
                target_lengths=tgt_lengths[channel], # Target lengths can be different for each channel
                blank=self.blank_idx,
                reduction="sum"
            )
        return ctc_loss_sum

    @staticmethod
    def reduce_metrics(logging_outputs):
        metrics.log_scalar(
            "loss", sum(log["loss"] for log in logging_outputs) / sum(log["ntokens"] for log in logging_outputs)
        )
        metrics.log_scalar(
            "c_errors", sum(log["c_errors"] if "c_errors" in log else float("nan") for log in logging_outputs), round=1
        )
        metrics.log_scalar(
            "c_len", sum(log["c_len"] if "c_len" in log else float("nan") for log in logging_outputs), round=1
        )
        metrics.log_scalar(
            "w_errors", sum(log["w_errors"] if "w_errors" in log else float("nan") for log in logging_outputs), round=1
        )
        metrics.log_scalar(
            "w_len", sum(log["w_len"] if "w_len" in log else float("nan") for log in logging_outputs), round=1
        )

        metrics.log_derived(
            "cer",
            lambda meter: meters.safe_round(meter["c_errors"].sum * 100 / meter["c_len"].sum, ndigits=2)
            if meter["c_len"].sum > 0
            else float("nan")
        )
        metrics.log_derived(
            "wer",
            lambda meter: meters.safe_round(meter["w_errors"].sum * 100 / meter["w_len"].sum, ndigits=2)
            if meter["w_len"].sum > 0
            else float("nan"),
        )

        
               
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import torch
    from fairseq.dataclass.configs import FairseqConfig, CommonConfig
    from fairseq.tasks.speech_dlm_for_asr_task import SpeechDLMForASRTaskConfig
    from fairseq.models.speech_dlm.speech_dlm_for_asr import SpeechDLMForASR, SpeechDLMForASRConfig

    torch.manual_seed(0)


    cfg = OmegaConf.structured(
        FairseqConfig()
    )

    common_cfg = OmegaConf.create(
        CommonConfig()
    )

    criterion_cfg = OmegaConf.create(
        SpeechDLMForASRCriterionConfig()
    )

    model_cfg = OmegaConf.create(
        SpeechDLMForASRConfig()
    )
    
    model_cfg.build_from_dgslm = True
    model_cfg.dgslm_path = "model_files"
    model_cfg.dgslm_checkpoint_file = "speech_dlm_base.pt"
    model_cfg._name = "model" #"speech_dlm_for_asr"

    task_cfg = OmegaConf.create(
        SpeechDLMForASRTaskConfig()
    )
    task_cfg.max_target_positions = 1024
    task_cfg.input_dict_path = "model_files/dict.unitA.txt"
    task_cfg.output_dict_path = "model_files/dict.ltr.txt"

    cfg.common = common_cfg
    cfg.criterion = criterion_cfg
    cfg.model = model_cfg
    cfg.task = task_cfg

    task = SpeechDLMForASRTask.setup_task(cfg.task)
    model = SpeechDLMForASR.build_model(cfg.model, task)
    criterion = SpeechDLMForASRCriterion(cfg.criterion, task)

    """batch_size = 10
    input_length = 20 # constant input length for simplicity
    target_lengths = {"0": 15, "1": 18} # constant target length for simplicity
    sample = {
        "id": torch.arange(batch_size),
        "source": {
            "0": torch.randint(4, 504, (batch_size, input_length)),
            "1": torch.randint(4, 504, (batch_size, input_length))
        }, # just for testing
        "target": {
            "0": torch.randint(1, 32, (batch_size, target_lengths["0"])),
            "1": torch.randint(1, 32, (batch_size, target_lengths["1"]))
        }, # Targets cannot be blank thus range is [1, 32)
        "source_lengths": torch.tensor(batch_size*[input_length]),
        "target_lengths": {
            "0": torch.tensor(batch_size*[target_lengths["0"]]),
            "1": torch.tensor(batch_size*[target_lengths["1"]])
        }
    }
    lprobs = {
        "0": torch.rand(input_length, batch_size, 28),
        "1": torch.rand(input_length, batch_size, 28)
    }"""
    import re
    SPACE_NORMALIZER = re.compile(r"\s+")

    def tokenize_characters(characters):
        characters = SPACE_NORMALIZER.sub(" ", characters)
        characters = characters.strip()
        return list(characters)

    target = "hello|world"
    target_encoded = torch.Tensor(
        task.output_dict.encode_line(
            target,
            line_tokenizer=tokenize_characters,
            add_if_not_exist=False
        )
    ).unsqueeze(0)

    pred = "hel_lllo|world"
    pred_encoded = torch.Tensor(
        task.output_dict.encode_line(
            pred,
            line_tokenizer=tokenize_characters,
            add_if_not_exist=False
        )
    ).unsqueeze(0)
    pred_onehot = F.one_hot(pred_encoded.long(), num_classes=len(task.output_dict)).float()

    pred_len = torch.LongTensor([len(pred)])

    print("pred_onehot", pred_onehot.size())
    print("target_encoded", target_encoded.size())
    print("pred_len", pred_len)

    wer = criterion.get_wer(torch.transpose(pred_onehot, 0, 1), target_encoded, pred_len)

    print("WER", wer)
    print("CER", wer[0]/wer[1] * 100, "%")
    print("WER", wer[2]/wer[3] * 100, "%")
