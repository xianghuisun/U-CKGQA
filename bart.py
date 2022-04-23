import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class MyBart(BartForConditionalGeneration):
    def forward(self, 
                input_ids, 
                attention_mask=None, 
                encoder_outputs=None,
                decoder_input_ids=None, 
                decoder_attention_mask=None, 
                #decoder_cached_states=None,
                use_cache=False, 
                is_training=False,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            #decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            return_dict=return_dict
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return loss
        #return (lm_logits, ) + outputs[1:]
        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate_from_string(self, _input, tokenizer=None, **generator_args):
        assert tokenizer is not None
        if isinstance(_input, str):
            _input = [[0] + tokenizer.encode(_input)]
        if isinstance(_input, list) and isinstance(_input[0], str):
            _input = [[0] + tokenizer.encode(i) for i in _input]
        if isinstance(_input, list):
            if isinstance(_input[0], int):
                _input = [_input]
            _input = torch.LongTensor(_input)
        res = self.generate(_input, **generator_args)
        return ([tokenizer.decode(x, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip() for x in res])

