import torch
import yaml
from g2p_en import G2p
from src.phonemizer.seq2seq_model import Seq2Seq



def init_to_zero(m):
    if hasattr(m, "weight") and isinstance(m.weight, torch.nn.parameter.Parameter):
        m.weight.data.fill_(0.)
    if hasattr(m, "bias") and isinstance(m.bias, torch.nn.parameter.Parameter):
        m.bias.data.fill_(0.)
    if hasattr(m, "weight_ih_l0") and isinstance(m.weight_ih_l0, torch.nn.parameter.Parameter):
        m.weight_ih_l0.data.fill_(0.)
    if hasattr(m, "bias_ih_l0") and isinstance(m.bias_ih_l0, torch.nn.parameter.Parameter):
        m.bias_ih_l0.data.fill_(0.)
    if hasattr(m, "weight_hh_l0") and isinstance(m.weight_hh_l0, torch.nn.parameter.Parameter):
        m.weight_hh_l0.data.fill_(0.)
    if hasattr(m, "bias_hh_l0") and isinstance(m.bias_hh_l0, torch.nn.parameter.Parameter):
        m.bias_hh_l0.data.fill_(0.)

if __name__ == "__main__":
    orig_model = G2p()

    my_model = Seq2Seq()

    my_model.apply(init_to_zero)

    state_dict = my_model.state_dict()

    state_dict["enc_emb.weight"] = torch.tensor(orig_model.enc_emb)
    
    state_dict["enc_gru.weight_ih_l0"] = torch.tensor(orig_model.enc_w_ih)
    state_dict["enc_gru.weight_hh_l0"] = torch.tensor(orig_model.enc_w_hh)
    state_dict["enc_gru.bias_ih_l0"] = torch.tensor(orig_model.enc_b_ih)
    state_dict["enc_gru.bias_hh_l0"] = torch.tensor(orig_model.enc_b_hh)
    
    
    # decoder
    state_dict["dec_emb.weight"] = torch.tensor(orig_model.dec_emb)
    state_dict["dec_gru.weight_ih_l0"] = torch.tensor(orig_model.dec_w_ih)
    state_dict["dec_gru.weight_hh_l0"] = torch.tensor(orig_model.dec_w_hh)
    state_dict["dec_gru.bias_ih_l0"] = torch.tensor(orig_model.dec_b_ih)
    state_dict["dec_gru.bias_hh_l0"] = torch.tensor(orig_model.dec_b_hh)

    state_dict["fc.weight"] = torch.tensor(orig_model.fc_w)
    state_dict["fc.bias"] = torch.tensor(orig_model.fc_b)

    my_model.load_state_dict(state_dict)

    torch.save(state_dict, "output/phonemizer/seq2seq_state_dict.pt")
