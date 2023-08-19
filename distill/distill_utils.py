import torch
import random
def MlmAdaptorWithLogitsMask(batch, model_outputs):
    labels = batch["labels"]
    logits_mask = torch.ge(labels, -1)
    return {'logits': (model_outputs["logits"]),
            'hidden': model_outputs["hidden_states"],
            "logits_mask": logits_mask}

def init_checkpoint(model, num_layers):
    prefix = "bert"
    state_dict = model.state_dict()
    compressed_sd = {}

    # Embeddings #
    for w in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
        param_name = f"{prefix}.embeddings.{w}.weight"
        compressed_sd[param_name] = state_dict[param_name]
    for w in ["weight", "bias"]:
        param_name = f"{prefix}.embeddings.LayerNorm.{w}"
        compressed_sd[param_name] = state_dict[param_name]

    # Transformer Blocks #
    std_idx = 0
    layers = sorted(random.sample(range(12),num_layers))
    for teacher_idx in layers:
        for layer in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "attention.output.LayerNorm",
            "intermediate.dense",
            "output.dense",
            "output.LayerNorm",
        ]:
            for w in ["weight", "bias"]:
                compressed_sd[f"{prefix}.encoder.layer.{std_idx}.{layer}.{w}"] = state_dict[
                    f"{prefix}.encoder.layer.{teacher_idx}.{layer}.{w}"
                ]
        std_idx += 1
    
    return compressed_sd

def divide_parameters(named_parameters, lr=None):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_parameters_names = list(zip(
        *[(p, n) for n, p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(
        zip(*[(p, n) for n, p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names) > 0:
        decay_parameters, decay_names = decay_parameters_names
        # print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params': decay_parameters,
                           'weight_decay': 0.01, 'lr': lr}
        else:
            decay_group = {'params': decay_parameters,
                           'weight_decay': 0.01}
        param_group.append(decay_group)

    if len(no_decay_parameters_names) > 0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        #print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters,
                              'weight_decay': 0.0, 'lr': lr}
        else:
            no_decay_group = {
                'params': no_decay_parameters, 'weight_decay': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group) > 0
    return param_group


H_312=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":3, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":6, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":9, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]},
                {"layer_T":12,"layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",312,768]}]


H_256=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":4, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":6, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":8, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":10, "layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]},
                {"layer_T":12, "layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",256,768]}]

H_768=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":2, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":4, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":6, "layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":8, "layer_S":4, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":10, "layer_S":5, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]},
                {"layer_T":12, "layer_S":6, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",768,768]}]

matches = {
    "h312":H_312,
    "h256":H_256,
    "h768":H_768
}