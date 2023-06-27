from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset10 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys, os
import argparse

def show_io(m):
    print("Inputs of the model:")
    for port, _input in enumerate(m.inputs):
        print("	[{}] {}".format(port, _input))
    print("Outputs of the model:")
    for port, _output in enumerate(m.outputs):
        print("	[{}] {}".format(port, _output))

LAYER_NUM = 28 #28 
HEAD_NUM = 32 #32
SIZE_PER_HEAD = 128 #128
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD #4096
INTERMEDIATE_SIZE = 16384 # ? should be inner_hidden_size
LAYER_NORM_EPS = 1e-5
MAX_POSITION_EMBEDDINGS = 1024 #2048
ROTARY_EMB_BASE = 10000
ROTARY_PCT = 1 # ?
USE_PARALLEL_RESIDUAL = True
VOCAB_SIZE = 130528
MAX_SEQ_LEN = 1024
FakeConstDict = {
    'transformer.word_embeddings.weight': np.zeros((VOCAB_SIZE, HIDDEN_SIZE), dtype=np.float32),
    'lm_head.weight': np.zeros((HIDDEN_SIZE, VOCAB_SIZE), dtype=np.float32),
    'transformer.final_layernorm.bias': np.zeros((HIDDEN_SIZE,), dtype=np.float32),
    'transformer.final_layernorm.weight': np.zeros((HIDDEN_SIZE,), dtype=np.float32),
    'transformer.layers.0.input_layernorm.bias': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.input_layernorm.weight': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.post_attention_layernorm.bias': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.post_attention_layernorm.weight': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.attention.query_key_value.weight': [
        np.zeros((HIDDEN_SIZE, HIDDEN_SIZE * 3), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.attention.dense.bias': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.attention.dense.weight': [
        np.zeros((HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.mlp.dense_h_to_4h.bias': [
        np.zeros((INTERMEDIATE_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.mlp.dense_h_to_4h.weight': [
        np.zeros((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.mlp.dense_4h_to_h.bias': [
        np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ] * LAYER_NUM,
    'transformer.layers.0.mlp.dense_4h_to_h.weight': [
        np.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=np.float32)
    ] * LAYER_NUM,
}
def layer(hidden_states, past_keys_num, beam_idx, attention_mask, position_ids, layer_idx, ConstDict, quant_dicts):
    def make_fc(name, data, weights):
        if name + '/fq_input_0' in quant_dicts:
            info_data = quant_dicts[name + '/fq_input_0'][0]
            info_weight = quant_dicts[name + '/fq_weights_1'][0]
            q_d = opset.fake_quantize(data, info_data['il'], info_data['ih'], info_data['ol'], info_data['oh'], info_data['levels'], info_data['auto_broadcast'], name=f'{name}/fq_input_0')
            q_w = opset.fake_quantize(weights, info_weight['il'], info_weight['ih'], info_weight['ol'], info_weight['oh'], info_weight['levels'], info_weight['auto_broadcast'], name=f'{name}/fq_weights_1')
            node = opset.matmul(q_d, q_w, transpose_a=False, transpose_b=True, name=name) #wildcard [?,?,7680]
        else:
            node = opset.matmul(data, weights, transpose_a=False, transpose_b=True, name=name) #wildcard [?,?,7680]
        return node
    def get_scale(prefix):
        def scale(info):
            return (info['levels'] - 1) / (info['ih'] - info['il'])
        if prefix + 'matmul1/fq_input_0' in quant_dicts:
            info_data0 = quant_dicts[prefix + 'matmul1/fq_input_0'][0]
            info_data1 = quant_dicts[prefix + 'matmul1/fq_input_1'][0]
            info_data2 = quant_dicts[prefix + 'matmul2/fq_input_0'][0]
            info_data3 = quant_dicts[prefix + 'matmul2/fq_input_1'][0]
            return scale(info_data0), scale(info_data1), scale(info_data2), scale(info_data3)
        else:
            return 0.0, 0.0, 0.0, 0.0
    input_layernorm_bias = opset.constant(ConstDict['transformer.layers.input_layernorm.bias'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.input_layernorm.bias')
    input_layernorm_weight = opset.constant(ConstDict['transformer.layers.input_layernorm.weight'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.input_layernorm.weight')
    # layerNorm operation
    input_layernorm_mvn = opset.mvn(hidden_states, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name=f'/transformer/layers.{layer_idx}/input_layernorm/mvn')
    input_layernorm_mul = opset.multiply(input_layernorm_mvn, input_layernorm_weight, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/input_layernorm/mul')
    input_layernorm = opset.add(input_layernorm_mul, input_layernorm_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/input_layernorm/add')

    ######### attention part begin
    query_key_value_bias = opset.constant(ConstDict['transformer.layers.attention.query_key_value.bias'][layer_idx], Type.f32, name=f'/transformer.layers.{layer_idx}.attention.query_key_value.bias')
    query_key_value_weights = opset.constant(ConstDict['transformer.layers.attention.query_key_value.weight'][layer_idx], Type.f32, name=f'/transformer.layers.{layer_idx}.attention.query_key_value.weight')
    name = f'/transformer/layers.{layer_idx}/attention/query_key_value/MatMul'
    qkv_ = make_fc(name, input_layernorm, query_key_value_weights)
    qkv = opset.add(qkv_, query_key_value_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/attention/query_key_value/Add') #wildcard [?,?,7680]

    # custom op
    q_quant, k_quant, qk_quant, v_quant = get_scale(f'/transformer/layers.{layer_idx}/attention/attn/')
    attn_output = opset.gpt_attn(qkv, past_keys_num, beam_idx, attention_mask, position_ids,
            layer_num=LAYER_NUM, head_num=HEAD_NUM, size_per_head=SIZE_PER_HEAD,
            rotary_emb_base=ROTARY_EMB_BASE, rotary_pct=ROTARY_PCT, cur_layer_num=layer_idx, max_seq_len=MAX_SEQ_LEN,use_position2d=True,
            q_quant=q_quant, k_quant=k_quant, qk_quant=qk_quant, v_quant=v_quant,
            name=f'/transformer/layers.{layer_idx}/attention/attn') #[b,seq,hidden_size]

    # attn_output = self.dense(attn_output) line: 157
    dense_weight = opset.constant(ConstDict['transformer.layers.attention.dense.weight'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.attention.dense.weight')
    name = f'/transformer/layers.{layer_idx}/attention/dense/MatMul'
    dense_ = make_fc(name, attn_output, dense_weight)
    dense_bias = opset.constant(ConstDict['transformer.layers.attention.dense.bias'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.attention.dense.bias')
    dense = opset.add(dense_, dense_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/attention/dense/Add') #wildcard [?,?,HIDDEN_SIZE]
    attn_output = dense
    ######### attention part end
    
    # Residual connection.
    alpha = np.array([(2 * LAYER_NUM) ** 0.5], dtype="float32")
    alpha_qkv = opset.multiply(input_layernorm, alpha, auto_broadcast="numpy",name=f'/transformer/layers.{layer_idx}/res1/Mul')
    hidden_states = opset.add(alpha_qkv, attn_output, auto_broadcast='numpy',name=f'/transformer/layers.{layer_idx}/res1/Add')

    assert(USE_PARALLEL_RESIDUAL == True)
    post_attention_layernorm_bias = opset.constant(ConstDict['transformer.layers.post_attention_layernorm.bias'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.post_attention_layernorm.bias')
    post_attention_layernorm_weight = opset.constant(ConstDict['transformer.layers.post_attention_layernorm.weight'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.post_attention_layernorm.weight')
    post_attention_layernorm_mvn = opset.mvn(hidden_states, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name=f'/transformer/layers.{layer_idx}/post_attention_layernorm/mvn')
    post_attention_layernorm_mul = opset.multiply(post_attention_layernorm_mvn, post_attention_layernorm_weight, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/post_attention_layernorm/mul')
    post_attention_layernorm = opset.add(post_attention_layernorm_mul, post_attention_layernorm_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/post_attention_layernorm/add')

    # mlp
    def mlp(states):
        dense_h_to_4h_bias = opset.constant(ConstDict['transformer.layers.mlp.dense_h_to_4h.bias'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.mlp.dense_h_to_4h.bias')
        dense_h_to_4h_weight = opset.constant(ConstDict['transformer.layers.mlp.dense_h_to_4h.weight'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.mlp.dense_h_to_4h.weight')
        name = f'/transformer/layers.{layer_idx}/mlp/dense_h_to_4h/MatMul'
        dense_h_to_4h_ = make_fc(name, states, dense_h_to_4h_weight)
        dense_h_to_4h = opset.add(dense_h_to_4h_, dense_h_to_4h_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/mlp/dense_h_to_4h/Add') #wildcard [?,?,INTERMEDIATE_SIZE]
        gelu = opset.gelu(dense_h_to_4h, approximation_mode='erf', name=f'/transformer/layers.{layer_idx}/mlp/dense_h_to_4h/Gelu')
        dense_4h_to_h_bias = opset.constant(ConstDict['transformer.layers.mlp.dense_4h_to_h.bias'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.mlp.dense_4h_to_h.bias')
        dense_4h_to_h_weight = opset.constant(ConstDict['transformer.layers.mlp.dense_4h_to_h.weight'][layer_idx], Type.f32, name=f'transformer.layers.{layer_idx}.mlp.dense_4h_to_h.weight')
        name = f'/transformer/layers.{layer_idx}/mlp/dense_4h_to_h/MatMul'
        dense_4h_to_h_ = make_fc(name, gelu, dense_4h_to_h_weight)
        dense_4h_to_h = opset.add(dense_4h_to_h_, dense_4h_to_h_bias, auto_broadcast='numpy', name=f'/transformer/layers.{layer_idx}/mlp/dense_4h_to_h/Add') #wildcard [?,?,HIDDEN_SIZE]
        return dense_4h_to_h

    mlp_output = mlp(post_attention_layernorm)
    # Second residual connection.
    output = opset.add(
        opset.multiply(post_attention_layernorm, alpha, auto_broadcast="numpy", name=f"/transformer/layers.{layer_idx}/post_atten_layernorm/res2/mul"),
        mlp_output, auto_broadcast='numpy', name=f"/transformer/layers.{layer_idx}/post_atten_layernorm/res2/add")
    return output

def create_model(ConstDict, quant_dicts):
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    past_keys_num = opset.parameter([1,], Type.i32, name='past_keys_num')
    beam_idx = opset.parameter([-1,], Type.i32, name='beam_idx')
    attention_mask = opset.parameter([-1, 1, -1, -1], Type.f32, name='attention_mask')
    position_ids = opset.parameter([-1, 2, -1], Type.i32, name='position_ids')

    embed_in_const = opset.constant(ConstDict['transformer.word_embeddings.weight'], Type.f32)
    inputs_embeds = opset.gather(embed_in_const, indices=input_ids, axis=0) # name='/transformer/word_embeddings/Gather') # [?,?,HIDDEN_SIZE]
    hidden_states = inputs_embeds

    for i in range(LAYER_NUM):
        hidden_states = layer(hidden_states, past_keys_num, beam_idx, attention_mask, position_ids, i, ConstDict, quant_dicts)
    # final_layernorm
    final_layernorm_bias = opset.constant(ConstDict['transformer.final_layernorm.bias'], Type.f32)
    final_layernorm_weight = opset.constant(ConstDict['transformer.final_layernorm.weight'], Type.f32)
    final_layer_norm_mvn = opset.mvn(hidden_states, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='/transformer/final_layernorm/mvn')
    final_layer_norm_mul = opset.multiply(final_layer_norm_mvn, final_layernorm_weight, auto_broadcast='numpy', name='/transformer/final_layernorm/mul')
    final_layernorm = opset.add(final_layer_norm_mul, final_layernorm_bias, auto_broadcast='numpy', name='/transformer/final_layernorm/add')
    # embed_out
    embed_out_weight = opset.constant(ConstDict['lm_head.weight'], Type.f32)
    name = 'logits'
    if name + '/fq_input_0' in quant_dicts:
        info_data = quant_dicts[name + '/fq_input_0'][0]
        info_weight = quant_dicts[name + '/fq_weights_1'][0]
        q_d = opset.fake_quantize(final_layernorm, info_data['il'], info_data['ih'], info_data['ol'], info_data['oh'], info_data['levels'], info_data['auto_broadcast'], name=f'{name}/fq_input_0')
        q_w = opset.fake_quantize(embed_out_weight, info_weight['il'], info_weight['ih'], info_weight['ol'], info_weight['oh'], info_weight['levels'], info_weight['auto_broadcast'], name=f'{name}/fq_weights_1')
        embed_out = opset.matmul(q_d, q_w, transpose_a=False,transpose_b=True, name=name) #wildcard [?,?,VOCAB_SIZE]
    else:
        embed_out = opset.matmul(final_layernorm, embed_out_weight, transpose_a=False,transpose_b=True, name=name) #wildcard [?,?,VOCAB_SIZE]
    embed_out_result = opset.result(embed_out, name='logits/sink_port_0') # [?,?,VOCAB_SIZE]
    return Model([embed_out_result], [input_ids, past_keys_num, beam_idx, attention_mask, position_ids])

def get_params_from_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    import torch
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to('cpu').eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    global LAYER_NUM, HEAD_NUM, SIZE_PER_HEAD, HIDDEN_SIZE, INTERMEDIATE_SIZE, LAYER_NORM_EPS, MAX_POSITION_EMBEDDINGS, ROTARY_EMB_BASE, ROTARY_PCT, USE_PARALLEL_RESIDUAL, VOCAB_SIZE, MAX_SEQ_LEN
    LAYER_NUM = model.config.num_layers
    HEAD_NUM = model.config.num_attention_heads
    SIZE_PER_HEAD = model.config.hidden_size // model.config.num_attention_heads
    HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
    INTERMEDIATE_SIZE = model.config.inner_hidden_size #intermediate_size
    LAYER_NORM_EPS = model.config.layernorm_epsilon
    MAX_POSITION_EMBEDDINGS = model.config.max_sequence_length #max_position_embeddings
    MAX_SEQ_LEN = model.config.max_sequence_length
    ROTARY_EMB_BASE = 10000 #model.config.rotary_emb_base
    ROTARY_PCT = 0.5 #model.config.rotary_pct
    USE_PARALLEL_RESIDUAL = True #model.config.use_parallel_residual
    VOCAB_SIZE = model.config.vocab_size
    ConstDict = {
        'transformer.word_embeddings.weight': model.transformer.word_embeddings.weight.detach().numpy(),
        'lm_head.weight': model.lm_head.weight.detach().numpy(),
        'transformer.final_layernorm.bias': model.transformer.final_layernorm.bias.detach().numpy(),
        'transformer.final_layernorm.weight': model.transformer.final_layernorm.weight.detach().numpy(),
        'transformer.layers.input_layernorm.bias': [
            l.input_layernorm.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.input_layernorm.weight': [
            l.input_layernorm.weight.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.post_attention_layernorm.bias': [
            l.post_attention_layernorm.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.post_attention_layernorm.weight': [
            l.post_attention_layernorm.weight.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.attention.query_key_value.bias': [
            l.attention.query_key_value.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.attention.query_key_value.weight': [
            l.attention.query_key_value.weight.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.attention.dense.bias': [
            l.attention.dense.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.attention.dense.weight': [
            l.attention.dense.weight.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.mlp.dense_h_to_4h.bias': [
            l.mlp.dense_h_to_4h.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.mlp.dense_h_to_4h.weight': [
            l.mlp.dense_h_to_4h.weight.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.mlp.dense_4h_to_h.bias': [
            l.mlp.dense_4h_to_h.bias.detach().numpy() for l in model.transformer.layers
        ],
        'transformer.layers.mlp.dense_4h_to_h.weight': [
            l.mlp.dense_4h_to_h.weight.detach().numpy() for l in model.transformer.layers
        ],
    }
    return ConstDict

def get_quant_params_from_model(path):
    org_dicts = {}
    core = Core()
    net = core.read_model(model=path)
    for op in net.get_ordered_ops():
        if op.get_type_name() == 'FakeQuantize':
            print(op.friendly_name, op.name, op.get_attributes())
            org_dicts[op.friendly_name] = {}
            # parants
            parents_name = ['il', 'ih', 'ol', 'oh']
            org_dicts[op.friendly_name]['levels'] = op.get_attributes()['levels']
            org_dicts[op.friendly_name]['auto_broadcast'] = op.get_attributes()['auto_broadcast']
            for i in range(1, 5):
                const_node = op.input_value(i).get_node()
                print(f'{parents_name[i - 1]} name:{const_node.name}, friend name:{const_node.friendly_name}, shape:{const_node.shape} data:{const_node.get_data()}')
                org_dicts[op.friendly_name][parents_name[i - 1]] = const_node.get_data()
    dicts = {}
    maps = {
        'attention/query_key_value/MatMul/fq_input_0': '/transformer/layers.{}/attention/query_key_value/MatMul/fq_input_0',
        'attention/query_key_value/MatMul/fq_weights_1': '/transformer/layers.{}/attention/query_key_value/MatMul/fq_weights_1',
        'attention/Transpose_4/fq_input_0': '/transformer/layers.{}/attention/dense/MatMul/fq_input_0',
        'attention/dense/MatMul/fq_weights_1': '/transformer/layers.{}/attention/dense/MatMul/fq_weights_1',
        'mlp/dense_h_to_4h/MatMul/fq_input_0': '/transformer/layers.{}/mlp/dense_h_to_4h/MatMul/fq_input_0',
        'mlp/dense_h_to_4h/MatMul/fq_weights_1': '/transformer/layers.{}/mlp/dense_h_to_4h/MatMul/fq_weights_1',
        'mlp/dense_4h_to_h/MatMul/fq_input_0': '/transformer/layers.{}/mlp/dense_4h_to_h/MatMul/fq_input_0',
        'mlp/dense_4h_to_h/MatMul/fq_weights_1': '/transformer/layers.{}/mlp/dense_4h_to_h/MatMul/fq_weights_1',
        
        'attention/Concat_3/fq_input_0': "/transformer/layers.{}/attention/attn/matmul1/fq_input_0",
        #'attention/Slice/fq_input_0': "/transformer/layers.{}/attention/attn/matmul1/fq_input_0",
        'attention/Mul_5/fq_input_1': "/transformer/layers.{}/attention/attn/matmul1/fq_input_1",
        'attention/MatMul_1/fq_input_0': '/transformer/layers.{}/attention/attn/matmul2/fq_input_0',
        'attention/Slice/fq_input_0': '/transformer/layers.{}/attention/attn/matmul2/fq_input_1',
        'start_logits/fq_input_0': 'logits/fq_input_0',
        'start_logits/fq_weights_1': 'logits/fq_weights_1'
    }
    for name, info in org_dicts.items():
        token = '/transformer/layers.'
        i = name.find(token)
        if i != -1:
            t = name[len(token):].split('/')
            idx = t[0]
            key = '/'.join(t[1:])
        else:
            idx = None
            key = name
        if key in maps:
            new_key = maps[key]
            if idx:
                new_key = new_key.format(idx)

            dicts[new_key] = (info, name)
    # union matmul1 fq parameter
    def combine_fqinfo(fq_info1, fq_info2):
        fq = fq_info1
        fq[0]['il'] = np.minimum(fq[0]['il'], fq_info2[0]['il'])
        fq[0]['ol'] = np.minimum(fq[0]['ol'], fq_info2[0]['ol'])
        fq[0]['ih'] = np.maximum(fq[0]['ih'], fq_info2[0]['ih'])
        fq[0]['oh'] = np.maximum(fq[0]['oh'], fq_info2[0]['oh'])
        return fq

    for i in range(LAYER_NUM):
        fq_info1 = dicts[f'/transformer/layers.{i}/attention/attn/matmul1/fq_input_0']
        fq_info2 = dicts[f'/transformer/layers.{i}/attention/attn/matmul2/fq_input_1']
        dicts[f'/transformer/layers.{i}/attention/attn/matmul1/fq_input_0'] = combine_fqinfo(fq_info1, fq_info2)
    return dicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("org_model_path")
    parser.add_argument("ov_model_path")
    parser.add_argument("quant_model_path", nargs='?', default="")
    args = parser.parse_args()

    quant_dicts = {}
    if len(args.quant_model_path):
        quant_dicts = get_quant_params_from_model(args.quant_model_path)
    dicts = get_params_from_model(args.org_model_path)
    model2 = create_model(dicts, quant_dicts)
    # model2 = create_model(FakeConstDict)
    print("====", "new model")
    show_io(model2)
    serialize(model2, args.ov_model_path)

