import os
import pandas as pd
from transformers import PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
from transformers.tokenization_utils_base import PaddingStrategy
import torch
import time
import transformers

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from typing import Optional, Tuple
import numpy as np

from openvino.runtime import Core, Model, Tensor, PartialShape, serialize, AsyncInferQueue
from openvino.runtime.passes import Manager
from openvino.runtime.passes import VisualizeTree
import argparse
import openvino.runtime as ov

OV_MODEL_PATH = '../hacked/chatglm.xml'
PYTORCH_MODEL_PATH = 'chatglm_6b'
RESULT_PATH = 'ov-results-attn.txt'
MAX_SEQ_SIZE = 2048
CONTENT_PATH = 'content'

class CausalLMModelForOV(PreTrainedModel):
    beam_idx = None
    def __init__(
        self, model_path, config=None
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        PreTrainedModel.__init__(self, config)

        self.core = Core()
        #self.core.set_property({'CACHE_DIR': './cache'})
        print(f'version: {ov.get_version()}')
        beg = time.time()
        self.net = self.core.read_model(model=OV_MODEL_PATH)
        end = time.time()
        print(f'read model cost {end - beg:.2f} seconds')
        self.batch = -1
        self.net.reshape({'input_ids': [self.batch, [1, MAX_SEQ_SIZE]],
                          'beam_idx': [self.batch],
                          'attention_mask': [self.batch, 1, [1, MAX_SEQ_SIZE], [1, MAX_SEQ_SIZE]],
                          'position_ids': [self.batch, 2, [1, MAX_SEQ_SIZE]]
                          })
        config = {'PERFORMANCE_HINT': 'UNDEFINED',
            'PERF_COUNT': 'YES',
            'NUM_STREAMS': '1',
            'CPU_RUNTIME_CACHE_CAPACITY': '5000000',
            'AFFINITY': 'CORE',
            'ENFORCE_BF16': 'YES'
            }
        beg = time.time()
        self.exec_net300 = self.core.compile_model(self.net, 'CPU', config)
        end = time.time()

        print(f'compile model cost {end - beg:.2f} seconds')
        self.req = self.exec_net300.create_infer_request()
        self.idx = 0
        self.first_token = True
        self.stat = {
            'init': 0,
            'cost_first': 0,
            'cost_second+': 0,
            'post': 0,
            'times': 0
        }

    @classmethod
    def from_pretrained(cls, model_name_path: str):
        return cls(model_path=model_name_path)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        beg = time.time()

        if self.first_token:
            past_key_num = np.array([0,], dtype=np.int32)
            self.idx = 0
            self.seq_offset = input_ids.shape[1]
        else:
            past_key_num = np.array([self.seq_offset + self.idx,], dtype=np.int32)
            self.idx += 1
        input_ids_np = input_ids.to(dtype=torch.int32).cpu().numpy()
        if CausalLMModelForOV.beam_idx is None:
            beam_idx_np = np.arange(input_ids.shape[0], dtype=np.int32)
        else:
            beam_idx_np = CausalLMModelForOV.beam_idx.to(dtype=torch.int32).cpu().numpy()
        inputs = {
            0: Tensor(input_ids_np),
            1: Tensor(past_key_num),
            2: Tensor(beam_idx_np),
            3: Tensor(attention_mask.cpu().numpy()),
            4: Tensor(position_ids.contiguous().cpu().numpy())
        }
        self.stat['init'] += time.time() - beg
        beg = time.time()
        self.req.set_input_tensors(inputs)
        if self.first_token:
            self.req.infer()
            self.stat['cost_first'] += time.time() - beg
            beg = time.time()
            logits, = self.req.outputs
        else:
            self.req.infer()
            self.stat['cost_second+'] += time.time() - beg
            beg = time.time()
            logits, = self.req.outputs

        x = torch.from_numpy(logits.data)
        self.stat['post'] += time.time() - beg
        self.stat['times'] += 1
        self.first_token = False

        return CausalLMOutputWithPast(
            loss=None,
            logits=x,
            past_key_values='not none',
            hidden_states=None,
            attentions=None,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        CausalLMModelForOV.beam_idx = beam_idx
        return 'not none'

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ):
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None:
                right = torch.empty((*attention_mask.shape[:3], 1), dtype=torch.float32)
                right[:] = -10000.0
                attention_mask = torch.cat(
                    [attention_mask, right], dim=3)
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = 0
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, new_attention_mask], dim=2
                )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs):
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, -1:]
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        else:
            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }

def set_random_seed(seed):
    import random

    random.seed(seed)

    # pytorch RNGs
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np

    np.random.seed(seed)

f = None
def debug_log(s):
    global f
    if not f:
        f = open(RESULT_PATH, 'w')
    print(s)
    f.write(s + '\n')

sentences = [
    # "你好",
    # "介绍一下清华大学",
    # "世界羽毛球史上最伟大的球员都有谁？",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content."
]
parameters = [
            #   (False, 2048, 1),
            #   (False, 64, 1),
            #   (True, 2048, 2),
                (False, 2048, 1),
                (False, 2048, 1),
                (False, 2048, 1),
                (False, 2048, 1),
                ]

def test_ov():
    debug_log('test ov...')
    f_content = open(CONTENT_PATH, 'w')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)
    model = CausalLMModelForOV.from_pretrained(PYTORCH_MODEL_PATH)

    # workaround model.device check begin
    old_get_parameter_device = transformers.modeling_utils.get_parameter_device
    def my_get_parameter_device(parameter):
        if parameter == model:
            return torch.device(device)
        else:
            return old_get_parameter_device(parameter)
    transformers.modeling_utils.get_parameter_device = my_get_parameter_device
    # workaround model.device check end

    in_prompt = "世界羽毛球史上最伟大的球员都有谁？"
    set_random_seed(42)
    inputs = tokenizer([in_prompt,], return_tensors="pt", padding=True)
    inputs.data['input_ids'] = inputs.data['input_ids'].to(torch.int32)
    inputs.data['position_ids'] = inputs.data['position_ids'].to(torch.int32)
    attn_mask = torch.zeros_like(inputs.data['attention_mask'], dtype=torch.float32)
    inputs.data['attention_mask'] = attn_mask.masked_fill_(inputs.data['attention_mask'], -10000.0)

    debug_log('warm up...')
    model.first_token = True
    beg = time.time()
    outputs = model.generate(**inputs, do_sample=False,
                            max_length=2048,
                            num_beams=2)
    end = time.time()
    outputs = outputs.tolist()[0]
    gen_len = model.stat['times']
    debug_log(f'#tokens input={inputs.data["input_ids"].size(1)}, output={gen_len}, cost {end - beg:.2f} sec, avg {(end - beg) / gen_len * 1000:.2f} ms/token, stat {model.stat}')
    model.stat = {
        'init': 0,
        'cost_first': 0,
        'cost_second+': 0,
        'post': 0,
        'times': 0
    }

    debug_log('\ntest...')
    for i, (do_sample, max_length, num_beams) in enumerate(parameters):
        set_random_seed(42)
        inputs = tokenizer([sentences[i],], return_tensors="pt", padding=True)
        inputs.data['input_ids'] = inputs.data['input_ids'].to(torch.int32)
        inputs.data['position_ids'] = inputs.data['position_ids'].to(torch.int32)
        attn_mask = torch.zeros_like(inputs.data['attention_mask'], dtype=torch.float32)
        inputs.data['attention_mask'] = attn_mask.masked_fill_(inputs.data['attention_mask'], -10000.0)
        model.first_token = True
        beg = time.time()
        outputs = model.generate(**inputs, do_sample=do_sample,
                                max_length=max_length,
                                num_beams=num_beams)
        end = time.time()
        outputs = outputs.tolist()[0]
        x = tokenizer.decode(outputs, skip_special_tokens=True)

        print(f"{x}")
        f_content.write(x + '\n\n\n')
        gen_len = model.stat['times']
        debug_log(f'#tokens input={inputs.data["input_ids"].size(1)}, output={gen_len}, cost {end - beg:.2f} sec, avg {(end - beg) / gen_len * 1000:.2f} ms/token, stat {model.stat}')
        model.stat = {
            'init': 0,
            'cost_first': 0,
            'cost_second+': 0,
            'post': 0,
            'times': 0
        }

    f_content.close()
    m = model.exec_net300.get_runtime_model()
    serialize(m, 'ov-special.xml', 'ov-special.bin')

def test_torch(use_ipex=False):
    f_content = open(CONTENT_PATH, 'w')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)
    model.to(device, dtype=torch.bfloat16)
    if use_ipex:
        debug_log('test ipex...')
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.bfloat16)
    else:
        debug_log('test torch...')

    in_prompt = "世界羽毛球史上最伟大的球员都有谁？"
    set_random_seed(42)
    inputs = tokenizer([in_prompt,], return_tensors="pt", padding=True)

    debug_log('warm up...')
    with torch.no_grad(), torch.cpu.amp.autocast():
        beg = time.time()
        outputs = model.generate(**inputs, do_sample=False,
                                max_length=2048,
                                num_beams=2)
        end = time.time()
    outputs = outputs.tolist()[0]
    gen_len = len(outputs) - inputs.data["input_ids"].size(1)
    debug_log(f'#tokens input={inputs.data["input_ids"].size(1)}, output={gen_len}, cost {end - beg:.2f} sec, avg {(end - beg) / gen_len * 1000:.2f} ms/token')

    debug_log('\ntest...')
    for i, (do_sample, max_length, num_beams) in enumerate(parameters):
        set_random_seed(42)
        inputs = tokenizer([sentences[i],], return_tensors="pt", padding=True)
        with torch.no_grad(), torch.cpu.amp.autocast():
            beg = time.time()
            outputs = model.generate(**inputs, do_sample=do_sample,
                                    max_length=max_length,
                                    num_beams=num_beams)
            end = time.time()
        outputs = outputs.tolist()[0]
        x = tokenizer.decode(outputs, skip_special_tokens=True)

        print(f"{x}")
        f_content.write(x + '\n\n\n')
        gen_len = len(outputs) - inputs.data["input_ids"].size(1)
        debug_log(f'#tokens input={inputs.data["input_ids"].size(1)}, output={gen_len}, cost {end - beg:.2f} sec, avg {(end - beg) / gen_len * 1000:.2f} ms/token')
    f_content.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("org_model_path")
    parser.add_argument("ov_model_path")
    parser.add_argument("result_path", nargs='?', default="chatglm.result")
    parser.add_argument("--use", help="set backend: ov,torch,ipex", nargs='?', default='ov')
    args = parser.parse_args()
    OV_MODEL_PATH = args.ov_model_path
    PYTORCH_MODEL_PATH = args.org_model_path
    RESULT_PATH = f'{args.result_path}.{args.use}'
    CONTENT_PATH += f'.{args.use}'
    if args.use == 'torch':
        test_torch()
    elif args.use == 'ov':
        test_ov()
    else:
        test_torch(True)
    f.close()
