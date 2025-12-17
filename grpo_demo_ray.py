import re
import torch
from datasets import load_dataset, Dataset, DownloadConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# PEFT 是什么？
# PEFT = Parameter-Efficient Fine-Tuning（参数高效微调）
# 库名：Hugging Face 的 peft 库
# 目的：用更少的参数和内存微调大模型


SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{resoning}      # 占位符，会被实际推理内容替换
</reasoning>
<answer>
{answer}        # 占位符，会被实际答案替换
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """
    从 XML 格式的文本中提取 <answer> 标签内的内容。
    
    工作原理：
    1. 第一步：使用 split("<answer>") 将文本分为两部分
       - 第一部分：<answer> 之前的所有内容（包括 <reasoning> 等）
       - 第二部分：<answer> 之后的所有内容（包括答案和 </answer> 标签）
       - 使用 [-1] 取最后一部分，即 <answer> 之后的所有内容
       - 注意：split() 会移除分隔符本身，所以结果中不包含 "<answer>"
    
    2. 第二步：在第一步结果的基础上，使用 split("</answer>") 再次分割
       - 第一部分：</answer> 之前的内容（即答案内容）
       - 第二部分：</answer> 之后的内容（如果有其他文本）
       - 使用 [0] 取第一部分，即答案内容
       - 注意：split() 会移除分隔符本身，所以结果中不包含 "</answer>"
    
    3. 第三步：使用 strip() 去除答案首尾的空白字符（空格、换行等）
    
    示例：
        输入: "<reasoning>思考过程</reasoning><answer>42</answer>"
        第一步后: "42</answer>"
        第二步后: "42"
        第三步后: "42" (去除空白)
        输出: "42"
    
    参数:
        text: 包含 XML 格式的文本，应该包含 <answer>...</answer> 标签
    
    返回:
        提取出的答案字符串（已去除首尾空白）
    """
    # 第一步：根据 <answer> 标签分割，取最后一部分（<answer> 之后的所有内容）
    # split() 会移除分隔符本身，所以结果中不包含 "<answer>"
    answer = text.split("<answer>")[-1]
    
    # 第二步：根据 </answer> 标签分割，取第一部分（</answer> 之前的内容）
    # split() 会移除分隔符本身，所以结果中不包含 "</answer>"
    answer = answer.split("</answer>")[0]
    
    # 第三步：去除首尾空白字符（空格、换行符等）
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """
    从使用 "###" 作为分隔符的文本中提取答案。
    
    输入样式示例：
        "这是一个问题###42"
        "计算 2+3###5"
        "问题内容###答案内容, 带逗号$符号"
    
    工作原理：
        1. 检查文本中是否包含 "###" 分隔符
        2. 如果不存在，返回 None
        3. 如果存在，使用 split("###") 分割文本
        4. 取第二部分 [1]（### 之后的内容）
        5. 去除首尾空白字符
        6. 移除逗号和美元符号
    
    参数:
        text: 包含 "###" 分隔符的文本，格式为 "问题###答案"
    
    返回:
        提取出的答案字符串（已去除逗号和美元符号），如果文本中没有 "###" 则返回 None
    """
    # 第一步：检查文本中是否包含 "###" 分隔符
    # "###" not in text 表示：如果 "###" 不在 text 中
    if "###" not in text:
        return None  # 如果没有分隔符，返回 None
    
    # 第二步：提取答案并清理格式
    # 这是一个链式调用（Method Chaining），从左到右依次执行：
    # 1. text.split("###")      -> 将文本按 "###" 分割成列表
    # 2. [1]                    -> 取列表的第二个元素（索引为1，即 ### 之后的部分）
    # 3. .strip()               -> 去除首尾空白字符（空格、换行等）
    # 4. .replace(",", "")      -> 将所有的逗号替换为空字符串（即删除逗号）
    # 5. .replace("$", "")      -> 将所有的美元符号替换为空字符串（即删除美元符号）
    return text.split("###")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split: str = "train") -> Dataset:
    """
    从 Hugging Face 加载 GSM8K 数据集，并将其转换为适合 GRPO 训练的格式。
    
    GSM8K 数据集简介：
        - GSM8K = Grade School Math 8K（小学数学 8K 题）
        - 包含 8000+ 道小学数学文字题
        - 用于测试和训练模型的数学推理能力
        - 由 OpenAI 创建并发布在 Hugging Face 上
    
    数据转换过程：
        原始数据格式（GSM8K 数据集）：
            {
                'question': 'Tom has 3 apples and buys 2 more. How many apples?',
                'answer': 'He starts with 3 and buys 2 more, so 3 + 2 = 5. #### 5'
            }
        
        转换为对话格式（用于模型训练）：
            {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},  # 系统提示，定义输出格式
                    {'role': 'user', 'content': '示例问题...'},    # 1-shot 示例问题（可选）
                    {'role': 'assistant', 'content': '示例回答...'}, # 1-shot 示例回答（可选）
                    {'role': 'user', 'content': '实际问题'}        # 当前要回答的问题
                ],
                'answer': '5'  # 从原始答案中提取的最终答案（去除推理过程）
            }
    
    1-shot Prompting（少样本提示）：
        - 代码中包含一个示例问题和回答（中间两条消息）
        - 这可以帮助模型理解期望的输出格式
        - 如果不需要 1-shot，可以注释掉中间两条消息
    
    参数:
        split: 数据集分割，可选值：
            - "train": 训练集（约 7473 题）
            - "test": 测试集（约 1000 题）
    
    返回:
        Dataset 对象，包含转换后的数据：
            - 'prompt': 对话格式的提示（列表，包含多个消息）
            - 'answer': 提取出的最终答案（字符串）
    
    代码详解:
        1. load_dataset('openai/gsm8k', 'main')[split]
           - 从 Hugging Face 加载 GSM8K 数据集
           - 提取指定的分割（train 或 test）
        
        2. data.map(lambda x: {...})
           - 对数据集中的每条数据进行转换
           - lambda x: x 是数据集中的一条记录（字典）
           - 返回转换后的新字典
        
        3. 'prompt' 字段构建：
           - system: 系统提示，告诉模型输出格式
           - user (示例): 1-shot 示例问题（可选）
           - assistant (示例): 1-shot 示例回答，使用 XML_COT_FORMAT.format() 格式化（可选）
           - user (实际): 当前要回答的问题 x['question']
        
        4. 'answer' 字段：
           - 使用 extract_hash_answer() 从原始答案中提取最终答案
           - 原始答案格式: "推理过程... #### 最终答案"
           - 提取后: "最终答案"（去除推理过程和符号）
    
    示例:
        >>> data = get_gsm8k_questions("train")
        >>> example = data[0]
        >>> print(example['prompt'])
        [
            {'role': 'system', 'content': 'Respond in the following format:...'},
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': '<reasoning>...</reasoning><answer>7</answer>'},
            {'role': 'user', 'content': 'Tom has 3 apples...'}
        ]
        >>> print(example['answer'])
        '5'
    """
    
    # 第一步：从 Hugging Face 加载 GSM8K 数据集
    # 直接在 OpenAI 的 Hugging Face 仓库下载这个数据集
    # split 参数指定是 train（训练集）还是 test（测试集）部分则通过split参数控制
    # load_dataset() 返回 DatasetDict，[split] 提取指定的分割（train 或 test）
    # data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    download_cfg = DownloadConfig(local_files_only=True)
    data = load_dataset(
        path="/home/liangshurui/agentic_rag/data/gsm8k",
        name="main",
        split=split,
    )  # type: ignore
    
    # 第二步：使用 map() 函数转换每条数据
    # lambda x: x 是数据集中的一条记录，包含 'question' 和 'answer' 字段
    data = data.map(lambda x: {  # type: ignore
        # 构建对话格式的提示（prompt）
        'prompt': [
            # 系统消息：定义模型应该使用的输出格式
            {'role': 'system', 'content': SYSTEM_PROMPT},
            
            # # 1-shot 示例：用户问题（可选，用于少样本学习）
            # # 这个示例帮助模型理解期望的输出格式
            # {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            
            # # 1-shot 示例：助手回答（可选，展示期望的输出格式）
            # # 使用 XML_COT_FORMAT.format() 格式化输出，包含推理过程和答案
            # # 注意：参数名是 'resoning'（注意拼写），因为 XML_COT_FORMAT 模板中用的是 {resoning}
            # {'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #     resoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
            #     answer="7"
            # )},
            
            # 实际要回答的问题：从数据集中的 'question' 字段获取
            {'role': 'user', 'content': x['question']}
        ],
        
        # 调用之前的 extract_hash_answer() 函数, 提取最终答案：从原始答案中提取（去除推理过程和 ### 符号）
        # 原始格式: "推理过程... #### 最终答案"
        # 提取后: "最终答案"
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    
    return data  # type: ignore

dataset = get_gsm8k_questions()

# Reward Function 
# 总共可以分为4部分
# 
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    1. 答案正确奖励
    奖励函数：根据模型生成的答案是否正确来计算奖励。
    
    这是 GRPO 训练中的核心函数，用于评估模型生成的质量。
    抽取<answer>中的内容作为模型的answer，然后和真正的grund truth进行比较
    如果模型提取的答案与正确答案完全匹配，给予 2.0 的奖励，否则给予 0.0。
    
    参数:
        prompts: 提示列表
                数据结构示例：
                [
                    [  # 第一个样本的提示
                        {'role': 'system', 'content': 'Respond in the following format:...'},
                        {'role': 'user', 'content': 'Tom has 3 apples and buys 2 more. How many apples?'}
                    ],
                    [  # 第二个样本的提示
                        {'role': 'system', 'content': 'Respond in the following format:...'},
                        {'role': 'user', 'content': 'What is 2 + 2?'}
                    ]
                ]
        
        completions: 模型生成的完成内容列表
                     数据结构示例：
                     [
                         [  # 第一个样本的生成结果
                             {'role': 'assistant', 'content': '<reasoning>...</reasoning><answer>5</answer>'}
                         ],
                         [  # 第二个样本的生成结果
                             {'role': 'assistant', 'content': '<reasoning>...</reasoning><answer>4</answer>'}
                         ]
                     ]
        
        answer: 正确答案列表（字符串列表）
                数据结构示例：
                ['5', '4', '7', ...]
        
        **kwargs: 其他可选参数（当前未使用）
    
    返回:
        list[float]: 奖励分数列表，每个元素对应一个样本
                     数据结构示例：
                     [2.0, 0.0, 2.0, ...]
                     - 2.0: 答案正确
                     - 0.0: 答案错误
    
    工作流程:
        1. 从 completions 中提取模型生成的文本内容
        2. 从 prompts 中提取问题（最后一个用户消息）
        3. 使用 extract_xml_answer() 从生成的文本中提取答案
        4. 将提取的答案与正确答案进行比较
        5. 返回奖励分数列表
    
    示例:
        >>> prompts = [[
        ...     {'role': 'system', 'content': '...'},
        ...     {'role': 'user', 'content': 'What is 2+3?'}
        ... ]]
        >>> completions = [[
        ...     {'role': 'assistant', 'content': '<reasoning>2+3=5</reasoning><answer>5</answer>'}
        ... ]]
        >>> answer = ['5']
        >>> correctness_reward_func(prompts, completions, answer)
        [2.0]  # 答案正确，返回 2.0
    """
    # 第一步：从 completions 中提取模型生成的文本内容
    # completions 的结构: [[{'role': 'assistant', 'content': '...'}], ...]
    # completion[0] 获取第一个生成结果（通常只有一个）
    # ['content'] 获取文本内容
    # 结果: ['<reasoning>...</reasoning><answer>5</answer>', ...]
    responses = [completion[0]['content'] for completion in completions]
    
    # 第二步：从 prompts 中提取问题
    # prompts[0] 获取第一个样本的提示列表
    # [-1] 获取最后一个消息（通常是用户的问题）
    # ['content'] 获取消息内容
    # 结果: 'Tom has 3 apples and buys 2 more. How many apples?'
    q = prompts[0][-1]['content']
    
    # 第三步：从模型生成的响应中提取答案
    # 使用 extract_xml_answer() 函数从 XML 格式的文本中提取 <answer> 标签内的内容
    # responses 示例: ['<reasoning>...</reasoning><answer>5</answer>', ...]
    # extracted_responses 结果: ['5', '4', ...]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # 第四步：打印调试信息（用于训练过程中的监控）
    # 显示问题、正确答案、模型完整响应、提取的答案
    print('-'*20, 
          f"Question:\n{q}", 
          f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    
    # 第五步：计算奖励分数
    # 使用列表推导式和 zip() 函数同时遍历提取的答案和正确答案
    # 如果提取的答案 (r) 等于正确答案 (a)，返回 2.0，否则返回 0.0
    # zip(extracted_responses, answer) 将两个列表配对：
    #   - 如果 extracted_responses = ['5', '4'], answer = ['5', '3']
    #   - zip 后: [('5', '5'), ('4', '3')]
    # 结果: [2.0, 0.0]  # 第一个正确，第二个错误
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def simple_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    一个“宽松版”的正确性奖励：
    只要模型完整输出中包含正确答案的字符串，就给 1.0，否则给 0.0。
    不要求必须有 <answer> 标签，先用它保证 reward 不是一直为 0。
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards: list[float] = []
    for resp, a in zip(responses, answer):
        gt = str(a).strip()
        rewards.append(0.25 if gt and gt in resp else 0.0)
    return rewards

# 为什么取prompt的时候要取0？
# 为什么这样写是“对的”？
# 1️⃣ prompts[0]
# 当前 reward 调用中，只有 1 个问题，所以它在 prompts[0]
# 2️⃣ completion[0]
# 每个 completion 是一次生成，当前只生成 1 条 assistant 消息，所以在 completion[0]

def int_reward_func(completions, **kwargs) -> list[float]:
    """
    2. 对于模型生成的答案是数字的奖励
    只要模型生成的答案是个数字，就奖励0.5，鼓励模型生成数字答案
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    3. 对于模型生成的答案格式正确的奖励（严格版）
    只要模型生成的的格式答案是（*严格）正确的，就给一个奖励
    
    ^表示匹配必须从字符串的开头开始。
    $表示匹配必须到字符串的结尾结束。
    组合使用 ^...$意味着整个字符串必须完全符合这个模式，不能多也不能少
    
    \n匹配一个换行符（Newline），用来确保标签独占一行并格式整齐
    .（点号）：是一个通配符，匹配除换行符（\n）之外的任意单个字符
    *（星号）：是一个量词，表示匹配前面的元素（即点号 .）零次或多次。这意味着它可以匹配任意长度的字符串（包括空字符串）
    ?（问号）：在这里紧跟在 *之后，它改变了匹配模式。*默认是“贪婪的”，会匹配尽可能多的字符。而 *?组合则表示非贪婪匹配（或叫懒惰匹配），即匹配尽可能少的字符，只要能使整个表达式匹配成功即可
    非贪婪模式的匹配一般来说，更安全，更高效
    
    跨行匹配
    re.DOTALL:  re.DOTALL = Single-line mode 把整个字符串当成“一行”来看，即：. 也可以匹配 \n
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    4. 软格式奖励（宽松版）
    只要模型生成的文本中“某处”包含一对 <reasoning>...</reasoning> 和 <answer>...</answer>，
    就给 0.5 奖励，不再强制要求标签必须从字符串开头开始。
    """

    # 使用更宽松的正则：允许标签前后有任意内容，内部也可以跨多行
    pattern = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]

    # 调试输出（如不需要可注释掉）
    print(f"[soft_format_reward_func] matches: {matches}")
    print(f"[soft_format_reward_func] rewards: {rewards}")
    return rewards

def count_xml(text) -> float:
    """
    5. 计数奖励
    对 <reasoning> 和 <answer> 标签的出现情况给连续、平滑的奖励。
    这一版和你贴的“原来版本”保持一致，只做了括号位置的修正。
    """
    count = 0.0

    # <reasoning>\n 出现一次
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    # \n</reasoning>\n 出现一次
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    # \n<answer>\n 出现一次
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # 严格惩罚 </answer> 之后多余的输出
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    # \n</answer> 出现一次
    if text.count("\n</answer>") == 1:
        count += 0.125
        # 允许 </answer> 后有 1 个额外字符（通常是一个换行）
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    5. xml计数奖励 Part2 
    对奖励<reasoning>和<answer>标签的奖励
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 根据不同的模型名字 设置输出路径
if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
elif "Qwen" in model_name:
    output_dir = "outputs/Qwen-1.5B-GRPO"
    run_name = "Qwen-1.5B-GRPO-gsm8k"
else:
    raise ValueError(f"Unsupported model: {model_name}")
        
# 训练参数
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1, # 每个设备上的训练批次大小
    gradient_accumulation_steps=4, # 梯度累积步数 相当于 实际的batch_size = per_device_train_batch_size * gradient_accumulation_steps 
    num_generations=8, # 原来的值:16 对于GRPO，对同一个 prompt，一次采样多少个候选 completion，用来做“组内相对比较（relative advantage）”
    generation_batch_size=8,
    max_prompt_length=256, # 最大提示长度
    max_completion_length=384, # 原来是786
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)


# 模型结构（参数）
# x
# │
# ├─ LayerNorm
# │
# ├─ Self-Attention
# │   ├─ q_proj
# │   ├─ k_proj
# │   ├─ v_proj
# │   └─ o_proj
# │
# ├─ 残差
# │
# ├─ LayerNorm
# │
# └─ MLP / FFN
#     ├─ gate_proj
#     ├─ up_proj
#     └─ down_proj
# ------------------------------------------------------------
# 拼接后的向量需要：映射回模型隐藏维度 d_model，融合各个 head 的信息
# 所以有：
# Output = Attention(Q,K,V) * W_O

# Lora配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=64, # Lora缩放系数, 实际缩放= r * lora_alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", # Attention中的4个线性层
                    "up_proj", "down_proj", "gate_proj" # MLP中的3个线性层
                    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# 模型实例化
# 尝试使用 FlashAttention2，如果不可用则回退到默认实现
try:
    import flash_attn
    attn_implementation = "flash_attention_2"  # 使用flash_attention_2加速注意力计算
    print("使用 FlashAttention2 加速注意力计算")
except ImportError:
    attn_implementation = None  # 回退到默认实现
    print("FlashAttention2 未安装，使用默认注意力实现")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map=None, # 自动分配设备
).to("cuda")

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 训练器实例化
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        simple_correctness_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset
)

# 打印 CUDA 和 GPU 信息
print("=" * 80)
print("CUDA 和 GPU 信息")
print("=" * 80)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 设备: {torch.cuda.current_device()}")
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("\n所有 GPU 详细信息:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    已分配内存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"    缓存内存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"    计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("警告: CUDA 不可用，将使用 CPU 训练（速度会很慢）")
print("=" * 80)
print()

trainer.train()