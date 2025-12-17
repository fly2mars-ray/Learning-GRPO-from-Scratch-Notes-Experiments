# Learning-GRPO-from-Scratch: Notes & Experiments

🧠 This repository records my **hands-on learning notes and experimental results** while studying **GRPO (Group Relative Policy Optimization)** for large language models.

The implementation and experiments are based on the excellent demo repository:

🔗 **Reference script (gist)**  
https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py

I reimplemented and ran the full training pipeline locally, analyzed reward behaviors, formatting constraints, and WandB profiling logs, and summarized key observations from a learner’s perspective.

This repo is intended as:
- a **personal GRPO learning log**
- a **lightweight reproduction record**
- a **concept + experiment–driven note**, not a production-ready framework

---

## 📌 What is GRPO (in one paragraph)

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning method proposed to replace PPO-style value modeling with **group-based relative rewards**.

Instead of training a value function, GRPO:
- samples **multiple completions per prompt**
- assigns rewards using **relative comparison inside the group**
- optimizes the policy directly based on **group-normalized advantages**

This makes GRPO especially suitable for:
- reasoning tasks (e.g. GSM8K)
- format-constrained outputs
- small / medium-scale models without a strong critic

---

## 🔧 What I Did in This Repo

Compared to the original demo, my work focuses on **running, observing, and understanding**, rather than extending the algorithm.

### ✅ 1. Re-ran the full GRPO training pipeline

- **Model**: Qwen-1.5B  
- **Task**: GSM8K  
- **Optimizer**: GRPO-style policy update  
- **Setup**: multiple reward functions enabled simultaneously  

---

### ✅ 2. Verified format-constrained generation

I confirmed that the model successfully learned **XML-style structured outputs**, for example:

```xml
<reasoning>
...
</reasoning>
<answer>
19
</answer>
```
Reward extraction and correctness checking work as expected once the formatting stabilizes.

***

✅ 3. Observed a key phenomenon:

“Format learned, reasoning not improved”

One important learning outcome:

⚠️ The model quickly learns the output format, but reasoning quality does not significantly improve.

Typical observations:
- XML tags are produced perfectly
- Answers are often correct
- Reasoning remains shallow or template-like

This is likely due to:
- small model size (1.5B)
- weak reward signal on reasoning depth
- GRPO optimizing relative correctness, not reasoning quality

This matches the intuition that:

GRPO is good at enforcing structure and relative correctness, but not sufficient alone for deep reasoning emergence.

***

📊 WandB Profiling & Reward Time Analysis

I logged detailed profiling metrics for each reward function and generation step.

Key tracked components include:
- transformers.generate
- xmlcount_reward_func
- strict_format_reward_func
- soft_format_reward_func
- simple_correctness_reward_func
- int_reward_func

Observations
- Generation dominates total time cost
- Reward functions are cheap individually but accumulate over group samples
- Formatting rewards converge early and become nearly constant
- Later training steps show reduced reward variance but no clear reasoning gain

📈 Profiling example (placeholder):

![WandB profiling panels](assets/p1.png)


![Correct but shallow reasoning](assets/p2.png)


![Early-stage failure example](assets/p3.png)

***

🧪 Example Outputs & Failure Modes

✔ Correct but shallow reasoning

The model outputs valid reasoning blocks but mostly performs surface-level arithmetic, rather than genuine multi-step abstraction.

***

🙏 Acknowledgements
- Original GRPO demo by willccbb
- GSM8K dataset
- Qwen model family
- WandB for experiment tracking

This repository is for learning and discussion only.

***

📬 Notes

This repository is not:
- an official GRPO implementation
- a benchmark result
- an optimized training recipe

It is:
- a faithful reproduction
- an honest learning record
- a reference for others starting with GRPO
