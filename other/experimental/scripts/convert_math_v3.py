import os
import re

def final_latex_fix(text):
    # 1. Fix variables with underscores to be consistent \text{var\_name}
    vars_list = [
        "value_loss", "vf_loss", "policy_loss", "entropy_loss", "total_loss",
        "vf_coeff", "value_pred", "value_target", "normalized_return", "raw_return",
        "value_init_bias", "target_mean", "target_std", "value_pred_mean", "value_pred_std"
    ]
    
    # Pre-process math blocks to handle Chinese and complex patterns
    def process_math_block(match):
        content = match.group(1).strip()
        
        # Replace common RL variables
        for v in vars_list:
            # Handle specific cases like target_mean -> \text{target}_{\text{mean}}
            if "_" in v:
                base, sub = v.split("_", 1)
                new_v = f"\\text{{{base}}}_{{\\text{{{sub}}}}}"
                content = re.sub(rf"\b{v}\b", new_v, content)
            else:
                content = re.sub(rf"\b{v}\b", f"\\text{{{v}}}", content)

        # Fix remaining isolated underscores in names
        content = re.sub(r"([a-z]+)_([a-z]+)", r"\\text{\1}_{\\text{\2}}", content)

        # Handle Chinese characters in math blocks: wrap in \text{}
        # This regex finds sequences of Chinese characters and spaces/colons
        content = re.sub(r'([\u4e00-\u9fff][\u4e00-\u9fff\s：:]*)', r'\\text{\1}', content)
        
        # Fix mathematical symbols
        content = content.replace(" × ", " \\times ")
        content = content.replace(" ≈ ", " \\approx ")
        content = content.replace(" · ", " \\cdot ")
        content = content.replace("mean/std", "\\frac{\\operatorname{mean}}{\\operatorname{std}}")
        
        # Cleanup nested \text{\text{...}}
        content = re.sub(r"\\text\{\\text\{", r"\\text{", content)
        content = content.replace("}}", "}")
        
        return f"$$\n{content}\n$$"

    # Convert remaining text blocks that are clearly math to math blocks first
    def text_to_math(match):
        content = match.group(1).strip()
        if any(op in content for op in ["=", "+", "-", "×", "≈", "·"]):
            return process_math_block(match)
        return match.group(0)

    # First, handle existing math blocks
    text = re.sub(r"\$\$\n(.*?)\n\$\$", process_math_block, text, flags=re.DOTALL)
    
    # Second, handle text blocks that should be math
    text = re.sub(r"```text\n(.*?)\n```", text_to_math, text, flags=re.DOTALL)

    # Inline fixes
    for v in vars_list:
        escaped_v = v.replace("_", "\\_")
        # If it's $vf_loss$, change to $\text{vf\_loss}$
        text = re.sub(rf"\${v.replace('_', '\\_')}\$", f"$\\text{{{escaped_v}}}$", text)
        # If it's $vf\_loss$, fix it
        text = re.sub(rf"\${v.replace('_', r'\_')}\$", f"$\\text{{{escaped_v}}}$", text)

    return text

file_path = "/home/aa/Documents/Obsidian Vault/消融前/02_RL基礎_Returns-GAE與NormalizeReturn.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

new_content = final_latex_fix(content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Deeply updated: {file_path}")
