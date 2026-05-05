import os
import re

def safe_latex_convert(text):
    # 1. First, fix the broken 'ext{' artifacts caused by previous bad run
    text = text.replace("        ext{", "\\text{")
    text = text.replace("      ext{", "\\text{")
    text = text.replace("    ext{", "\\text{")
    text = text.replace("   ext{", "\\text{")
    text = text.replace("  ext{", "\\text{")
    text = text.replace(" ext{", "\\text{")
    
    # Define math patterns to wrap in $$ if they are in ```text
    def text_block_to_math(match):
        content = match.group(1).strip()
        if any(op in content for op in ["=", "+", "-", "×", "≈", "·"]):
            # Wrap entire block in $$
            return f"$$\n{content}\n$$"
        return match.group(0)

    text = re.sub(r"```text\n(.*?)\n```", text_block_to_math, text, flags=re.DOTALL)

    # Define variables and their LaTeX equivalents
    replacements = {
        "value_loss": r"\text{value\_loss}",
        "vf_loss": r"\text{vf\_loss}",
        "policy_loss": r"\text{policy\_loss}",
        "entropy_loss": r"\text{entropy\_loss}",
        "total_loss": r"\text{total\_loss}",
        "vf_coeff": r"\text{vf\_coeff}",
        "value_pred": r"\text{value\_pred}",
        "value_target": r"\text{value\_target}",
        "target_mean": r"\text{target}_{\text{mean}}",
        "target_std": r"\text{target}_{\text{std}}",
        "value_pred_mean": r"\text{value\_pred}_{\text{mean}}",
        "value_pred_std": r"\text{value\_pred}_{\text{std}}",
        "value_init_bias": r"\text{value\_init\_bias}",
        "normalized_return": r"\text{normalized\_return}"
    }

    # Process each Math block carefully
    def process_block(match):
        c = match.group(1)
        # Apply variable replacements
        for k, v in replacements.items():
            # Use negative lookbehind/lookahead to avoid double wrapping
            c = re.sub(rf"(?<!\\text{{)\b{k}\b(?!}})", v, c)
        
        # Fix symbols
        c = c.replace(" × ", r" \times ")
        c = c.replace(" ≈ ", r" \approx ")
        c = c.replace(" · ", r" \cdot ")
        
        # Wrap Chinese characters in \text{} inside math blocks
        c = re.sub(r'([\u4e00-\u9fff]+)', r'\\text{\1}', c)
        
        return f"$$\n{c}\n$$"

    text = re.sub(r"\$\$\n(.*?)\n\$\$", process_block, text, flags=re.DOTALL)

    # Inline variable fixes
    for k, v in replacements.items():
        # Replace `var_name` or $var\_name$ with $LaTeX$
        text = re.sub(rf"`{k}`", f"${v}$", text)
        # Fix cases where I might have messed up underscores in previous steps
        text = re.sub(rf"\${k.replace('_', r'\_')}\$", f"${v}$", text)

    return text

file_path = "/home/aa/Documents/Obsidian Vault/消融前/02_RL基礎_Returns-GAE與NormalizeReturn.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

new_content = safe_latex_convert(content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Fixed and Updated: {file_path}")
