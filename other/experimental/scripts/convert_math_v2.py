import os
import re

def convert_to_latex(text):
    # 1. Convert text code blocks that look like math to $$ block math $$
    def replace_code_block(match):
        content = match.group(1).strip()
        
        # Check if it looks like math: has symbols and not too much Chinese
        has_math_symbols = any(op in content for op in ["=", "+", "-", "*", "/", "^", "≈", "·", ">", "<"])
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        
        # If it's mostly English/Math and has math symbols, or if it's a specific formula known to be math
        if has_math_symbols and (chinese_chars < len(content) * 0.3 or "value_loss" in content or "return" in content):
            # Simple conversion logic
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                # Replace symbols
                line = line.replace(" * ", " \\cdot ")
                line = line.replace(" × ", " \\times ")
                line = line.replace(" ≈ ", " \\approx ")
                line = line.replace(" · ", " \\cdot ")
                
                # Functions
                line = re.sub(r"(\w+)\(", r"\\operatorname{\1}(", line)
                
                # Common variables to \text or special formatting
                vars_to_text = ["target", "prediction", "return", "steps", "value", "loss", "policy", "vf", "entropy", "coeff", "gain", "bias"]
                for v in vars_to_text:
                    line = re.sub(rf"\b{v}\b", rf"\\text{{{v}}}", line)
                
                # Handle underscores: value_pred -> \text{value\_pred} or value_{pred}
                line = line.replace("_", "\\_")
                
                # Handle fractions but avoid simple mean/std in text
                if "/" in line and any(c in line for c in "+-=*"):
                    line = re.sub(r"(\w+)\s*/\s*(\w+)", r"\\frac{\1}{\2}", line)
                
                new_lines.append(line)
            
            return "$$\n" + "\n".join(new_lines) + "\n$$"
        
        return f"```text\n{content}\n```"

    # Use a more careful regex for code blocks to avoid overlapping
    text = re.sub(r"```text\n(.*?)\n```", replace_code_block, text, flags=re.DOTALL)

    # 2. Convert specific inline patterns
    # V(s) ≈ b -> $V(s) \approx b$
    text = re.sub(r"`V\(s\) ≈ (.*?)`", r"$V(s) \approx \1$", text)
    text = re.sub(r"`gain=(.*?)`", r"$\text{gain}=\1$", text)
    text = re.sub(r"`bias=(.*?)`", r"$\text{bias}=\1$", text)
    
    # Simple variables in backticks
    inline_vars = ["V(s)", "value_pred", "value_target", "vf_loss", "vf_coeff", "value_head", "VE", "MSE", "ReLU(x)"]
    for var in inline_vars:
        safe_var = var.replace("(", "\\(").replace(")", "\\)")
        escaped_var = var.replace("_", "\\_")
        text = re.sub(f"`{safe_var}`", f"${escaped_var}$", text)

    # Specific ranges like -10 ~ 200
    text = re.sub(r"`(-?\d+)\s*~\s*(-?\d+)`", r"$\1 \\sim \2$", text)

    return text

dir_path = "/home/aa/Documents/Obsidian Vault/消融前"
for filename in os.listdir(dir_path):
    if filename.endswith(".md"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # We need to revert my previous mess if possible or just run on original-ish content
        # Since I don't have a backup, I'll try to make the conversion idempotent or better.
        # Actually, let's try to fix the mess by running the improved logic.
        
        new_content = convert_to_latex(content)
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated: {filename}")
        else:
            print(f"No changes: {filename}")
